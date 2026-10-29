# HierarchicalSpec Structure

This document records the implemented three-stage speculative-decoding design. Keep it synchronized with the runtime and benchmark code whenever ownership, state semantics, CLI behavior, or supported topology changes.

## Runtime Topology

The request path is:

```text
A: Domino DFlash drafter -> B: independent intermediate target
       (K-1 refinement rounds; final A-only round) -> C: final target block verification
```

`HierarchicalSpecWorker` owns the cascade. `CascadeCoordinator` owns `IntermediateStage` (B) and the injected `DFlashWorkerV2` inner kernel. DFlashWorkerV2 is reused only for the high-performance A-to-B execution path; B is not the scheduler draft model, draft cache, or draft-only lifecycle. C is the only stage allowed to advance committed output.

## State and Cache Ownership

Each stage has an independent model runner and KV pool. A and B may run ahead with provisional tokens. C verifies the accumulated B block, commits its accepted prefix plus recovery token, and causes A/B provisional suffixes to be rolled back and resynchronized. Rejected tokens must not remain in stage caches, sequence state, or request metadata. The coordinator tracks committed length, pending token ids/rounds, per-stage cache/checkpoint metadata, and reserved RNG state fields.

## Window Geometry

`--speculative-dflash-block-size L` is the inner A-to-B DFlash width and resolves to `speculative_num_draft_tokens`; explicit conflicting values are rejected. Each non-final A-to-B round contributes up to `L` tokens: up to `L-1` accepted A proposals and one B bonus that anchors the next inner round. In the final round A contributes its `L-1` proposals directly; B does not verify them, and C generates the outer-window bonus. With `K=--speculative-cascade-rounds`, C therefore receives a fixed width of `K*L`: one existing C anchor plus up to `K*L-1` pending tokens. For `K=3,L=8`, C receives 24 positions and may commit up to 24 new tokens including its bonus token.

## Supported Contract

The current v1 path is linear and greedy (`temperature=0`, `top_k=1`), uses one tokenizer/vocabulary, and supports a single GPU/fixed parallel configuration. A/B/C must share one device; independent stage placement and cross-device transfers are not implemented. CUDA Graphs remain disabled by default for HierarchicalSpec. Setting `SGLANG_ENABLE_HIERARCHICAL_A_CUDA_GRAPH=1`, `SGLANG_ENABLE_HIERARCHICAL_B_CUDA_GRAPH=1`, or `SGLANG_ENABLE_HIERARCHICAL_C_CUDA_GRAPH=1` enables the corresponding A, B, or C graph independently. A and B use the inner width `L`; C uses the outer width `K*L` and captures FULL hidden states needed by the A handoff. The three roles capture separate graph instances for their own model runners, KV pools, hidden-state buffers, and fixed widths. Any unsupported graph capture or replay falls back to the corresponding eager path where a role-specific fallback is installed.

## Quantization and Benchmark

The three-way wrapper is `benchmark/hierarchical/run_sglang_benchmark.sh`; its driver is `benchmark_three_way.py`. By default it runs A+B+C, C-only, and A+C with matched prompts and request settings; `BENCHMARK_MODES` selects a subset such as `abc`. The C-to-B KV-copy experiment fixes this to `abc`, because it needs only matched copy-off and copy-on HierarchicalSpec runs. `DTYPE` applies to every mode and defaults to BF16. Intermediate `AWQ` is normalized to `awq_marlin`, matching SGLang's BF16-capable automatic AWQ path; plain `awq` may reject BF16. Intermediate quantization values `unquant` and `none` mean that no intermediate quantization override is passed; for an unquantized target this loads B in its checkpoint-native dtype. The ABC benchmark keeps CUDA Graphs disabled by default for an eager baseline; set `HIERARCHICAL_A_CUDA_GRAPH=1`, `HIERARCHICAL_B_CUDA_GRAPH=1`, and/or `HIERARCHICAL_C_CUDA_GRAPH=1` (and leave `DISABLE_CUDA_GRAPH=0`) to enable the corresponding role graph path.

Reports are written under `benchmark/hierarchical/outputs/`. Record graph mode, K/L, acceptance length, verification counts, throughput, latency, and KV-capacity effects when comparing configurations. The B-only graph run should report B graph replays with zero fallbacks, an eager A draft path, and an eager C target path.

For apples-to-apples comparisons with HierarchicalSpec, set
`DISABLE_OVERLAP_SCHEDULE=1` in `run_sglang_benchmark.sh` (or pass
`--disable-overlap-schedule` to `benchmark_three_way.py`). The switch is applied
to ABC, AC, and C-only server launches; it disables the scheduler's cross-batch
CPU/GPU overlap without changing the speculative algorithm or CUDA Graph role
settings.

For HierarchicalSpec A and B, the graph captures the model forward and still
uses FlashInfer's direct `fast_prefill_plan`. Both workers reserve storage for
the complete outer window, so their host `seq_lens_cpu` can be an allocation
upper bound while each inner round has a shorter live prefix. Before an A/B
replay, the fast-plan path uses the live device prefix plus the fixed inner
query width as its host planning length, rather than the allocation bound. C
uses its outer verify layout and can now opt into its own graph with
`SGLANG_ENABLE_HIERARCHICAL_C_CUDA_GRAPH=1`; its graph captures FULL hidden
states for the A handoff. The dynamic A/B path performs one live-prefix
device-to-host metadata read per replay instead of rebuilding the regular
planner, preventing attention from reading provisional or uninitialized KV
rows.

When `SGLANG_ENABLE_HIERARCHICAL_C_KV_COPY=1`, the outer boundary copies the
complete fixed C verify window into B's corresponding physical slots. The
implementation resolves each worker's underlying KV data pool through its KV
allocator (the worker memory-pool API returns a request pool and allocator pair).
The copy requires ordinary contiguous, unquantized NHD KV buffers with identical
row strides in B and C. This is the optimized cache handoff: B verifies only the
first `K-1` inner rounds, the final A proposals go directly to C, and C's `K*L`
rows replace B's reserved window at the outer boundary. The copy includes
rejected suffix rows; B's committed sequence length controls which rows are
subsequently read. The copy is launched on a dedicated CUDA stream after C
completes, while the next A round starts immediately from the already-published
C hidden states. The next B round waits on the copy event only after A has
finished. The timing reporter separates `c_to_b_kv_copy_launch`, asynchronous
`c_to_b_kv_copy_async`, `b_wait_for_c_to_b_kv_copy`, and `b_window_roll`.

With the handoff enabled, B's fixed window advances as a committed-prefix plus
new-tail allocation: rows that leave the speculative window are already valid
committed KV and remain allocated, while only the newly exposed tail is
allocated. The first decode or any rollback/regression uses the conservative
trim-and-rebuild path. Request release and cache reset synchronize an outstanding
copy before returning physical slots to the allocator.
Use `benchmark/hierarchical/run_c_kv_copy_experiment.sh` to run matched
baseline and copy-enabled cases in separate output directories and compare
`accept_length`, `c_to_b_kv_copy_async`, and end-to-end time. The script writes the
comparison to `c_kv_copy_comparison.md` and `.json` under the experiment root.

The wrapper can enable `SGLANG_LOG_HIERARCHICAL_TIMING=1`. During the
A+B+C run, CUDA events split each outer decode into B cache preparation and
rollback, candidate-buffer setup/finalization, every A draft and B verification
round, and C verification plus hidden-state publication. A
single synchronization of the main stream at the end of the outer decode makes
the linear event durations available; the asynchronous copy event is consumed
by the next B1 wait before its duration is reported. Logs report interval
averages and percentages, separate
`a_rounds_total` and `b_rounds_total` aggregates, plus an unattributed
host/launch/sync residual. Set `HIERARCHICAL_TIMING=0` to measure uninstrumented
throughput or `HIERARCHICAL_TIMING_INTERVAL=1` to print every outer decode batch.

The same environment switch is also honored by standalone DFlash A+C runs.
Those logs are prefixed `DFlash A+C timing` and split each decode into the A
draft, C verify setup, C model verify, acceptance/post-processing, hidden-state
publication, result finalization, and host/launch/synchronization residual.
When HierarchicalSpec embeds DFlash for its A-to-B rounds, timing is disabled
on that nested worker so only the outer `HierarchicalSpec timing` record is
emitted.

## Code Map and Tests

- `python/sglang/srt/speculative/hierarchical_worker.py`: stage construction, execution, cache planning, and synchronization.
- `python/sglang/srt/speculative/hierarchical_types.py`: request state and greedy commit/rollback helpers.
- `python/sglang/srt/speculative/hierarchical_config.py`: window geometry, device-map parsing, and request validation.
- `python/sglang/srt/arg_groups/speculative_hook.py`: validation and CLI resolution.
- `test/registered/unit/speculative/test_hierarchical_types.py`: geometry, acceptance, rollback, and ownership tests.

Run focused tests with `python -m pytest test/registered/unit/speculative/test_hierarchical_types.py` when pytest is available. GPU benchmark validation is separate and hardware-dependent.
