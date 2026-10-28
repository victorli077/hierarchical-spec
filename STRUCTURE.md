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

The current v1 path is linear and greedy (`temperature=0`, `top_k=1`), uses one tokenizer/vocabulary, and supports a single GPU/fixed parallel configuration. A/B/C must share one device; independent stage placement and cross-device transfers are not implemented. CUDA Graphs are disabled for HierarchicalSpec. C-only and A+C may use them unless a benchmark disables them for all modes.

## Quantization and Benchmark

The three-way wrapper is `benchmark/hierarchical/run_sglang_benchmark.sh`; its driver is `benchmark_three_way.py`. By default it runs A+B+C, C-only, and A+C with matched prompts and request settings; `BENCHMARK_MODES` selects a subset such as `abc`. The C-to-B KV-copy experiment fixes this to `abc`, because it needs only matched copy-off and copy-on HierarchicalSpec runs. `DTYPE` applies to every mode and defaults to BF16. Intermediate `AWQ` is normalized to `awq_marlin`, matching SGLang's BF16-capable automatic AWQ path; plain `awq` may reject BF16. Intermediate quantization values `unquant` and `none` mean that no intermediate quantization override is passed; for an unquantized target this loads B in its checkpoint-native dtype.

Reports are written under `benchmark/hierarchical/outputs/`. Record graph mode, K/L, acceptance length, verification counts, throughput, latency, and KV-capacity effects when comparing configurations.

When `SGLANG_ENABLE_HIERARCHICAL_C_KV_COPY=1`, the outer boundary copies the
complete fixed C verify window into B's corresponding physical slots. The
implementation resolves each worker's underlying KV data pool through its KV
allocator (the worker memory-pool API returns a request pool and allocator pair).
The copy requires ordinary contiguous, unquantized NHD KV buffers with identical
row strides in B and C. This is the optimized cache handoff: B verifies only the
first `K-1` inner rounds, the final A proposals go directly to C, and C's `K*L`
rows replace B's reserved window at the outer boundary. The copy includes
rejected suffix rows; B's committed sequence length controls which rows are
subsequently read. The timing reporter exposes the fused operation as
`c_to_b_kv_copy`.
Use `benchmark/hierarchical/run_c_kv_copy_experiment.sh` to run matched
baseline and copy-enabled cases in separate output directories and compare
`accept_length`, `c_to_b_kv_copy`, and end-to-end time. The script writes the
comparison to `c_kv_copy_comparison.md` and `.json` under the experiment root.

The wrapper can enable `SGLANG_LOG_HIERARCHICAL_TIMING=1`. During the
A+B+C run, CUDA events split each outer decode into B cache preparation and
rollback, candidate-buffer setup/finalization, every A draft and B verification
round, and C verification plus hidden-state publication. A
single synchronization at the end of the outer decode makes the event durations
available. Logs report interval averages and percentages, separate
`a_rounds_total` and `b_rounds_total` aggregates, plus an unattributed
host/launch/sync residual. Set `HIERARCHICAL_TIMING=0` to measure uninstrumented
throughput or `HIERARCHICAL_TIMING_INTERVAL=1` to print every outer decode batch.

## Code Map and Tests

- `python/sglang/srt/speculative/hierarchical_worker.py`: stage construction, execution, cache planning, and synchronization.
- `python/sglang/srt/speculative/hierarchical_types.py`: request state and greedy commit/rollback helpers.
- `python/sglang/srt/speculative/hierarchical_config.py`: window geometry, device-map parsing, and request validation.
- `python/sglang/srt/arg_groups/speculative_hook.py`: validation and CLI resolution.
- `test/registered/unit/speculative/test_hierarchical_types.py`: geometry, acceptance, rollback, and ownership tests.

Run focused tests with `python -m pytest test/registered/unit/speculative/test_hierarchical_types.py` when pytest is available. GPU benchmark validation is separate and hardware-dependent.
