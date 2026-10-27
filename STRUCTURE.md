# HierarchicalSpec Structure

This document records the implemented three-stage speculative-decoding design. Keep it synchronized with the runtime and benchmark code whenever ownership, state semantics, CLI behavior, or supported topology changes.

## Runtime Topology

The request path is:

```text
A: Domino DFlash drafter -> B: independent intermediate target
       (K refinement rounds) -> C: final target block verification
```

`HierarchicalSpecWorker` owns the cascade. `CascadeCoordinator` owns `IntermediateStage` (B) and the injected `DFlashWorkerV2` inner kernel. DFlashWorkerV2 is reused only for the high-performance A-to-B execution path; B is not the scheduler draft model, draft cache, or draft-only lifecycle. C is the only stage allowed to advance committed output.

## State and Cache Ownership

Each stage has an independent model runner and KV pool. A and B may run ahead with provisional tokens. C verifies the accumulated B block, commits its accepted prefix plus recovery token, and causes A/B provisional suffixes to be rolled back and resynchronized. Rejected tokens must not remain in stage caches, sequence state, or request metadata. The coordinator tracks committed length, pending token ids/rounds, per-stage cache/checkpoint metadata, and reserved RNG state fields.

## Window Geometry

`--speculative-dflash-block-size L` is the inner A-to-B DFlash width and resolves to `speculative_num_draft_tokens`; explicit conflicting values are rejected. With `K=--speculative-cascade-rounds`, each round contributes up to `L` B tokens, so C verifies a fixed width of `1 + K*L`. For `K=3,L=8`, C receives 25 positions and may commit up to 25 new tokens including its recovery token.

## Supported Contract

The current v1 path is linear and greedy (`temperature=0`, `top_k=1`), uses one tokenizer/vocabulary, and supports a single GPU/fixed parallel configuration. A/B/C must share one device; independent stage placement and cross-device transfers are not implemented. CUDA Graphs are disabled for HierarchicalSpec. C-only and A+C may use them unless a benchmark disables them for all modes.

## Quantization and Benchmark

The three-way wrapper is `benchmark/hierarchical/run_sglang_benchmark.sh`; its driver is `benchmark_three_way.py`. It runs A+B+C, C-only, and A+C with matched prompts and request settings. `DTYPE` applies to every mode and defaults to BF16. Intermediate `AWQ` is normalized to `awq_marlin`, matching SGLang's BF16-capable automatic AWQ path; plain `awq` may reject BF16.

Reports are written under `benchmark/hierarchical/outputs/`. Record graph mode, K/L, acceptance length, verification counts, throughput, latency, and KV-capacity effects when comparing configurations.

## Code Map and Tests

- `python/sglang/srt/speculative/hierarchical_worker.py`: stage construction, execution, cache planning, and synchronization.
- `python/sglang/srt/speculative/hierarchical_types.py`: request state and greedy commit/rollback helpers.
- `python/sglang/srt/speculative/hierarchical_config.py`: window geometry, device-map parsing, and request validation.
- `python/sglang/srt/arg_groups/speculative_hook.py`: validation and CLI resolution.
- `test/registered/unit/speculative/test_hierarchical_types.py`: geometry, acceptance, rollback, and ownership tests.

Run focused tests with `python -m pytest test/registered/unit/speculative/test_hierarchical_types.py` when pytest is available. GPU benchmark validation is separate and hardware-dependent.
