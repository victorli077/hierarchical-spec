# Repository Guidelines

## Project Structure

SGLang runtime code is under `python/sglang/srt/`: scheduler and request lifecycle in `managers/`, model execution in `model_executor/`, speculative algorithms in `speculative/`, and KV allocation in `mem_cache/`. Tests are under `test/registered/` (CI-registered) and `test/manual/` (hardware-dependent). Benchmarks live in `benchmark/`; documentation is in `docs/`.

## Development and Tests

Run focused tests from the repository root, for example:

```bash
python -m pytest test/registered/unit/speculative/test_hierarchical_types.py
pre-commit run --files python/sglang/srt/speculative/hierarchical_worker.py
```

Use the matching CUDA/PyTorch environment for GPU tests. Add regressions alongside the relevant registered test area; broaden testing when shared scheduler, runner, or cache behavior changes.

## Style and Naming

Follow nearby conventions: four-space Python indentation, type annotations for cross-module interfaces, `snake_case` functions/modules, and `PascalCase` classes. Use Ruff and pre-commit for Python lint/format checks. Document non-obvious cache or scheduling invariants close to the code.

## HierarchicalSpec Reference

The detailed three-stage architecture, state/cache ownership, window formulas, CLI resolution, quantization behavior, benchmark workflow, limitations, and code map are maintained in [`STRUCTURE.md`](STRUCTURE.md). Read it before modifying HierarchicalSpec or its benchmark.

Any change to cascade ownership, token/commit semantics, KV allocation or rollback, supported topology, CLI flags, quantization defaults, or benchmark workflow must update `STRUCTURE.md` and relevant focused tests in the same change. Do not duplicate stale architecture details here.

## Commits and Pull Requests

Recent history uses concise conventional-style subjects, often prefixed by scope (for example, `[Feat] spec: ...`, `[Fix] ...`, or `[Refactor] spec: ...`). Use the same pattern. Pull requests should explain behavior and compatibility, list tests and hardware/configuration used, link related issues, and include benchmark results for performance-sensitive changes.
