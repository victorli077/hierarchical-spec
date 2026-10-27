"""Configuration helpers for the fixed-topology HierarchicalSpec runtime."""

from __future__ import annotations

from typing import Any


def cascade_pending_width(inner_block_size: int, rounds: int) -> int:
    """Maximum number of B outputs accumulated across one outer window.

    DFlash's block size includes its input anchor, but its published result
    excludes that anchor and contains up to ``inner_block_size`` tokens: up to
    ``inner_block_size - 1`` accepted A proposals plus one B recovery token.
    That recovery is the next round's anchor and remains part of C's pending
    sequence, so each round contributes up to the full block size.
    """
    if inner_block_size <= 0 or rounds <= 0:
        raise ValueError("inner_block_size and rounds must be positive")
    return inner_block_size * rounds


def cascade_verify_width(inner_block_size: int, rounds: int) -> int:
    """C input width: one committed-prefix anchor plus all pending B tokens."""
    return 1 + cascade_pending_width(inner_block_size, rounds)


def validate_hierarchical_request(req: Any) -> str | None:
    """Return an admission error for features outside the v1 contract.

    Keeping this check scheduler-side prevents one unsupported request from
    raising inside the model worker and terminating the whole serving process.
    SamplingParams normalizes temperature=0 to top_k=1.
    """
    sampling = req.sampling_params
    if sampling.top_k != 1:
        return "HierarchicalSpec v1 supports greedy decoding only (temperature=0)."
    if any(
        (
            sampling.json_schema,
            sampling.regex,
            sampling.ebnf,
            sampling.structural_tag,
        )
    ):
        return "HierarchicalSpec v1 does not support constrained decoding."
    if req.return_hidden_states:
        return "HierarchicalSpec v1 does not support return_hidden_states."
    return None


def parse_device_map(
    value: str | None, *, device_count: int | None = None
) -> dict[str, tuple[int, ...]]:
    """Parse ``A:0,B:1,C:2`` into stage -> GPU ids.

    A stage may use a fixed contiguous group (``B:1,2``).  Duplicate stages,
    empty groups, malformed ids, and overlapping groups are rejected early.
    """
    if not value:
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for item in value.split(","):
        if ":" not in item:
            raise ValueError(f"Invalid cascade device map entry: {item!r}")
        stage, raw_ids = (part.strip() for part in item.split(":", 1))
        stage = stage.upper()
        if stage not in {"A", "B", "C"} or stage in result:
            raise ValueError(
                f"Cascade device map stage must be unique A/B/C: {stage!r}"
            )
        try:
            ids = tuple(int(raw.strip()) for raw in raw_ids.split(";") if raw.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid GPU id in cascade device map: {item!r}") from exc
        if not ids or any(gpu < 0 for gpu in ids):
            raise ValueError(f"Cascade device map has no valid GPU ids: {item!r}")
        if device_count is not None and any(gpu >= device_count for gpu in ids):
            raise ValueError(
                f"Cascade device id in {item!r} is outside available range "
                f"[0, {device_count})."
            )
        result[stage] = ids
    missing = {"A", "B", "C"} - result.keys()
    if missing:
        raise ValueError(f"Cascade device map is missing stages: {sorted(missing)}")
    return result
