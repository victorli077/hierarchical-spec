#!/usr/bin/env python3
"""Summarize a matched HierarchicalSpec C-to-B KV copy experiment."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

COPY_TIME_RE = re.compile(r"c_to_b_kv_copy=([0-9.]+) ms")


def load_rows(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            rows[(row["dataset"], int(row["concurrency"]))] = row
    return rows


def load_copy_times(path: Path) -> list[float]:
    return [float(match) for match in COPY_TIME_RE.findall(path.read_text())]


def percent_change(after: float, before: float) -> float:
    return 100.0 * (after - before) / before if before else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()

    root = args.experiment_root
    baseline = load_rows(root / "baseline" / "abc_hierarchical.jsonl")
    copied = load_rows(root / "c_kv_copy" / "abc_hierarchical.jsonl")
    if set(baseline) != set(copied):
        raise ValueError("baseline and C-to-B copy result rows do not match")

    copy_times = load_copy_times(root / "c_kv_copy" / "benchmark.log")
    lines = [
        "# C-to-B KV copy experiment",
        "",
        "| dataset | conc | baseline accept | copy accept | accept delta | accept delta % | baseline tok/s | copy tok/s | tok/s delta % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    output_rows = []
    for key in sorted(baseline):
        before = baseline[key]
        after = copied[key]
        before_accept = float(before["spec_accept_length"])
        after_accept = float(after["spec_accept_length"])
        before_tps = float(before["output_toks_per_s"])
        after_tps = float(after["output_toks_per_s"])
        row = {
            "dataset": key[0],
            "concurrency": key[1],
            "baseline_accept_length": before_accept,
            "copy_accept_length": after_accept,
            "accept_length_delta": after_accept - before_accept,
            "accept_length_delta_percent": percent_change(after_accept, before_accept),
            "baseline_output_toks_per_s": before_tps,
            "copy_output_toks_per_s": after_tps,
            "output_toks_per_s_delta_percent": percent_change(after_tps, before_tps),
        }
        output_rows.append(row)
        lines.append(
            f"| {key[0]} | {key[1]} | {before_accept:.3f} | "
            f"{after_accept:.3f} | {row['accept_length_delta']:+.3f} | "
            f"{row['accept_length_delta_percent']:+.2f}% | {before_tps:.2f} | "
            f"{after_tps:.2f} | {row['output_toks_per_s_delta_percent']:+.2f}% |"
        )

    lines.extend(["", "## Copy timing", ""])
    if copy_times:
        timing = {
            "num_reports": len(copy_times),
            "mean_ms": statistics.mean(copy_times),
            "median_ms": statistics.median(copy_times),
            "min_ms": min(copy_times),
            "max_ms": max(copy_times),
        }
        lines.append(
            f"`c_to_b_kv_copy`: mean {timing['mean_ms']:.3f} ms, "
            f"median {timing['median_ms']:.3f} ms, min {timing['min_ms']:.3f} ms, "
            f"max {timing['max_ms']:.3f} ms over {timing['num_reports']} reports."
        )
    else:
        timing = None
        lines.append("No `c_to_b_kv_copy` timing samples were found.")

    report = "\n".join(lines) + "\n"
    (root / "c_kv_copy_comparison.md").write_text(report, encoding="utf-8")
    (root / "c_kv_copy_comparison.json").write_text(
        json.dumps(
            {"rows": output_rows, "c_to_b_kv_copy_timing": timing},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
