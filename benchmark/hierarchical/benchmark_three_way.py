"""Self-contained Domino-style benchmark for A+B+C, C, and A+C.

This file intentionally does not import or execute code from the Domino
checkout. Dataset formatting, server lifecycle, warmup, concurrency sweeps,
metrics, per-mode reports, and the three-way comparison live here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import signal
import socket
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/mnt/workspace/lijiasheng/huggingface")

import requests
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_TASKS = (
    "gsm8k:128,math500:128,humaneval:164,mbpp:128,livecodebench:128,alpaca:128"
)
DEFAULT_TARGET = (
    "/mnt/workspace/lijiasheng/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
DEFAULT_DRAFT = (
    "/mnt/workspace/lijiasheng/huggingface/hub/models--Huang2020--"
    "Qwen3-4B-Domino-b16/snapshots/1a3f54218e6c42e943f13b1a19232682ea9810de"
)
DEFAULT_INTERMEDIATE = (
    "/mnt/workspace/lijiasheng/huggingface/hub/models--Qwen--Qwen3-4B-FP8/"
    "snapshots/96b30dc13593a244a5e59e84687309f53c375cfa"
)
MODE_OUTPUT_NAMES = {
    "abc": "abc_hierarchical",
    "c": "c_only",
    "ac": "ac_dflash",
}
SERVER_START_TIMEOUT_S = 3600


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    spec_accept_length: float | None
    spec_verify_ct_sum: int


def parse_tasks(spec: str) -> list[tuple[str, int]]:
    tasks = []
    for raw in spec.replace("\n", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Task entry must be dataset:count, got {raw!r}")
        name, count_text = raw.split(":", 1)
        count = int(count_text)
        if not name.strip() or count <= 0:
            raise ValueError(f"Invalid task entry: {raw!r}")
        tasks.append((name.strip(), count))
    if not tasks:
        raise ValueError("No tasks specified")
    return tasks


def _load_task_dataset(name: str):
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [
            f"{x['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            for x in ds
        ]
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            f"{x['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            for x in ds
        ]
    if name == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        return [
            f"{x['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            for x in ds
        ]
    if name == "aime25":
        ds = load_dataset("MathArena/aime_2025", split="train")
        return [
            f"{x['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            for x in ds
        ]
    if name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        return [
            (
                f"{x['instruction']}\n\nInput:\n{x['input']}"
                if x["input"]
                else x["instruction"]
            )
            for x in ds
        ]
    if name == "mt-bench":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        return [x["prompt"][0] for x in ds]
    if name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            f"```python\n{x['prompt']}\n```"
            for x in ds
        ]
    if name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        return [x["prompt"] for x in ds]
    if name == "lbpp":
        url = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        ds = load_dataset("parquet", data_files={"test": url})["test"]
        return [x["instruction"] for x in ds]
    if name == "swe-bench":
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        return [
            f"Problem Statement:\n{x['problem_statement']}\nPlease fix the issue described above."
            for x in ds
        ]
    if name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        files = [
            base + f"test{i}.jsonl" if i > 1 else base + "test.jsonl"
            for i in range(1, 7)
        ]
        ds = load_dataset("json", data_files={"test": files})["test"]

        def format_lcb(x: dict[str, Any]) -> str:
            system = (
                "You are an expert Python programmer. You will be given a question "
                "(problem specification) and will generate a correct Python program "
                "that matches the specification and passes all tests. You will NOT "
                "return anything except for the program"
            )
            starter = x.get("starter_code")
            if starter:
                form = "### Format: Use the following code structure:"
                code = f"```python\n{starter}\n```"
            else:
                form = "### Format: Write your code in the following format:"
                code = "```python\n# YOUR CODE HERE\n```"
            return (
                f"{system}\n\n### Question:\n{x['question_content']}\n\n"
                f"{form}\n{code}\n\n### Answer: (use the provided format with backticks)"
            )

        return [format_lcb(x) for x in ds]
    raise ValueError(
        f"Unsupported task {name!r}. Supported: gsm8k, math500, aime24, aime25, "
        "alpaca, mt-bench, humaneval, mbpp, lbpp, swe-bench, livecodebench"
    )


def build_prompts(name: str, tokenizer: AutoTokenizer, required: int) -> list[str]:
    raw = _load_task_dataset(name)
    if not raw:
        raise RuntimeError(f"Dataset {name!r} is empty")
    if len(raw) < required:
        print(
            f"Warning: dataset={name} has {len(raw)} items, need {required}; cycling prompts"
        )
    prompts = []
    for content in raw[:required]:
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return prompts


def cycle_take(pool: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if not pool:
        raise RuntimeError("Cannot take prompts from an empty pool")
    return [pool[i % len(pool)] for i in range(count)]


def compose_run_prompts(
    prompts: list[str], timed_count: int, warmup_count: int, skip_count: int
) -> tuple[list[str], str]:
    timed = cycle_take(prompts, timed_count)
    warmup_pool = prompts[timed_count:] or timed
    source = "post_timed" if prompts[timed_count:] else "timed_reuse"
    return (
        cycle_take(warmup_pool, warmup_count)
        + cycle_take(warmup_pool, skip_count)
        + timed,
        source,
    )


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def find_port(base: int = 20000) -> int:
    port = base + random.randint(100, 1000)
    while not port_available(port):
        port = port + 42 if port < 60000 else port - 43
    return port


def launch_server(model: str, url: str, args: list[str], timeout: int):
    host_port = url.removeprefix("http://")
    host, port = host_port.rsplit(":", 1)
    command = [
        "sglang",
        "serve",
        "--model-path",
        model,
        *args,
        "--host",
        host,
        "--port",
        port,
    ]
    print(f"Server command: {shlex.join(command)}")
    proc = subprocess.Popen(command, env=os.environ.copy(), start_new_session=True)
    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {proc.returncode}")
        try:
            response = requests.get(url + "/health", timeout=5)
            if response.status_code == 200:
                return proc
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    raise TimeoutError(f"Timed out starting SGLang server at {url}: {last_error}")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def flush_cache(url: str) -> None:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url + "/flush_cache", timeout=60)
            response.raise_for_status()
            return
        except requests.HTTPError as exc:
            last_error = exc
            if exc.response is None or exc.response.status_code != 400:
                raise
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1 + attempt)
    print(f"Warning: /flush_cache failed; continuing: {last_error}")


def generate(
    url: str,
    prompt: str | list[str],
    max_new_tokens: int,
    timeout: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict | list[dict]:
    response = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def run_requests(
    url: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    concurrency: int,
    batch_requests: bool,
    timeout_s: int,
    temperature: float,
    top_p: float,
    top_k: int,
    warmup_requests: int,
    warmup_max_new_tokens: int,
    skip_first_requests: int,
    expect_spec: bool,
) -> BenchMetrics:
    params = (temperature, top_p, top_k)

    def issue(items: list[str], token_count: int, batch: bool) -> list[dict]:
        if batch:
            results = generate(url, items, token_count, timeout_s, *params)
            if not isinstance(results, list) or len(results) != len(items):
                raise RuntimeError(
                    "Batched /generate returned an unexpected result count"
                )
            return results
        with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
            futures = [
                pool.submit(generate, url, item, token_count, timeout_s, *params)
                for item in items
            ]
            return [future.result() for future in as_completed(futures)]

    bs = max(concurrency, 1)
    warmup_n = max(warmup_requests, 0) * bs
    if warmup_n and len(prompts) > warmup_n:
        for start in range(0, warmup_n, bs if batch_requests else warmup_n):
            issue(
                prompts[
                    start : min(start + (bs if batch_requests else warmup_n), warmup_n)
                ],
                warmup_max_new_tokens or max_new_tokens,
                batch_requests,
            )
        prompts = prompts[warmup_n:]

    skip_n = max(skip_first_requests, 0)
    if skip_n and len(prompts) > skip_n:
        issue(prompts[:skip_n], max_new_tokens, batch_requests)
        prompts = prompts[skip_n:]

    start_time = time.perf_counter()
    outputs: list[dict] = []
    if batch_requests:
        for start in range(0, len(prompts), bs):
            outputs.extend(issue(prompts[start : start + bs], max_new_tokens, True))
    else:
        with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
            futures = [
                pool.submit(generate, url, prompt, max_new_tokens, timeout_s, *params)
                for prompt in prompts
            ]
            outputs = [future.result() for future in as_completed(futures)]
    latency = time.perf_counter() - start_time

    total_tokens = 0
    verify_count = 0
    accept_lengths = []
    for output in outputs:
        meta = output.get("meta_info", {}) or {}
        total_tokens += int(meta.get("completion_tokens", 0))
        verify_count += int(meta.get("spec_verify_ct", 0))
        if meta.get("spec_accept_length") is not None:
            try:
                accept_lengths.append(float(meta["spec_accept_length"]))
            except (ValueError, TypeError):
                pass
    if expect_spec and verify_count <= 0:
        raise RuntimeError("Speculative benchmark returned no spec_verify_ct metadata")
    return BenchMetrics(
        latency_s=latency,
        output_tokens=total_tokens,
        output_toks_per_s=total_tokens / max(latency, 1e-6),
        spec_accept_length=statistics.mean(accept_lengths) if accept_lengths else None,
        spec_verify_ct_sum=verify_count,
    )


def server_args(args: argparse.Namespace, mode: str) -> list[str]:
    result = [
        "--trust-remote-code",
        "--attention-backend",
        args.attention_backend,
        "--tp-size",
        str(args.tp_size),
        "--dtype",
        args.dtype,
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(args.max_running_requests),
        "--page-size",
        str(args.page_size),
    ]
    if args.disable_radix_cache:
        result.append("--disable-radix-cache")
    # Historically the ABC benchmark disabled CUDA graphs unconditionally so
    # that the eager hierarchical path could be measured.  Keep that default,
    # but allow explicit HierarchicalSpec role-graph experiments to opt back in
    # through the role-specific environment variables exported by
    # run_sglang_benchmark.sh.
    hierarchical_graph_enabled = any(
        os.environ.get(name, "0") == "1"
        for name in (
            "SGLANG_ENABLE_HIERARCHICAL_A_CUDA_GRAPH",
            "SGLANG_ENABLE_HIERARCHICAL_B_CUDA_GRAPH",
            "SGLANG_ENABLE_HIERARCHICAL_C_CUDA_GRAPH",
        )
    )
    if args.disable_cuda_graph or (mode == "abc" and not hierarchical_graph_enabled):
        result.extend(
            [
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
            ]
        )
    else:
        result.extend(
            [
                "--cuda-graph-bs-decode",
                *[str(i) for i in range(1, args.cuda_graph_max_bs + 1)],
                "--cuda-graph-max-bs-decode",
                str(args.cuda_graph_max_bs),
            ]
        )
    if mode == "ac":
        result.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.draft_model,
            ]
        )
        if args.dflash_block_size is not None:
            result.extend(
                ["--speculative-dflash-block-size", str(args.dflash_block_size)]
            )
        if args.speculative_draft_model_quantization:
            result.extend(
                [
                    "--speculative-draft-model-quantization",
                    args.speculative_draft_model_quantization,
                ]
            )
    elif mode == "abc":
        result.extend(
            [
                "--speculative-algorithm",
                "HierarchicalSpec",
                "--speculative-draft-model-path",
                args.draft_model,
                "--speculative-intermediate-model-path",
                args.intermediate_model,
                "--speculative-cascade-rounds",
                str(args.cascade_rounds),
                "--speculative-dflash-block-size",
                str(args.dflash_block_size or 4),
            ]
        )
        if args.speculative_draft_model_quantization:
            result.extend(
                [
                    "--speculative-draft-model-quantization",
                    args.speculative_draft_model_quantization,
                ]
            )
        if args.intermediate_model_quantization:
            result.extend(
                [
                    "--speculative-intermediate-quantization",
                    args.intermediate_model_quantization,
                ]
            )
    return result


def fmt(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def write_mode_report(
    path: Path,
    *,
    mode: str,
    args: argparse.Namespace,
    tasks: list[tuple[str, int]],
    concurrencies: list[int],
    results: dict[tuple[str, int], BenchMetrics],
) -> None:
    title = {
        "abc": "A+B+C HierarchicalSpec",
        "c": "C-only baseline",
        "ac": "A+C DFlash",
    }[mode]
    rows = []
    for dataset, count in tasks:
        for concurrency in concurrencies:
            metric = results.get((dataset, concurrency))
            if metric is None:
                continue
            rows.append(
                {
                    "mode": mode,
                    "dataset": dataset,
                    "num_questions": count,
                    "concurrency": concurrency,
                    "target_model": args.target_model,
                    "draft_model": args.draft_model if mode != "c" else None,
                    "intermediate_model": args.intermediate_model
                    if mode == "abc"
                    else None,
                    "intermediate_model_quantization": (
                        args.intermediate_model_quantization if mode == "abc" else None
                    ),
                    "cascade_rounds": args.cascade_rounds if mode == "abc" else None,
                    "dflash_block_size": args.dflash_block_size
                    if mode != "c"
                    else None,
                    "dtype": args.dtype,
                    "attention_backend": "flashinfer"
                    if mode == "abc"
                    else args.attention_backend,
                    "tp_size": args.tp_size,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                    **asdict(metric),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    md = [
        f"# {title} multi-task benchmark",
        "",
        f"- target_model: `{args.target_model}`",
        f"- draft_model: `{args.draft_model if mode != 'c' else 'N/A'}`",
        f"- intermediate_model: `{args.intermediate_model if mode == 'abc' else 'N/A'}`",
        f"- cascade_rounds: `{args.cascade_rounds if mode == 'abc' else 'N/A'}`",
        f"- block_size: `{args.dflash_block_size}`",
        f"- dtype: `{args.dtype}`, temperature: `{args.temperature}`, top_p: `{args.top_p}`, top_k: `{args.top_k}`",
        f"- warmup_requests: `{args.warmup_requests}`, warmup_max_new_tokens: `{args.warmup_max_new_tokens}`, skip_first_requests: `{args.skip_first_requests}`",
        "",
        "| dataset | requests | concurrency | output tokens | latency (s) | tok/s | accept length | verify count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['dataset']} | {row['num_questions']} | {row['concurrency']} | "
            f"{row['output_tokens']} | {fmt(row['latency_s'])} | {fmt(row['output_toks_per_s'])} | "
            f"{fmt(row['spec_accept_length'], 3)} | "
            f"{row['spec_verify_ct_sum']} |"
        )
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def summarize_three_way(out_dir: Path) -> None:
    groups: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for name, file_name in (
        ("abc", "abc_hierarchical.jsonl"),
        ("c", "c_only.jsonl"),
        ("ac", "ac_dflash.jsonl"),
    ):
        rows = {}
        with (out_dir / file_name).open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                rows[(row["dataset"], int(row["concurrency"]))] = row
        groups[name] = rows
    expected = set(groups["c"])
    for name, rows in groups.items():
        if set(rows) != expected:
            raise ValueError(
                f"Rows differ for {name}: missing={expected - set(rows)}, extra={set(rows) - expected}"
            )
    comparisons = []
    for dataset, concurrency in sorted(expected):
        abc, c, ac = (
            groups["abc"][(dataset, concurrency)],
            groups["c"][(dataset, concurrency)],
            groups["ac"][(dataset, concurrency)],
        )
        c_tps, ac_tps, abc_tps = (
            float(row["output_toks_per_s"]) for row in (c, ac, abc)
        )
        comparisons.append(
            {
                "dataset": dataset,
                "concurrency": concurrency,
                "c_toks_per_s": c_tps,
                "ac_toks_per_s": ac_tps,
                "abc_toks_per_s": abc_tps,
                "ac_speedup_vs_c": ac_tps / c_tps,
                "abc_speedup_vs_c": abc_tps / c_tps,
                "abc_speedup_vs_ac": abc_tps / ac_tps,
                "c_latency_s": c["latency_s"],
                "ac_latency_s": ac["latency_s"],
                "abc_latency_s": abc["latency_s"],
                "ac_accept_length": ac["spec_accept_length"],
                "abc_accept_length": abc["spec_accept_length"],
                "c_output_tokens": c["output_tokens"],
                "ac_output_tokens": ac["output_tokens"],
                "abc_output_tokens": abc["output_tokens"],
            }
        )
    (out_dir / "comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comparisons),
        encoding="utf-8",
    )
    md = [
        "# A+B+C vs C vs A+C",
        "",
        "| dataset | conc | C tok/s | A+C tok/s | A+B+C tok/s | A+C/C | A+B+C/C | A+B+C/A+C | C latency | A+C latency | A+B+C latency | A+C accept | A+B+C accept |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        md.append(
            f"| {row['dataset']} | {row['concurrency']} | {fmt(row['c_toks_per_s'])} | "
            f"{fmt(row['ac_toks_per_s'])} | {fmt(row['abc_toks_per_s'])} | "
            f"{fmt(row['ac_speedup_vs_c'], 3)}x | {fmt(row['abc_speedup_vs_c'], 3)}x | "
            f"{fmt(row['abc_speedup_vs_ac'], 3)}x | {fmt(row['c_latency_s'])}s | "
            f"{fmt(row['ac_latency_s'])}s | {fmt(row['abc_latency_s'])}s | "
            f"{fmt(row['ac_accept_length'], 3)} | {fmt(row['abc_accept_length'], 3)} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT)
    parser.add_argument("--intermediate-model", default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--intermediate-model-quantization", default=None)
    parser.add_argument("--speculative-draft-model-quantization", default=None)
    parser.add_argument(
        "--modes",
        default="abc,c,ac",
        help="Comma-separated benchmark modes: abc, c, ac.",
    )
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--concurrencies", default="1,2,4,8,16,32,64")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--cascade-rounds", type=int, default=3)
    parser.add_argument("--dflash-block-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=64)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--warmup-max-new-tokens", type=int, default=1024)
    parser.add_argument("--skip-first-requests", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--batch-requests", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    modes = [mode.strip().lower() for mode in args.modes.split(",") if mode.strip()]
    invalid_modes = [mode for mode in modes if mode not in MODE_OUTPUT_NAMES]
    if not modes or invalid_modes:
        parser.error(
            "--modes must contain one or more of abc,c,ac; "
            f"invalid values: {invalid_modes}"
        )
    if len(modes) != len(set(modes)):
        parser.error("--modes must not contain duplicates")
    if (
        args.intermediate_model_quantization is not None
        and args.intermediate_model_quantization.lower() in {"none", "unquant"}
    ):
        args.intermediate_model_quantization = None

    if args.temperature != 0 or args.top_k != 1:
        parser.error(
            "HierarchicalSpec v1 benchmark requires greedy decoding: temperature=0 and top_k=1"
        )
    if args.cascade_rounds <= 0 or args.dflash_block_size <= 0:
        parser.error("--cascade-rounds and --dflash-block-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    tasks = parse_tasks(args.tasks)
    concurrencies = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    if not concurrencies or min(concurrencies) < 1:
        raise ValueError("--concurrencies must contain positive integers")
    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent
            / "outputs"
            / time.strftime("three_way_%Y%m%d_%H%M%S")
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    required_count = (
        max(count for _, count in tasks)
        + args.warmup_requests * max(concurrencies)
        + args.skip_first_requests
    )
    print("Loading and formatting all benchmark prompts...")
    prompts_by_task = {
        name: build_prompts(name, tokenizer, required_count) for name, _ in tasks
    }

    for mode in modes:
        output_name = MODE_OUTPUT_NAMES[mode]
        port = find_port()
        url = f"http://127.0.0.1:{port}"
        cmd_args = server_args(args, mode)
        print(f"\n=== Starting {output_name} at {url} ===")
        proc = launch_server(args.target_model, url, cmd_args, SERVER_START_TIMEOUT_S)
        results: dict[tuple[str, int], BenchMetrics] = {}
        try:
            generate(
                url,
                "Hello",
                8,
                min(args.timeout_s, 300),
                args.temperature,
                args.top_p,
                args.top_k,
            )
            for dataset, count in tasks:
                for concurrency in concurrencies:
                    flush_cache(url)
                    prompt_set, warmup_source = compose_run_prompts(
                        prompts_by_task[dataset],
                        count,
                        args.warmup_requests * concurrency,
                        args.skip_first_requests,
                    )
                    print(
                        f"[{output_name}] dataset={dataset} conc={concurrency} n={count} "
                        f"warmup={args.warmup_requests} source={warmup_source}"
                    )
                    metrics = run_requests(
                        url,
                        prompt_set,
                        max_new_tokens=args.max_new_tokens,
                        concurrency=concurrency,
                        batch_requests=args.batch_requests,
                        timeout_s=args.timeout_s,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        warmup_requests=args.warmup_requests,
                        warmup_max_new_tokens=args.warmup_max_new_tokens,
                        skip_first_requests=args.skip_first_requests,
                        expect_spec=mode != "c",
                    )
                    results[(dataset, concurrency)] = metrics
                    message = (
                        f"[{output_name}] {dataset} conc={concurrency} "
                        f"tok/s={metrics.output_toks_per_s:,.2f} latency={metrics.latency_s:.2f}s"
                    )
                    if metrics.spec_accept_length is not None:
                        message += f" accept_len={metrics.spec_accept_length:.3f}"
                    print(message)
                    write_mode_report(
                        args.output_dir / f"{output_name}.jsonl",
                        mode=mode,
                        args=args,
                        tasks=tasks,
                        concurrencies=concurrencies,
                        results=results,
                    )
        finally:
            stop_server(proc)

    if set(modes) == set(MODE_OUTPUT_NAMES):
        summarize_three_way(args.output_dir)
    print(f"Benchmark reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
