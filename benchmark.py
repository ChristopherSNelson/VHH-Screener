"""
benchmark.py — VHH-Screener Benchmark Runner
=============================================

Runs the screening loop N times for one or more seed/model combinations and
reports success rate, iterations, and cost per design.

Usage:
    python benchmark.py --seed naive --n 5
    python benchmark.py --seed all --n 3 --model deepseek-ai/DeepSeek-V3
    python benchmark.py --seed naive pembrolizumab --n 5

Results are saved to logs/benchmark_<seed>_<model_short>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

from agent_loop import (
    DEFAULT_MODEL,
    NAIVE_SEED,
    PEMBROLIZUMAB_VH_SEED,
    RunResult,
    run_screening_loop,
)

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ANSI
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

SEED_MAP = {
    "naive": NAIVE_SEED,
    "pembrolizumab": PEMBROLIZUMAB_VH_SEED,
    "none": None,
}


def _pass_label(passed: bool) -> str:
    return f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"


def _fmt(value: float | None, fmt: str = ".2f") -> str:
    return f"{value:{fmt}}" if value is not None else "N/A"


def run_benchmark(
    seed_label: str,
    n: int,
    model: str,
) -> list[RunResult]:
    """Run N screening loops for a given seed and return all RunResults."""
    seed_seq = SEED_MAP[seed_label]

    print(f"\n{BOLD}{CYAN}{'=' * 72}")
    print(f"  Benchmark: seed={seed_label}  model={model}  n={n}")
    print(f"{'=' * 72}{RESET}\n")

    results: list[RunResult] = []

    for i in range(1, n + 1):
        print(f"\n{BOLD}--- Run {i}/{n} ---{RESET}")

        os.environ["MODEL_ID"] = model
        result = run_screening_loop(
            seed_sequence=seed_seq,
            plot_name=f"benchmark_{seed_label}_run{i}",
            suppress_plot=True,
            seed_label=seed_label,
        )
        results.append(result)

        status = _pass_label(result.passed)
        limit_flag = f" {YELLOW}[HIT LIMIT]{RESET}" if result.hit_iteration_limit else ""
        print(
            f"\n  Run {i}: {status}{limit_flag}  "
            f"iters={result.iterations}  "
            f"cost=${result.total_cost_usd:.4f}  "
            f"pI={_fmt(result.final_pi)}  "
            f"GRAVY={_fmt(result.final_gravy, '.3f')}  "
            f"liabilities={result.final_liability_count}  "
            f"APR={_fmt(result.final_apr_percentile, '.1f')}th%ile"
        )

    return results


def print_summary(results: list[RunResult], seed_label: str, model: str) -> None:
    """Print a summary table for a set of benchmark results."""
    n = len(results)
    passing = [r for r in results if r.passed]
    failing = [r for r in results if not r.passed]
    pass_rate = len(passing) / n * 100

    all_iters = [r.iterations for r in results]
    pass_iters = [r.iterations for r in passing]
    fail_iters = [r.iterations for r in failing]
    all_costs = [r.total_cost_usd for r in results]
    pass_costs = [r.total_cost_usd for r in passing]

    print(f"\n{BOLD}{CYAN}{'=' * 72}")
    print(f"  Summary: seed={seed_label}  model={model}  n={n}")
    print(f"{'=' * 72}{RESET}")

    # Per-run table
    print(
        f"\n{'Run':<5} {'Result':<8} {'Iters':<7} {'Cost':>8}  {'pI':>6}  {'GRAVY':>7}  {'Liab':>5}  {'APR%ile':>8}"
    )
    print("-" * 65)
    for i, r in enumerate(results, 1):
        hit = "*" if r.hit_iteration_limit else " "
        print(
            f"{i:<5} {('PASS' if r.passed else 'FAIL'):<8}{hit}"
            f"{r.iterations:<6} ${r.total_cost_usd:>7.4f}  "
            f"{_fmt(r.final_pi):>6}  "
            f"{_fmt(r.final_gravy, '.3f'):>7}  "
            f"{str(r.final_liability_count):>5}  "
            f"{_fmt(r.final_apr_percentile, '.1f'):>7}th"
        )
    print("-" * 65)
    print(f"{'* hit iteration limit':>65}")

    # Aggregate stats
    print(f"\n{BOLD}Pass rate:{RESET}         {len(passing)}/{n} ({pass_rate:.1f}%)")
    print(f"{BOLD}Mean iterations:{RESET}   {statistics.mean(all_iters):.1f}", end="")
    if pass_iters:
        print(f"  (passing: {statistics.mean(pass_iters):.1f}", end="")
        if fail_iters:
            print(f",  failing: {statistics.mean(fail_iters):.1f}", end="")
        print(")", end="")
    print()

    if n > 1:
        print(f"{BOLD}Std dev iters:{RESET}     {statistics.stdev(all_iters):.1f}")

    print(f"{BOLD}Mean cost/run:{RESET}     ${statistics.mean(all_costs):.4f}")
    if pass_costs:
        print(f"{BOLD}Cost/passing:{RESET}      ${statistics.mean(pass_costs):.4f}")
    print(f"{BOLD}Total spent:{RESET}       ${sum(all_costs):.4f}")
    print(
        f"{BOLD}Total tokens:{RESET}      {sum(r.input_tokens for r in results):,} in / "
        f"{sum(r.output_tokens for r in results):,} out"
    )


def save_results(results: list[RunResult], seed_label: str, model: str) -> Path:
    """Save benchmark results to a timestamped JSON file."""
    model_short = model.split("/")[-1]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = LOG_DIR / f"benchmark_{seed_label}_{model_short}_{ts}.json"

    payload = {
        "seed": seed_label,
        "model": model,
        "n": len(results),
        "timestamp": ts,
        "pass_rate": sum(1 for r in results if r.passed) / len(results),
        "runs": [r.to_dict() for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VHH-Screener benchmark runner")
    parser.add_argument(
        "--seed",
        nargs="+",
        choices=["naive", "pembrolizumab", "none", "all"],
        default=["naive"],
        help="Seed(s) to benchmark. 'all' expands to naive + pembrolizumab + none.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of runs per seed/model combination (default: 3).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_ID", DEFAULT_MODEL),
        help="Model ID to use (Together AI). Defaults to MODEL_ID env var or DeepSeek-V3.",
    )
    args = parser.parse_args()

    if not os.environ.get("TOGETHER_API_KEY"):
        print("ERROR: Set TOGETHER_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    # Expand 'all'
    seeds: list[str] = []
    for s in args.seed:
        if s == "all":
            seeds.extend(["naive", "pembrolizumab", "none"])
        else:
            seeds.append(s)
    seeds = list(dict.fromkeys(seeds))  # deduplicate, preserve order

    all_results: dict[str, list[RunResult]] = {}

    for seed in seeds:
        results = run_benchmark(seed_label=seed, n=args.n, model=args.model)
        all_results[seed] = results
        print_summary(results, seed_label=seed, model=args.model)
        out = save_results(results, seed_label=seed, model=args.model)
        print(f"\n{DIM}Results saved: {out}{RESET}")

    if len(seeds) > 1:
        # Cross-seed comparison
        print(f"\n{BOLD}{CYAN}{'=' * 72}")
        print("  Cross-seed comparison")
        print(f"{'=' * 72}{RESET}")
        print(f"\n{'Seed':<16} {'Pass%':>6}  {'MeanIters':>10}  {'MeanCost':>10}")
        print("-" * 50)
        for seed, results in all_results.items():
            n = len(results)
            pass_pct = sum(1 for r in results if r.passed) / n * 100
            mean_iters = statistics.mean(r.iterations for r in results)
            mean_cost = statistics.mean(r.total_cost_usd for r in results)
            print(f"{seed:<16} {pass_pct:>5.1f}%  {mean_iters:>10.1f}  ${mean_cost:>9.4f}")
