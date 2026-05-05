"""
agent_loop.py — Developability Screening Loop for VHH Design
=============================================================

Implements a "generate → screen → critique → mutate" loop using
Together AI (DeepSeek V3) via the OpenAI-compatible API.

The agent acts as a Senior Biologics Lead designing a VHH nanobody
binder for Human PD-1 that targets the same epitope as Pembrolizumab.

Zero-shot binding strategy inspired by the Escalante 180-line approach:
  https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/

The screening tools (scan_structural_liabilities,
calculate_biophysical_profile, vhh_hallmark_audit) are deterministic regex /
BioPython checks — not generative — ensuring ground-truth developability
constraints.

Chain of Thought is printed in green and logged to logs/agent_cot.log.

Configuration (env vars):
  TOGETHER_API_KEY  — Required. Your Together AI API key.
  MODEL_ID          — Optional. Defaults to deepseek-ai/DeepSeek-V3.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
from openai import OpenAI

# ---------------------------------------------------------------------------
# Import deterministic screening tools from the MCP server module
# ---------------------------------------------------------------------------
from biologics_server import (
    calculate_biophysical_profile,
    scan_aggregation_patches,
    scan_structural_liabilities,
    vhh_hallmark_audit,
)

# ---------------------------------------------------------------------------
# Model / provider config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3"
BASE_URL = "https://api.together.xyz/v1"

# Pricing per million tokens (USD) — update if rates change
PRICE_PER_M_INPUT = 0.20
PRICE_PER_M_OUTPUT = 0.60

# ---------------------------------------------------------------------------
# Logging — all Chain of Thought goes to logs/agent_cot.log AND terminal
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

COT_LOG = LOG_DIR / "agent_cot.log"

cot_logger = logging.getLogger("agent-cot")
cot_logger.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler(COT_LOG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
cot_logger.addHandler(_file_handler)

# ANSI green for terminal CoT
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"


@dataclass
class RunResult:
    """Result of a single screening loop run, used for benchmarking."""

    seed_label: str
    model: str
    passed: bool
    iterations: int
    hit_iteration_limit: bool
    total_cost_usd: float
    input_tokens: int
    output_tokens: int
    final_sequence: str | None
    final_pi: float | None
    final_gravy: float | None
    final_liability_count: int | None
    final_apr_percentile: float | None
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def cot_print(msg: str) -> None:
    """Print chain-of-thought in green and log to file."""
    print(f"{GREEN}{msg}{RESET}")
    cot_logger.info(msg)


def header_print(msg: str) -> None:
    """Print a section header in bold cyan."""
    print(f"\n{BOLD}{CYAN}{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}{RESET}\n")


def warn_print(msg: str) -> None:
    """Print a warning in yellow."""
    print(f"{YELLOW}{msg}{RESET}")


# ---------------------------------------------------------------------------
# Tool registry — single source of truth for all agent tools.
# TOOLS and TOOL_DISPATCH are derived automatically; add new tools here only.
# ---------------------------------------------------------------------------
TOOL_REGISTRY: list[dict] = [
    {
        "name": "scan_structural_liabilities",
        "description": (
            "Scan a protein sequence for post-translational modification "
            "hotspots: Deamidation (NG/NS/NA), Isomerization (DG), and "
            "N-glycosylation (N-X-S/T). Returns JSON with liabilities list, "
            "count, and overall PASS/FAIL flag."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Single-letter amino-acid sequence.",
                },
            },
            "required": ["sequence"],
        },
        "fn": lambda args: scan_structural_liabilities(args["sequence"]),
    },
    {
        "name": "calculate_biophysical_profile",
        "description": (
            "Calculate isoelectric point (pI) and GRAVY hydropathy score "
            "for a protein sequence. Flags aggregation risk: pI < 7.5 = FAIL, "
            "GRAVY > 0.0 = FAIL. Returns structured JSON."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Single-letter amino-acid sequence.",
                },
            },
            "required": ["sequence"],
        },
        "fn": lambda args: calculate_biophysical_profile(args["sequence"]),
    },
    {
        "name": "vhh_hallmark_audit",
        "description": (
            "Audit FR2 hallmark positions (Kabat 37, 44, 45, 47) for "
            "camelid vs. human VH identity. Returns per-position audit "
            "with humanization suggestions and warnings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Single-letter amino-acid VHH sequence.",
                },
                "framework2_start": {
                    "type": "integer",
                    "description": "0-based index where FR2 begins. Default 36.",
                },
            },
            "required": ["sequence"],
        },
        "fn": lambda args: vhh_hallmark_audit(args["sequence"], args.get("framework2_start", 36)),
    },
    {
        "name": "scan_aggregation_patches",
        "description": (
            "Scan for aggregation-prone regions (APRs) using clinically-"
            "calibrated sliding-window hydrophobicity. Each 7-residue "
            "window is scored against a reference distribution of 13 "
            "clinical-stage VH/VHH domains. Returns z-scores, percentiles, "
            "Caplacizumab comparison, and PASS/FAIL against the 95th "
            "percentile screening threshold."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Single-letter amino-acid sequence.",
                },
                "window_size": {
                    "type": "integer",
                    "description": "Sliding window width (default 7).",
                },
            },
            "required": ["sequence"],
        },
        "fn": lambda args: scan_aggregation_patches(args["sequence"], args.get("window_size", 7)),
    },
]

# Derived from TOOL_REGISTRY — do not edit directly
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        },
    }
    for t in TOOL_REGISTRY
]
TOOL_DISPATCH: dict[str, callable] = {t["name"]: t["fn"] for t in TOOL_REGISTRY}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a **Senior Biologics Lead** at an agentic drug discovery startup.

## Mission
Design a VHH (camelid nanobody) binder for **Human PD-1** that targets the \
same epitope as **Pembrolizumab** (Keytruda). Use a zero-shot binding strategy \
inspired by the Escalante 180-line approach \
(https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/).

## Developability Screening Protocol
Each iteration you will:

1. **Reason**: Analyse the screening results from the previous iteration. \
For each FAIL, state the exact position, motif, mechanism, and proposed point mutation.

2. **Propose**: Write your revised full VHH sequence (starting with EVQLV...). \
CDR3 must mimic the Pembrolizumab heavy-chain CDR3 binding geometry against \
the PD-1 CC' loop / FG loop epitope. Every iteration MUST produce a \
DIFFERENT sequence from the previous one — apply your proposed mutations.

3. **Format**: ALWAYS output your proposed sequence on its own line, enclosed \
in a fenced code block labelled `sequence`:

```sequence
EVQLV...
```

The system will automatically screen your sequence with all four tools and \
return the results. Do not call any tools yourself.

## Developability Constraints (hard requirements)
- pI > 7.5 (avoid precipitation near physiological pH; add Lys/Arg to raise pI)
- GRAVY ≤ 0.0 (hydrophilic surface → lower aggregation; replace hydrophobic residues)
- No aggregation-prone patches exceeding the 95th percentile of clinical-stage therapeutics
- Zero deamidation motifs (NG, NS, NA): mutate N→Q or following G/S/A→A
- Zero isomerization motifs (DG): mutate D→E or G→A
- Zero N-glycosylation sequons (N-X-S/T, X≠Pro): mutate N→Q
- FR2 hallmark tetrad (positions 37/44/45/47) must be assessed and documented

## Key charge-engineering guidance
- pI < 7.5 means the sequence has too many acidic residues (D, E). Replace surface D/E \
in framework regions with K or R. Adding 3-4 Lys/Arg substitutions typically raises pI \
by 1-2 units.

## Output format
Think step by step. Show reasoning for each mutation. Then write the full revised \
sequence in the ```sequence block. Do not truncate — write the complete sequence.
"""

# ---------------------------------------------------------------------------
# Seed sequences for different starting points
# ---------------------------------------------------------------------------

# Option 1: Deliberately bad naive VHH — 7 liabilities, low pI, APR 96.6th %ile,
# fully humanized FR2. Designed to require ~8 iterations to pass all constraints.
# CDR3 = ASIVFSYDGY: IVF cluster drives APR failure (worst window CASIVFS, 96.6th %ile).
# The agent can fix APR with a single F→Y or V→E substitution in CDR3.
# DG at CDR3 pos 8-9 adds an isomerization liability alongside the framework NG/NS motifs.
NAIVE_SEED = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSNGYMSNGWVRQAPGKGLEWVSDGISNGGS"
    "TYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCASIVFSYDGYWGQGTLVTVSS"
)

# Option 2: Pembrolizumab heavy chain VH (PDB 5DK3, chain B, residues 1-120) — real clinical sequence,
# 1 liability (NG), good pI, but fully humanized FR2 (needs camelid conversion).
PEMBROLIZUMAB_VH_SEED = (
    "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTN"
    "FNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
)

# ---------------------------------------------------------------------------
# Main screening loop
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 10  # Safety cap to prevent runaway loops
MAX_TOKENS = 8192  # Per-iteration output budget; doubled automatically on length failures
CONTEXT_KEEP_MESSAGES = 20  # Keep system + user prompt + last ~3 full iterations of history


PLOT_DIR = Path(__file__).parent / "assets"


def _plot_biophysical_trajectory(
    points: list[dict],
    plot_name: str = "biophysical_trajectory",
    auto_open: bool = True,
) -> Path:
    """Generate a multi-panel developability dashboard.

    Four panels track the optimization trajectory across iterations:
      1. pI — isoelectric point with pass/fail threshold
      2. GRAVY — hydropathy with pass/fail threshold
      3. Liability count — PTM hotspots per iteration
      4. APR percentile — worst hydrophobic patch vs. clinical distribution
    """
    PLOT_DIR.mkdir(exist_ok=True)
    out_path = PLOT_DIR / f"{plot_name}.png"

    def _is_imputed(point: dict, key: str) -> bool:
        return key in point.get("_imputed", set())

    iters = [p["iteration"] for p in points]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0a0a0a")
    fig.suptitle(
        "VHH-Screener: Developability Optimization Dashboard",
        color="white",
        fontsize=14,
        fontfamily="monospace",
        y=0.97,
    )

    for ax in axes.flat:
        ax.set_facecolor("#0a0a0a")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.set_xlabel("Iteration", color="white", fontsize=9, fontfamily="monospace")

    # --- Panel 1: pI trajectory ---
    ax1 = axes[0, 0]
    pis = [p.get("pI", 0) for p in points]
    colors_pi = ["#00ff41" if pi > 7.5 else "#ff3333" for pi in pis]
    ax1.axhline(7.5, color="#00ff41", linewidth=1, linestyle="--", alpha=0.4)
    ax1.axhspan(0, 7.5, alpha=0.06, color="#ff3333", zorder=0)
    ax1.axhspan(7.5, max(max(pis) + 0.5, 10), alpha=0.06, color="#00ff41", zorder=0)
    ax1.plot(iters, pis, color="#555555", linewidth=1.5, zorder=1)
    ax1.scatter(iters, pis, c=colors_pi, s=70, zorder=2, edgecolors="white", linewidths=0.8)
    for i, (it, pi) in enumerate(zip(iters, pis)):
        label = "NA" if _is_imputed(points[i], "pI") else f"{pi:.1f}"
        ax1.annotate(
            label,
            (it, pi),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#888888" if label == "NA" else "white",
            fontfamily="monospace",
        )
    ax1.set_ylabel("Isoelectric Point (pI)", color="white", fontsize=9, fontfamily="monospace")
    ax1.set_title("pI (threshold: 7.5)", color="white", fontsize=10, fontfamily="monospace")
    ax1.text(
        max(iters),
        7.5,
        " PASS",
        color="#00ff41",
        fontsize=7,
        va="bottom",
        fontfamily="monospace",
        alpha=0.6,
    )
    ax1.text(
        max(iters),
        7.5,
        " FAIL",
        color="#ff3333",
        fontsize=7,
        va="top",
        fontfamily="monospace",
        alpha=0.6,
    )

    # --- Panel 2: GRAVY trajectory ---
    ax2 = axes[0, 1]
    gravys = [p.get("gravy", 0) for p in points]
    colors_gv = ["#00ff41" if g <= 0.0 else "#ff3333" for g in gravys]
    ax2.axhline(0.0, color="#00ff41", linewidth=1, linestyle="--", alpha=0.4)
    ax2.axhspan(min(min(gravys) - 0.05, -0.3), 0.0, alpha=0.06, color="#00ff41", zorder=0)
    ax2.axhspan(0.0, max(max(gravys) + 0.05, 0.1), alpha=0.06, color="#ff3333", zorder=0)
    ax2.plot(iters, gravys, color="#555555", linewidth=1.5, zorder=1)
    ax2.scatter(iters, gravys, c=colors_gv, s=70, zorder=2, edgecolors="white", linewidths=0.8)
    for i, (it, gv) in enumerate(zip(iters, gravys)):
        label = "NA" if _is_imputed(points[i], "gravy") else f"{gv:.3f}"
        ax2.annotate(
            label,
            (it, gv),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#888888" if label == "NA" else "white",
            fontfamily="monospace",
        )
    ax2.set_ylabel("GRAVY Score", color="white", fontsize=9, fontfamily="monospace")
    ax2.set_title("GRAVY (threshold: 0.0)", color="white", fontsize=10, fontfamily="monospace")
    ax2.text(
        max(iters),
        0.0,
        " FAIL",
        color="#ff3333",
        fontsize=7,
        va="bottom",
        fontfamily="monospace",
        alpha=0.6,
    )
    ax2.text(
        max(iters),
        0.0,
        " PASS",
        color="#00ff41",
        fontsize=7,
        va="top",
        fontfamily="monospace",
        alpha=0.6,
    )

    # --- Panel 3: Liability count ---
    ax3 = axes[1, 0]
    liabs = [p.get("liability_count", 0) for p in points]
    colors_li = ["#00ff41" if lc == 0 else "#ff3333" for lc in liabs]
    ax3.axhline(0, color="#00ff41", linewidth=1, linestyle="--", alpha=0.4)
    ax3.bar(
        iters,
        liabs,
        color=colors_li,
        width=0.6,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        zorder=2,
    )
    for i, (it, lc) in enumerate(zip(iters, liabs)):
        label = "NA" if _is_imputed(points[i], "liability_count") else str(lc)
        ax3.annotate(
            label,
            (it, lc),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
            color="#888888" if label == "NA" else "white",
            fontfamily="monospace",
            fontweight="bold",
        )
    ax3.set_ylabel("Liability Count", color="white", fontsize=9, fontfamily="monospace")
    ax3.set_title(
        "PTM Liabilities (target: 0)",
        color="white",
        fontsize=10,
        fontfamily="monospace",
    )
    ax3.set_ylim(bottom=-0.2)

    # --- Panel 4: APR percentile ---
    ax4 = axes[1, 1]
    apr_pcts = [p.get("apr_percentile", 0) for p in points]
    colors_apr = ["#00ff41" if pct < 95 else "#ff3333" for pct in apr_pcts]
    ax4.axhline(95, color="#ff3333", linewidth=1, linestyle="--", alpha=0.4)
    ax4.axhline(76.9, color="#00aaff", linewidth=1, linestyle=":", alpha=0.3)
    ax4.axhspan(95, 100, alpha=0.06, color="#ff3333", zorder=0)
    ax4.plot(iters, apr_pcts, color="#555555", linewidth=1.5, zorder=1)
    ax4.scatter(
        iters,
        apr_pcts,
        c=colors_apr,
        s=70,
        zorder=2,
        edgecolors="white",
        linewidths=0.8,
    )
    for i, (it, pct) in enumerate(zip(iters, apr_pcts)):
        label = "NA" if _is_imputed(points[i], "apr_percentile") else f"{pct:.0f}%"
        ax4.annotate(
            label,
            (it, pct),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#888888" if label == "NA" else "white",
            fontfamily="monospace",
        )
    ax4.set_ylabel("Percentile vs CSTs", color="white", fontsize=9, fontfamily="monospace")
    ax4.set_title(
        "APR Patch (threshold: 95th %ile)",
        color="white",
        fontsize=10,
        fontfamily="monospace",
    )
    ax4.set_ylim(0, 105)
    ax4.text(
        max(iters),
        95,
        " FAIL",
        color="#ff3333",
        fontsize=7,
        va="bottom",
        fontfamily="monospace",
        alpha=0.6,
    )
    ax4.text(
        max(iters),
        76.9,
        " Caplacizumab",
        color="#00aaff",
        fontsize=7,
        va="bottom",
        fontfamily="monospace",
        alpha=0.5,
    )

    # Integer x-ticks
    for ax in axes.flat:
        ax.set_xticks(iters)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)

    if auto_open:
        subprocess.run(["open", str(out_path)], check=False)

    return out_path


def extract_sequence(text: str) -> str | None:
    """Parse a proposed VHH sequence from agent text.

    Looks for content inside a fenced ```sequence ... ``` block.
    Falls back to the longest line that contains only standard amino acid
    characters and is 100-140 residues long.
    """
    import re

    # Primary: fenced sequence block
    m = re.search(r"```sequence\s*\n([ACDEFGHIKLMNPQRSTVWY\n\s]+?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).replace("\n", "").replace(" ", "").strip()

    # Fallback: bare line of amino acids in the right length range
    for line in text.splitlines():
        line = line.strip()
        if 100 <= len(line) <= 145 and re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", line):
            return line

    return None


def build_screening_message(
    seq: str,
    bp: dict,
    sl: dict,
    ap: dict,
    ha: dict,
    iteration: int,
    best_seq: str | None = None,
    best_score: int = 0,
) -> str:
    """Build a rich screening results message to inject into the conversation."""
    pi = bp.get("isoelectric_point")
    gravy = bp.get("gravy")
    liab_count = sl.get("liability_count", 0)
    liabilities = sl.get("liabilities", [])
    apr_data = ap.get("candidate_max_patch", {})
    apr_pct = apr_data.get("percentile")
    hallmarks = ha.get("hallmark_audit", [])

    pi_ok = pi is not None and pi > 7.5
    gravy_ok = gravy is not None and gravy <= 0.0
    liab_ok = liab_count == 0
    apr_ok = apr_pct is not None and apr_pct < 95.0
    score = sum([pi_ok, gravy_ok, liab_ok, apr_ok])

    def flag(ok: bool) -> str:
        return "✓" if ok else "✗"

    lines = [
        f"[Screener] Iteration {iteration} results ({len(seq)} AA):",
        f"  Sequence: {seq}",
        f"  pI    = {pi:.2f} {flag(pi_ok)}  ({'PASS' if pi_ok else 'FAIL — must be > 7.5; add Lys/Arg substitutions'})",
        f"  GRAVY = {gravy:.3f} {flag(gravy_ok)}  ({'PASS' if gravy_ok else 'FAIL — must be ≤ 0.0; replace hydrophobic residues'})",
        f"  PTM liabilities: {liab_count} {flag(liab_ok)}",
    ]
    for liab in liabilities:
        lines.append(
            f"    - {liab['liability_type']} motif {liab['motif']} at position "
            f"{liab['position']}: {liab.get('context', '')} — {liab.get('mechanism', '')}"
        )
    apr_label = f"{apr_pct:.0f}th percentile" if apr_pct is not None else "?"
    lines.append(
        f"  APR   = {apr_label} {flag(apr_ok)}  ({'PASS' if apr_ok else 'FAIL — must be < 95th percentile; replace hydrophobic CDR3/FR4 residues'})"
    )

    # FR2 hallmark summary
    camelid_count = sum(1 for h in hallmarks if h.get("is_camelid_hallmark"))
    lines.append(f"  FR2 hallmarks: {camelid_count}/4 camelid positions confirmed")

    suffix = (
        "ALL CONSTRAINTS SATISFIED." if score == 4 else f"{4 - score} constraint(s) remaining."
    )
    lines.append(f"\nScore: {score}/4 — {suffix}")
    if score == 4:
        lines.append("Write your final summary of all mutations made.")
    else:
        # Carry-forward: always remind the agent of the best sequence so far so it
        # doesn't revert to an earlier, worse design after context truncation.
        if best_seq and best_seq != seq and best_score >= score:
            lines.append(
                f"\nBEST SEQUENCE SO FAR (score {best_score}/4) — start mutations from this:\n"
                f"```sequence\n{best_seq}\n```"
            )
        lines.append(
            "Analyse the failures above, apply targeted mutations to the sequence above, "
            "and write your revised full sequence in a ```sequence block."
        )

    return "\n".join(lines)


def run_screening_loop(
    seed_sequence: str | None = None,
    plot_name: str = "biophysical_trajectory",
    suppress_plot: bool = False,
    seed_label: str = "none",
) -> RunResult:
    """Run the generate → screen → critique → mutate loop.

    Args:
        seed_sequence: Optional starting sequence. If provided, the agent is
            asked to optimize it rather than designing from scratch.
        plot_name: Base name for the dashboard PNG (without extension).
        suppress_plot: If True, skip dashboard generation. Use for benchmark runs.
        seed_label: Human-readable seed name for RunResult (e.g. "naive").
    """
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print(
            "ERROR: Set TOGETHER_API_KEY environment variable before running.\n"
            "  Get one at https://api.together.xyz/settings/api-keys",
            file=sys.stderr,
        )
        sys.exit(1)

    model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    _display_label = seed_label if seed_label != "none" else "zero-shot"
    header_print(f"VHH-Screener — Developability Screening Loop ({_display_label})")
    cot_print(f"Session started: {datetime.now(UTC).isoformat()}")
    cot_print("Target: Human PD-1 (Pembrolizumab epitope)")
    cot_print("Scaffold: Camelid VHH nanobody")
    cot_print(f"Provider: Together AI ({BASE_URL})")
    cot_print(f"Model: {model_id}")
    cot_print(f"Max iterations: {MAX_ITERATIONS}")
    if seed_sequence:
        cot_print(f"Seed sequence: {seed_sequence[:40]}...")
    cot_print(f"CoT log: {COT_LOG.resolve()}\n")

    # Build user prompt — seeded or zero-shot
    if seed_sequence:
        user_prompt = (
            f"Optimize the following seed VHH sequence for targeting Human PD-1 "
            f"at the Pembrolizumab epitope. This sequence has known developability "
            f"issues. Follow the developability screening protocol exactly: first "
            f"run all four tools on this seed, then critique and mutate.\n\n"
            f"Seed sequence:\n{seed_sequence}"
        )
    else:
        user_prompt = (
            "Design a VHH nanobody targeting Human PD-1 at the "
            "Pembrolizumab epitope. Follow the developability screening "
            "protocol exactly. Begin by proposing your first candidate "
            "sequence, then immediately screen it with all four tools."
        )

    # Initial user message kicks off the loop
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Cost tracking
    total_input_tokens = 0
    total_output_tokens = 0

    # Per-iteration tracking for dashboard plot
    iteration_metrics: dict[int, dict] = {}  # iteration -> merged metrics

    # Capture iteration-0 baseline if a seed was provided
    if seed_sequence:
        bp = json.loads(calculate_biophysical_profile(seed_sequence))
        sl = json.loads(scan_structural_liabilities(seed_sequence))
        ap = json.loads(scan_aggregation_patches(seed_sequence))
        iteration_metrics[0] = {
            "iteration": 0,
            "pI": bp.get("isoelectric_point", 0),
            "gravy": bp.get("gravy", 0),
            "liability_count": sl.get("liability_count", 0),
            "apr_percentile": ap.get("candidate_max_patch", {}).get("percentile", 0),
        }
        cot_print("[Baseline] Seed sequence profiled as iteration 0")

    iteration = 0
    final_seq: str | None = None
    hit_limit = False
    _all_passed = False
    _consecutive_no_seq = 0  # guard against repeated parse failures
    best_seq: str | None = seed_sequence  # best sequence seen so far (highest score)
    best_score: int = 0  # score of best_seq

    for iteration in range(1, MAX_ITERATIONS + 1):
        header_print(f"ITERATION {iteration}")

        # Truncate message history — keep system + user prompt + last N messages
        if len(messages) > CONTEXT_KEEP_MESSAGES + 2:
            messages = messages[:2] + messages[-CONTEXT_KEEP_MESSAGES:]
            cot_print(f"[Context] Truncated history to {len(messages)} messages.")

        # --- Phase 1: LLM proposes a new sequence (text-only, no tool calling) ---
        # temperature=0 for reproducible benchmark results.
        # Double max_tokens on length failures (up to 3 doublings).
        _max_tokens = MAX_TOKENS
        for _attempt in range(4):
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=_max_tokens,
                temperature=0,
                messages=messages,
                # No tools= arg: pure text response, forces agent to reason and write a sequence
            )
            if response.choices[0].finish_reason != "length":
                break
            warn_print(
                f"[Warning] finish_reason=length at max_tokens={_max_tokens}. "
                f"Doubling to {_max_tokens * 2}."
            )
            _max_tokens *= 2

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        message = choice.message

        # Track token usage and cost
        if response.usage:
            iter_in = response.usage.prompt_tokens
            iter_out = response.usage.completion_tokens
            total_input_tokens += iter_in
            total_output_tokens += iter_out
            iter_cost = (
                iter_in * PRICE_PER_M_INPUT / 1_000_000 + iter_out * PRICE_PER_M_OUTPUT / 1_000_000
            )
            running_cost = (
                total_input_tokens * PRICE_PER_M_INPUT / 1_000_000
                + total_output_tokens * PRICE_PER_M_OUTPUT / 1_000_000
            )
            cot_print(
                f"[Cost] Iteration: {iter_in:,} in / {iter_out:,} out "
                f"= ${iter_cost:.4f}  |  Running total: ${running_cost:.4f}"
            )

        cot_print(f"[Iteration {iteration}] Finish reason: {finish_reason}")

        # Print reasoning
        text = message.content or ""
        if text:
            cot_print(f"\n[Agent CoT — Iteration {iteration}]")
            for line in text.splitlines():
                cot_print(f"  {line}")

        # Append assistant message to history
        messages.append(message.model_dump(exclude_none=True))

        # If agent signals it's done (all passed on prior iteration)
        if _all_passed:
            header_print("SCREENING LOOP COMPLETE")
            cot_print("Agent delivered final report.")
            break

        # --- Parse proposed sequence from agent text ---
        new_seq = extract_sequence(text)
        if not new_seq:
            _consecutive_no_seq += 1
            warn_print(
                f"[Warning] No valid sequence found in agent response "
                f"(attempt {_consecutive_no_seq})."
            )
            if _consecutive_no_seq >= 3:
                warn_print("[Warning] 3 consecutive parse failures — stopping loop.")
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "No valid VHH sequence was found in your response. "
                        "You MUST write your full proposed sequence (100-135 AA) "
                        "in a fenced code block:\n\n"
                        "```sequence\nEVQLV...\n```\n\n"
                        "Do not truncate it. Provide the complete sequence now."
                    ),
                }
            )
            continue
        _consecutive_no_seq = 0

        cot_print(f"\n[Sequence] Parsed {len(new_seq)} AA: {new_seq[:40]}...")
        final_seq = new_seq

        # --- Phase 2: Deterministic screening (no LLM, no tool calling) ---
        cot_print("[Screener] Running all 4 tools...")
        bp = json.loads(calculate_biophysical_profile(new_seq))
        sl = json.loads(scan_structural_liabilities(new_seq))
        ap = json.loads(scan_aggregation_patches(new_seq))
        ha = json.loads(vhh_hallmark_audit(new_seq))

        # Capture metrics for dashboard plot
        if iteration not in iteration_metrics:
            iteration_metrics[iteration] = {"iteration": iteration}
        m = iteration_metrics[iteration]
        if "error" not in bp:
            m["pI"] = bp["isoelectric_point"]
            m["gravy"] = bp["gravy"]
        if "error" not in sl:
            m["liability_count"] = sl["liability_count"]
        if "error" not in ap:
            m["apr_percentile"] = ap["candidate_max_patch"]["percentile"]

        # Check objective pass
        pi = bp.get("isoelectric_point")
        gravy = bp.get("gravy")
        liab = sl.get("liability_count")
        apr_pct = ap.get("candidate_max_patch", {}).get("percentile")
        _all_passed = (
            pi is not None
            and pi > 7.5
            and gravy is not None
            and gravy <= 0.0
            and liab == 0
            and apr_pct is not None
            and apr_pct < 95.0
        )

        # Update carry-forward best
        current_score = sum(
            [
                pi is not None and pi > 7.5,
                gravy is not None and gravy <= 0.0,
                liab == 0,
                apr_pct is not None and apr_pct < 95.0,
            ]
        )
        if current_score >= best_score:
            best_seq = new_seq
            best_score = current_score

        # Build rich results message and inject into conversation
        screening_msg = build_screening_message(
            new_seq, bp, sl, ap, ha, iteration, best_seq=best_seq, best_score=best_score
        )
        cot_print(screening_msg)
        messages.append({"role": "user", "content": screening_msg})

        if _all_passed:
            cot_print("[Screener] ALL CONSTRAINTS SATISFIED — running final report iteration.")
            # Loop continues one more time so agent can write its final summary

    else:
        hit_limit = True
        warn_print(
            f"\nWARNING: Reached max iterations ({MAX_ITERATIONS}) "
            "without converging on a passing design."
        )

    # Final cost summary
    final_cost = (
        total_input_tokens * PRICE_PER_M_INPUT / 1_000_000
        + total_output_tokens * PRICE_PER_M_OUTPUT / 1_000_000
    )
    header_print("COST SUMMARY")
    cot_print(f"Input tokens:  {total_input_tokens:,}")
    cot_print(f"Output tokens: {total_output_tokens:,}")
    cot_print(f"Total cost:    ${final_cost:.4f}")

    # Generate developability dashboard — fill missing metrics
    # Strategy: carry forward from previous; if no previous, back-fill from
    # the next known value. Track which points were imputed so the plot can
    # label them "NA".
    dashboard_points = sorted(iteration_metrics.values(), key=lambda x: x["iteration"])
    _fill_keys = ("pI", "gravy", "liability_count", "apr_percentile")

    # Forward fill
    for i in range(1, len(dashboard_points)):
        for key in _fill_keys:
            if key not in dashboard_points[i]:
                if key in dashboard_points[i - 1]:
                    dashboard_points[i][key] = dashboard_points[i - 1][key]
                    dashboard_points[i].setdefault("_imputed", set()).add(key)

    # Back-fill any remaining gaps (first points with no predecessor)
    for i in range(len(dashboard_points) - 2, -1, -1):
        for key in _fill_keys:
            if key not in dashboard_points[i]:
                if key in dashboard_points[i + 1]:
                    dashboard_points[i][key] = dashboard_points[i + 1][key]
                    dashboard_points[i].setdefault("_imputed", set()).add(key)

    if not suppress_plot:
        if len(dashboard_points) >= 2:
            plot_path = _plot_biophysical_trajectory(dashboard_points, plot_name)
            cot_print(f"Developability dashboard saved: {plot_path}")
        elif dashboard_points:
            cot_print("Only one iteration with data — skipping dashboard.")
        else:
            cot_print("No metric data captured — skipping dashboard.")

    # Write session summary to log
    cot_print(f"\nSession ended: {datetime.now(UTC).isoformat()}")
    cot_print(f"Total iterations: {iteration}")
    cot_print(f"Full CoT log: {COT_LOG.resolve()}")

    # --- Objective pass/fail: run all 4 tools on the final sequence ---
    final_pi: float | None = None
    final_gravy: float | None = None
    final_liability_count: int | None = None
    final_apr_percentile: float | None = None
    passed = False

    if final_seq:
        bp = json.loads(calculate_biophysical_profile(final_seq))
        sl = json.loads(scan_structural_liabilities(final_seq))
        ap = json.loads(scan_aggregation_patches(final_seq))

        if "error" not in bp:
            final_pi = bp.get("isoelectric_point")
            final_gravy = bp.get("gravy")
        if "error" not in sl:
            final_liability_count = sl.get("liability_count")
        if "error" not in ap:
            final_apr_percentile = ap.get("candidate_max_patch", {}).get("percentile")

        passed = (
            (final_pi is not None and final_pi > 7.5)
            and (final_gravy is not None and final_gravy <= 0.0)
            and (final_liability_count == 0)
            and (final_apr_percentile is not None and final_apr_percentile < 95.0)
        )

    result_label = "PASS" if passed else "FAIL"
    cot_print(f"\nObjective result: {result_label}")
    if final_seq:
        cot_print(
            f"  pI={final_pi:.2f}  GRAVY={final_gravy:.3f}  "
            f"liabilities={final_liability_count}  APR={final_apr_percentile:.1f}th%ile"
        )

    return RunResult(
        seed_label=seed_label,
        model=os.environ.get("MODEL_ID", DEFAULT_MODEL),
        passed=passed,
        iterations=iteration,
        hit_iteration_limit=hit_limit,
        total_cost_usd=final_cost,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        final_sequence=final_seq,
        final_pi=final_pi,
        final_gravy=final_gravy,
        final_liability_count=final_liability_count,
        final_apr_percentile=final_apr_percentile,
        timestamp=datetime.now(UTC).isoformat(),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VHH-Screener screening loop")
    parser.add_argument(
        "--seed",
        choices=["naive", "pembrolizumab", "none"],
        default="naive",
        help="Seed sequence: 'naive' (deliberately bad), 'pembrolizumab' (real VH), 'none' (zero-shot)",
    )
    parser.add_argument(
        "--plot-name",
        default="biophysical_trajectory",
        help="Base name for the dashboard PNG",
    )
    args = parser.parse_args()

    seed_map = {
        "naive": NAIVE_SEED,
        "pembrolizumab": PEMBROLIZUMAB_VH_SEED,
        "none": None,
    }
    result = run_screening_loop(
        seed_sequence=seed_map[args.seed],
        plot_name=args.plot_name,
        seed_label=args.seed,
    )
    sys.exit(0 if result.passed else 1)
