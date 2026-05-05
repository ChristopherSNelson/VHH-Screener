"""
antifold_inverse_fold.py - AntiFold inverse folding for VHH CDR optimization.

Given a 3D structure (PDB file from Boltz-2), redesigns CDR sequences using
AntiFold — a purpose-built antibody inverse folding model that outperforms
ProteinMPNN on CDR sequence recovery benchmarks.

AntiFold is trained on paired antibody structures from SAbDab and uses
ESM-IF1 as its backbone, fine-tuned with antibody-specific CDR masking.

GPU is strongly recommended but not required. CPU inference is ~10x slower.

References:
  Hoie et al., AntiFold: Improved antibody structure-based design using
  inverse folding. NeurIPS 2023 workshop.
  https://github.com/oxpig/AntiFold

Prerequisite: predict_vhh_complex_structure must be run first to obtain
a PDB file. Pass the structure_path from that tool's output here.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("vhh-screener")

_VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

# CDR definitions in IMGT numbering (standard for AntiFold)
_CDR_IMGT_RANGES = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}


def _check_antifold_installed() -> bool:
    """Return True if AntiFold Python package is importable."""
    try:
        import antifold  # noqa: F401

        return True
    except ImportError:
        return False


def optimize_cdrs(
    pdb_path: str,
    vhh_chain: str = "A",
    cdrs_to_redesign: list[str] | None = None,
    num_samples: int = 10,
    temperature: float = 0.20,
    device: str = "cpu",
) -> dict[str, Any]:
    """Redesign VHH CDR sequences conditioned on a 3D scaffold.

    Args:
        pdb_path: Path to a PDB file (output from predict_vhh_complex_structure).
        vhh_chain: Chain ID of the VHH in the PDB file. Default "A".
        cdrs_to_redesign: List of CDRs to redesign, e.g. ["CDR3"] or ["CDR1", "CDR2", "CDR3"].
                         Defaults to CDR3 only (lowest disruption to binding geometry).
        num_samples: Number of sequence candidates to generate. Default 10.
        temperature: Sampling temperature. Lower = more conservative (closer to WT).
                    0.20 is AntiFold's recommended developability-focused setting.
        device: "cpu", "mps", or "cuda". GPU recommended.

    Returns:
        dict with:
            status: "success" | "error" | "antifold_not_installed"
            redesigned_sequences: list of dicts with sequence, cdr_sequences, score
            fixed_regions: framework regions held constant during redesign
            cdrs_redesigned: list of CDR names that were sampled
            pdb_path: input structure path
            error: str | None
    """
    if cdrs_to_redesign is None:
        cdrs_to_redesign = ["CDR3"]

    pdb = Path(pdb_path)
    if not pdb.exists():
        return {
            "status": "error",
            "error": f"PDB file not found: {pdb_path}. Run predict_vhh_complex_structure first.",
            "redesigned_sequences": [],
        }

    if not _check_antifold_installed():
        return {
            "status": "antifold_not_installed",
            "error": (
                "AntiFold is not installed. Install with: pip install antifold\n"
                "Then re-run. GPU strongly recommended (CPU inference is ~10x slower)."
            ),
            "redesigned_sequences": [],
            "install_command": "pip install antifold",
        }

    try:
        import antifold
        from antifold.main import run_antifold

        # AntiFold expects a CSV specifying which chains and positions to redesign.
        # Format: pdb_path, heavy_chain, light_chain, [optional per-residue mask]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Write input CSV
            csv_path = tmp / "input.csv"
            csv_path.write_text(f"pdb_path,heavy_chain,light_chain\n{pdb_path},{vhh_chain},\n")
            out_dir = tmp / "antifold_out"
            out_dir.mkdir()

            run_antifold(
                pdb_or_zip_dir=str(pdb_path),
                out_dir=str(out_dir),
                heavy_chain=vhh_chain,
                light_chain="",
                num_seq_per_target=num_samples,
                sampling_temp=temperature,
                regions=cdrs_to_redesign,
                device=device,
            )

            # Parse output FASTA files
            results = _parse_antifold_output(out_dir, pdb.stem, cdrs_to_redesign)

        logger.info(
            "optimize_cdrs | pdb=%s | cdrs=%s | n_samples=%d | n_results=%d",
            pdb_path,
            cdrs_to_redesign,
            num_samples,
            len(results),
        )

        return {
            "status": "success",
            "pdb_path": pdb_path,
            "vhh_chain": vhh_chain,
            "cdrs_redesigned": cdrs_to_redesign,
            "temperature": temperature,
            "num_samples": num_samples,
            "redesigned_sequences": results,
            "note": (
                "Sequences are ranked by AntiFold log-likelihood (higher = better). "
                "Run all 4 developability screens on each candidate before advancing."
            ),
        }

    except Exception as exc:
        logger.exception("optimize_cdrs failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "redesigned_sequences": [],
        }


def _parse_antifold_output(
    out_dir: Path,
    pdb_stem: str,
    cdrs_redesigned: list[str],
) -> list[dict[str, Any]]:
    """Parse AntiFold output FASTA into a list of candidate dicts."""
    results = []
    fasta_files = list(out_dir.glob("**/*.fasta")) + list(out_dir.glob("**/*.fa"))
    if not fasta_files:
        return results

    for fa in fasta_files:
        current_header: str | None = None
        current_seq: list[str] = []
        for line in fa.read_text().splitlines():
            line = line.strip()
            if line.startswith(">"):
                if current_header and current_seq:
                    seq = "".join(current_seq)
                    score = _parse_score_from_header(current_header)
                    results.append(
                        {
                            "sequence": seq,
                            "log_likelihood": score,
                            "header": current_header,
                        }
                    )
                current_header = line[1:]
                current_seq = []
            elif re.match(r"^[A-Za-z*-]+$", line):
                current_seq.append(line.replace("-", "").replace("*", ""))
        if current_header and current_seq:
            seq = "".join(current_seq)
            score = _parse_score_from_header(current_header)
            results.append(
                {
                    "sequence": seq,
                    "log_likelihood": score,
                    "header": current_header,
                }
            )

    # Sort descending by log-likelihood
    results.sort(key=lambda x: x.get("log_likelihood", float("-inf")), reverse=True)
    return results


def _parse_score_from_header(header: str) -> float | None:
    """Extract log-likelihood from AntiFold FASTA header."""
    # AntiFold headers look like: >sample_1, score=1.234, ...
    m = re.search(r"score=([0-9.\-]+)", header)
    if m:
        return float(m.group(1))
    return None
