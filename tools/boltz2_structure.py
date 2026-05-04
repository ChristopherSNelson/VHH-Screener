"""
boltz2_structure.py - Boltz-2 structure prediction for VHH-antigen complexes.

Wraps the `boltz predict` CLI to run Boltz-2 on a VHH nanobody and optional
antigen sequence, then parses the confidence JSON output into a structured
result for the screening pipeline.

Boltz-2 is the recommended open-source structure predictor for antibody-antigen
complexes (MIT license, surpasses AlphaFold-Multimer on antibody benchmarks).

GPU requirements:
  - Minimum: 8 GB VRAM (A10G, RTX 3080, T4)
  - Recommended: 24 GB VRAM (A10G, A100) for complex with recycling_steps=3
  - CPU inference is possible but ~100x slower (not practical for screening)

References:
  Wohlwend et al., Boltz-2: Democratizing Biomolecular Structure Prediction
  https://github.com/jwohlwend/boltz

Binding strategy reference:
  Escalante 180-line approach:
  https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Standard 20 amino acids accepted by Boltz-2
_VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

# Human PD-1 ectodomain (PDB 4ZQK chain B; 118 AA)
# Extracellular IgV-like domain used in co-crystal structures with PD-L1.
# Contains the C93S engineering mutation common in PD-1 expression constructs.
# Used as the default antigen target for PD-1/VHH complex prediction.
PD1_ECTODOMAIN = (
    "NPPTFSPALLVVTEGDNATFTCSFSNTSESFVLNWYRMSPSNQTDKLAAFPEDRSQPGQDS"
    "RFRVTQLPNGRDFHMSVVRARRNDSGTYLCGAISLAPKAQIKESLRAELRVTERRAE"
)


def _clean(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def _validate_sequence(seq: str, label: str) -> str | None:
    """Return None if valid, or an error message string."""
    cleaned = _clean(seq)
    if not cleaned:
        return f"{label}: empty sequence after stripping whitespace/digits"
    invalid = set(cleaned) - _VALID_AAS
    if invalid:
        return (
            f"{label}: non-standard residues {sorted(invalid)} - Boltz-2 requires standard 20 AAs"
        )
    if len(cleaned) < 10:
        return f"{label}: sequence too short ({len(cleaned)} AA) - minimum 10 required"
    return None


def _write_boltz_yaml(
    vhh_seq: str,
    antigen_seq: str | None,
    output_path: Path,
) -> None:
    """Write a Boltz-2 YAML input file for a VHH or VHH-antigen complex."""
    chains: list[dict[str, Any]] = [{"protein": {"id": "A", "sequence": vhh_seq}}]
    if antigen_seq:
        chains.append({"protein": {"id": "B", "sequence": antigen_seq}})

    with output_path.open("w") as f:
        # Write manually to keep the format clean and Boltz-compatible
        f.write("version: 1\n")
        f.write("sequences:\n")
        for chain in chains:
            entity = "protein"
            chain_id = chain[entity]["id"]
            seq = chain[entity]["sequence"]
            f.write("  - protein:\n")
            f.write(f"      id: {chain_id}\n")
            f.write(f'      sequence: "{seq}"\n')


def _parse_confidence_json(confidence_path: Path) -> dict[str, Any]:
    """Parse Boltz-2 confidence JSON into a flat, agent-readable dict."""
    with confidence_path.open() as f:
        raw = json.load(f)

    # Core metrics - most important for developability decision
    result: dict[str, Any] = {
        "confidence_score": raw.get("confidence_score"),
        "ptm": raw.get("ptm"),  # Predicted TM-score (global fold quality)
        "iptm": raw.get("iptm"),  # Interface TM-score (binding quality)
        "complex_plddt": raw.get("complex_plddt"),  # Per-residue confidence avg
    }

    # Interface confidence between VHH (chain A) and antigen (chain B)
    pair_iptm = raw.get("pair_chains_iptm", {})
    if "A" in pair_iptm and "B" in pair_iptm.get("A", {}):
        result["vhh_antigen_iptm"] = pair_iptm["A"]["B"]
    elif "0" in pair_iptm and "1" in pair_iptm.get("0", {}):
        result["vhh_antigen_iptm"] = pair_iptm["0"]["1"]
    else:
        result["vhh_antigen_iptm"] = None

    # Quality interpretation
    iptm = result["iptm"]
    if iptm is not None:
        if iptm >= 0.8:
            interpretation = "High confidence complex - strong predicted binding interface"
        elif iptm >= 0.6:
            interpretation = (
                "Medium confidence - interface geometry plausible, verify experimentally"
            )
        else:
            interpretation = "Low confidence - interface uncertain, consider redesign"
        result["interpretation"] = interpretation

    return result


def predict_structure(
    vhh_sequence: str,
    antigen_sequence: str | None = None,
    out_dir: str | None = None,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Boltz-2 structure prediction on a VHH or VHH-antigen complex.

    Args:
        vhh_sequence: Single-letter VHH amino acid sequence.
        antigen_sequence: Optional antigen sequence. If None, predicts the
            VHH monomer structure only. Defaults to PD-1 ectodomain if
            called via the MCP tool without an explicit antigen.
        out_dir: Directory for Boltz-2 output. Defaults to a temp dir.
        accelerator: "gpu", "cpu", or "tpu". Use "cpu" for testing without GPU.
        recycling_steps: Boltz-2 recycling iterations (default 3, max 10).
        diffusion_samples: Number of structure samples (default 1 for speed).
        dry_run: If True, validate inputs and write the YAML but skip inference.
            Returns a structured placeholder result. Used for testing and for
            validating inputs before queueing a GPU job.

    Returns:
        dict with keys:
            status: "success" | "dry_run" | "error"
            vhh_length: int
            antigen_length: int | None
            input_yaml: str (path to the Boltz-2 input file)
            output_dir: str | None (path to predictions)
            confidence: dict | None (parsed confidence JSON)
            structure_path: str | None (path to best .cif file)
            error: str | None
    """
    vhh_clean = _clean(vhh_sequence)
    antigen_clean = _clean(antigen_sequence) if antigen_sequence else None

    # Validate
    err = _validate_sequence(vhh_clean, "VHH")
    if err:
        return {"status": "error", "error": err}

    if antigen_clean:
        err = _validate_sequence(antigen_clean, "antigen")
        if err:
            return {"status": "error", "error": err}

    # Set up output dir
    tmp_dir = None
    if out_dir:
        pred_dir = Path(out_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="boltz2_")
        pred_dir = Path(tmp_dir)

    # Write YAML input
    input_yaml = pred_dir / "vhh_complex.yaml"
    _write_boltz_yaml(vhh_clean, antigen_clean, input_yaml)

    base_result: dict[str, Any] = {
        "vhh_length": len(vhh_clean),
        "antigen_length": len(antigen_clean) if antigen_clean else None,
        "input_yaml": str(input_yaml),
        "output_dir": str(pred_dir),
        "confidence": None,
        "structure_path": None,
        "error": None,
    }

    if dry_run:
        base_result["status"] = "dry_run"
        base_result["message"] = (
            "Input validated and YAML written. Re-run with dry_run=False on a "
            "GPU-equipped machine (min 8 GB VRAM) to execute Boltz-2 inference. "
            "Recommended: g5.xlarge on AWS us-west-2 (~$1/hr) or A10G on Lambda Labs."
        )
        return base_result

    # Run Boltz-2 inference
    cmd = [
        "boltz",
        "predict",
        str(input_yaml),
        "--out_dir",
        str(pred_dir),
        "--accelerator",
        accelerator,
        "--recycling_steps",
        str(recycling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--model",
        "boltz2",
        "--output_format",
        "pdb",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {
            **base_result,
            "status": "error",
            "error": "Boltz-2 inference timed out (>10 min). Check GPU availability.",
        }
    except FileNotFoundError:
        return {
            **base_result,
            "status": "error",
            "error": "boltz CLI not found. Install with: pip install boltz",
        }

    if proc.returncode != 0:
        return {
            **base_result,
            "status": "error",
            "error": f"boltz predict failed (exit {proc.returncode}): {proc.stderr[-500:]}",
        }

    # Find output files - Boltz writes to {out_dir}/predictions/vhh_complex/
    predictions_dir = pred_dir / "predictions" / "vhh_complex"

    confidence_path = predictions_dir / "confidence_vhh_complex_model_0.json"
    structure_glob = list(predictions_dir.glob("vhh_complex_model_0.pdb"))

    if not confidence_path.exists():
        return {
            **base_result,
            "status": "error",
            "error": f"Confidence JSON not found at {confidence_path}. "
            f"Boltz stdout: {proc.stdout[-300:]}",
        }

    confidence = _parse_confidence_json(confidence_path)
    structure_path = str(structure_glob[0]) if structure_glob else None

    return {
        **base_result,
        "status": "success",
        "confidence": confidence,
        "structure_path": structure_path,
    }
