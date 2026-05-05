"""
VHH-Screener — FastMCP Developability Screening Server
=====================================================

Provides deterministic, agent-readable tools for evaluating
developability constraints on VHH (nanobody) sequences.

Biophysical thresholds follow standard developability guidance:
  - pI  > 7.5   → acceptable (avoids precipitation near physiological pH)
  - GRAVY ≤ 0.0 → acceptable (hydrophilic → lower aggregation propensity)

References:
  Kyte & Doolittle (1982) for hydropathy; IPC2 / Bjellqvist for pI.
  Escalante blog for the broader Nipah VHH design strategy:
  https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from fastmcp import FastMCP

# --- Logging setup ------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("vhh-screener")
logger.setLevel(logging.INFO)

_handler = logging.FileHandler(LOG_DIR / "biologics_server.log")
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_handler)

mcp = FastMCP("VHH-Screener")


def _clean_sequence(seq: str) -> str:
    """Strip whitespace/digits and upper-case a raw protein sequence."""
    return re.sub(r"[^A-Za-z]", "", seq).upper()


@mcp.tool()
def calculate_biophysical_profile(sequence: str) -> str:
    """Calculate pI and GRAVY for a protein sequence and flag aggregation risk.

    Args:
        sequence: Single-letter amino-acid sequence (whitespace/digits ignored).

    Returns:
        JSON string with fields:
            sequence_length, isoelectric_point, gravy,
            pI_flag, gravy_flag, overall_risk, flags (list of human-readable warnings).
    """
    seq = _clean_sequence(sequence)

    if not seq:
        return json.dumps({"error": "Empty or invalid sequence provided."})

    # Reject non-standard residues that ProteinAnalysis cannot handle
    invalid = set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
    if invalid:
        return json.dumps(
            {
                "error": f"Non-standard residues detected: {sorted(invalid)}. "
                "Remove or replace them before profiling.",
            }
        )

    analysis = ProteinAnalysis(seq)

    pi = round(analysis.isoelectric_point(), 2)
    gravy = round(analysis.gravy(), 4)

    # --- Flag logic -----------------------------------------------------------
    flags: list[str] = []

    pi_flag = "PASS" if pi > 7.5 else "FAIL"
    if pi_flag == "FAIL":
        flags.append(
            f"pI = {pi} (< 7.5): High risk of precipitation near physiological pH. "
            "Consider charge-engineering (Lys/Arg substitutions in framework)."
        )

    gravy_flag = "PASS" if gravy <= 0.0 else "FAIL"
    if gravy_flag == "FAIL":
        flags.append(
            f"GRAVY = {gravy} (> 0.0): Elevated hydrophobicity — aggregation-prone. "
            "Inspect solvent-exposed hydrophobic patches."
        )

    overall_risk = (
        "Low" if (pi_flag == "PASS" and gravy_flag == "PASS") else "High-Risk for Aggregation"
    )

    report = {
        "sequence_length": len(seq),
        "isoelectric_point": pi,
        "gravy": gravy,
        "pI_flag": pi_flag,
        "gravy_flag": gravy_flag,
        "overall_risk": overall_risk,
        "flags": flags,
    }

    result = json.dumps(report, indent=2)
    logger.info("biophysical_profile | %s", result)
    return result


# --- Structural liability patterns --------------------------------------------
# Deterministic regex rules — no LLM inference. These represent well-characterized
# post-translational modification hotspots that cause manufacturing failure.

_LIABILITY_PATTERNS: list[tuple[str, str, str]] = [
    # (motif_name, regex_pattern, mechanism)
    ("Deamidation", r"N[GSA]", "Asn deamidation via succinimide intermediate"),
    ("Isomerization", r"DG", "Asp isomerization to iso-Asp via succinimide"),
    (
        "N-Glycosylation",
        r"N[^P][ST]",
        "N-linked glycosylation sequon (Asn-X-Ser/Thr, X≠Pro)",
    ),
]

# Pre-compile for performance
_COMPILED_LIABILITIES: list[tuple[str, re.Pattern[str], str]] = [
    (name, re.compile(pat), mech) for name, pat, mech in _LIABILITY_PATTERNS
]


@mcp.tool()
def scan_structural_liabilities(sequence: str) -> str:
    """Scan a protein sequence for post-translational modification hotspots.

    Identifies deterministic sequence motifs that are known to cause
    manufacturing failures in biologics:
      - Deamidation: NG, NS, NA
      - Isomerization: DG
      - N-glycosylation: N[^P][ST] (Asn-X-Ser/Thr where X ≠ Pro)

    Args:
        sequence: Single-letter amino-acid sequence.

    Returns:
        JSON string with fields:
            sequence_length, liabilities (list of hits), liability_count,
            overall_flag ("PASS" or "FAIL").
    """
    seq: str = _clean_sequence(sequence)

    if not seq:
        return json.dumps({"error": "Empty or invalid sequence provided."})

    liabilities: list[dict[str, str | int]] = []

    for name, pattern, mechanism in _COMPILED_LIABILITIES:
        for match in pattern.finditer(seq):
            # Show surrounding context so LLMs can locate the motif visually
            start = match.start()
            ctx_left = seq[max(0, start - 5) : start]
            ctx_right = seq[match.end() : match.end() + 5]
            context = f"...{ctx_left}[{match.group()}]{ctx_right}..."

            liabilities.append(
                {
                    "liability_type": name,
                    "motif": match.group(),
                    "position": match.start() + 1,  # 1-based for biologists
                    "context": context,
                    "mechanism": mechanism,
                }
            )

    # Sort by position for readability
    liabilities.sort(key=lambda h: h["position"])

    report: dict = {
        "sequence_length": len(seq),
        "liabilities": liabilities,
        "liability_count": len(liabilities),
        "overall_flag": "FAIL" if liabilities else "PASS",
    }

    result: str = json.dumps(report, indent=2)
    logger.info("scan_structural_liabilities | %s", result)
    return result


# --- VHH Hallmark Tetrad Audit ------------------------------------------------
# Kabat/Chothia FR2 positions that distinguish camelid VHH from conventional VH.
# Canonical camelid residues: F37, E44, R45, G47
# Humanizing substitutions: V37, G44, L45, W47
#
# WARNING: Humanization of these positions can destabilize the VHH. The camelid
# hallmarks compensate for the absence of VL by providing a hydrophilic interface
# where conventional VH has a hydrophobic VH-VL contact surface.

_HALLMARK_POSITIONS: list[dict[str, str | int]] = [
    {
        "kabat_position": 37,
        "camelid_residue": "F",
        "human_vh_residue": "V",
        "role": "Core packing; compensates for missing VL contact",
    },
    {
        "kabat_position": 44,
        "camelid_residue": "E",
        "human_vh_residue": "G",
        "role": "Hydrophilic substitution at former VH-VL interface",
    },
    {
        "kabat_position": 45,
        "camelid_residue": "R",
        "human_vh_residue": "L",
        "role": "Charged residue replacing hydrophobic VL contact",
    },
    {
        "kabat_position": 47,
        "camelid_residue": "G",
        "human_vh_residue": "W",
        "role": "Flexible Gly replacing bulky Trp at VL interface",
    },
]

_HUMANIZATION_WARNING: str = (
    "Humanization of FR2 hallmarks in VHH can significantly reduce solubility "
    "and increase aggregation propensity. The camelid tetrad (F37/E44/R45/G47) "
    "evolved to compensate for the absence of VL. Reverting to human VH residues "
    "re-exposes the hydrophobic VH-VL interface without a binding partner, "
    "often leading to self-association. Proceed only with experimental validation."
)


@mcp.tool()
def vhh_hallmark_audit(sequence: str, framework2_start: int = 36) -> str:
    """Audit FR2 hallmark positions for camelid vs. human VH identity.

    Checks Kabat/Chothia positions 37, 44, 45, and 47 for the canonical
    camelid VHH tetrad (F, E, R, G). If camelid residues are present,
    suggests humanizing mutations but warns about solubility trade-offs.

    Args:
        sequence: Single-letter amino-acid VHH sequence.
        framework2_start: 0-based index where FR2 begins in the linear
            sequence. Default 36 assumes standard VHH numbering where
            Kabat position 36 maps to index 35 (0-based), making position
            37 = index 36. Adjust if your numbering differs.

    Returns:
        JSON string with per-position audit, humanization suggestions,
        and an overall assessment.
    """
    seq: str = _clean_sequence(sequence)

    if not seq:
        return json.dumps({"error": "Empty or invalid sequence provided."})

    # Map Kabat positions to 0-based sequence indices.
    # Kabat 37 → framework2_start + (37 - 37) = framework2_start + 0
    # Offsets relative to Kabat 37:
    kabat_to_offset: dict[int, int] = {37: 0, 44: 7, 45: 8, 47: 10}

    audits: list[dict[str, str | int | bool]] = []
    camelid_count: int = 0

    for hallmark in _HALLMARK_POSITIONS:
        kabat_pos: int = hallmark["kabat_position"]  # type: ignore[assignment]
        offset: int = kabat_to_offset[kabat_pos]
        seq_index: int = framework2_start + offset

        if seq_index >= len(seq):
            audits.append(
                {
                    "kabat_position": kabat_pos,
                    "status": "ERROR",
                    "detail": f"Sequence too short to contain Kabat position {kabat_pos} "
                    f"(need index {seq_index}, have {len(seq)} residues).",
                }
            )
            continue

        observed: str = seq[seq_index]
        is_camelid: bool = observed == hallmark["camelid_residue"]
        is_human: bool = observed == hallmark["human_vh_residue"]

        if is_camelid:
            camelid_count += 1

        audit_entry: dict[str, str | int | bool] = {
            "kabat_position": kabat_pos,
            "sequence_index": seq_index + 1,  # 1-based
            "observed_residue": observed,
            "expected_camelid": hallmark["camelid_residue"],
            "expected_human": hallmark["human_vh_residue"],
            "is_camelid_hallmark": is_camelid,
            "is_human_vh": is_human,
            "role": hallmark["role"],
        }

        if is_camelid:
            audit_entry["suggestion"] = (
                f"Humanize {observed}{kabat_pos}{hallmark['human_vh_residue']} "
                f"if regulatory/immunogenicity concerns require it."
            )

        audits.append(audit_entry)

    # Overall assessment
    if camelid_count == 4:
        identity = "Fully camelid VHH (all 4 hallmarks present)"
    elif camelid_count == 0:
        identity = "Fully humanized FR2 (no camelid hallmarks)"
    else:
        identity = f"Chimeric FR2 ({camelid_count}/4 camelid hallmarks)"

    report: dict = {
        "sequence_length": len(seq),
        "framework2_start_index": framework2_start + 1,  # 1-based
        "hallmark_audit": audits,
        "camelid_hallmark_count": camelid_count,
        "identity": identity,
    }

    if camelid_count > 0:
        report["humanization_warning"] = _HUMANIZATION_WARNING

    result: str = json.dumps(report, indent=2)
    logger.info("vhh_hallmark_audit | %s", result)
    return result


# --- Aggregation-Prone Region (APR) scanning ---------------------------------
# Sliding-window hydrophobicity analysis calibrated against clinical-stage
# therapeutics. Instead of arbitrary cutoffs, each patch is scored as a z-score
# relative to a pre-computed distribution of max-patch hydrophobicity from
# approved/clinical-stage antibody VH and VHH domains.
#
# Calibration set (n=13): sequences extracted from public PDB structures and
# patent filings for approved/clinical-stage therapeutics.
#   VHH (3): Caplacizumab, Ozoralizumab, Envafolimab
#   mAb VH (10): Pembrolizumab, Nivolumab, Trastuzumab, Adalimumab, Rituximab,
#                Bevacizumab, Atezolizumab, Durvalumab, Ipilimumab, Crizanlizumab
#
# A design fails screening only if its worst patch exceeds the 95th percentile
# of successful clinical antibodies — grounding the constraint in empirical
# manufacturing survival, not textbook heuristics.

_KD_SCALE: dict[str, float] = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "D": -3.5,
    "E": -3.5,
    "N": -3.5,
    "Q": -3.5,
    "K": -3.9,
    "R": -4.5,
}

_APR_WINDOW_SIZE = 7

# --- Clinical-stage calibration set ------------------------------------------
# Pre-computed max 7-residue KD window scores for approved/clinical VH and VHH
# domains. Used to derive the reference distribution for z-score calculation.

_CST_MAX_PATCH_SCORES: dict[str, float] = {
    "Caplacizumab_VHH": 1.686,  # PDB 7EOW chain B (verified crystal structure)
    "Ozoralizumab_VHH": 2.086,
    "Envafolimab_VHH": 1.371,
    "Pembrolizumab_VH": 1.029,
    "Nivolumab_VH": 2.014,
    "Trastuzumab_VH": 1.371,
    "Adalimumab_VH": 1.371,
    "Rituximab_VH": 1.400,
    "Bevacizumab_VH": 1.371,
    "Atezolizumab_VH": 1.371,
    "Durvalumab_VH": 1.371,
    "Ipilimumab_VH": 1.457,
    "Crizanlizumab_VH": 1.029,
}

# Distribution statistics (7-residue KD window, n=13 clinical-stage therapeutics)
_CST_MEAN = 1.456  # mean of max-patch scores
_CST_STD = 0.313  # sample standard deviation
_CST_N = 13

# Screening threshold: 95th percentile (parametric, one-tailed)
_APR_SCREENING_THRESHOLD = round(_CST_MEAN + 1.645 * _CST_STD, 3)  # ~1.934

# Gold standard: Caplacizumab (first approved VHH, anti-vWF; PDB 7EOW)
_CAPLACIZUMAB_MAX_PATCH = 1.686


def _compute_patch_z_score(max_patch_kd: float) -> float:
    """Compute z-score of a max-patch KD score against the CST distribution."""
    if _CST_STD == 0:
        return 0.0
    return round((max_patch_kd - _CST_MEAN) / _CST_STD, 2)


def _compute_patch_percentile(z_score: float) -> float:
    """Approximate percentile from z-score using the error function.

    Uses math.erf for a closed-form normal CDF — no scipy dependency.
    """
    import math

    cdf = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    return round(cdf * 100, 1)


@mcp.tool()
def scan_aggregation_patches(
    sequence: str,
    window_size: int = _APR_WINDOW_SIZE,
) -> str:
    """Scan for aggregation-prone regions using clinically-calibrated sliding-window
    hydrophobicity.

    Each 7-residue window is scored on the Kyte-Doolittle scale and compared
    against a pre-computed distribution of max-patch scores from 13
    clinical-stage antibody VH/VHH domains (Caplacizumab, Pembrolizumab,
    Trastuzumab, etc.). A design fails screening only if its worst hydrophobic
    patch exceeds the 95th percentile of successful clinical therapeutics.

    Returns z-scores and percentiles relative to the clinical-stage reference
    distribution, plus the Caplacizumab gold-standard comparison.

    Args:
        sequence: Single-letter amino-acid sequence.
        window_size: Sliding window width (default 7).

    Returns:
        JSON string with per-patch details, z-scores, percentiles, clinical
        comparison, and overall PASS/FAIL flag.
    """
    seq = _clean_sequence(sequence)

    if not seq:
        return json.dumps({"error": "Empty or invalid sequence provided."})

    invalid = set(seq) - set(_KD_SCALE.keys())
    if invalid:
        return json.dumps({"error": f"Non-standard residues detected: {sorted(invalid)}."})

    if len(seq) < window_size:
        return json.dumps(
            {"error": f"Sequence too short ({len(seq)} aa) for window size {window_size}."}
        )

    scores = [_KD_SCALE[aa] for aa in seq]

    # Compute all window means
    window_means: list[tuple[int, float, str]] = []
    for i in range(len(seq) - window_size + 1):
        mean_kd = sum(scores[i : i + window_size]) / window_size
        window_means.append((i, mean_kd, seq[i : i + window_size]))

    # Find the max patch score for the candidate
    max_patch_kd = max(m for _, m, _ in window_means)
    max_z = _compute_patch_z_score(max_patch_kd)
    max_percentile = _compute_patch_percentile(max_z)

    # Flag patches that exceed the 95th percentile screening threshold
    flagged_patches: list[dict[str, str | int | float]] = []
    for i, mean_kd, patch_seq in window_means:
        if mean_kd >= _APR_SCREENING_THRESHOLD:
            z = _compute_patch_z_score(mean_kd)
            flagged_patches.append(
                {
                    "start_position": i + 1,
                    "end_position": i + window_size,
                    "patch_sequence": patch_seq,
                    "mean_hydrophobicity": round(mean_kd, 3),
                    "z_score": z,
                    "percentile": _compute_patch_percentile(z),
                    "suggestion": (
                        f"Break hydrophobic patch '{patch_seq}' by introducing a "
                        f"polar residue (Ser, Thr, Asn, Asp, or Glu) at a "
                        f"solvent-exposed position within residues "
                        f"{i + 1}-{i + window_size}."
                    ),
                }
            )

    # Determine pass/fail against the clinical screening threshold
    failed = max_patch_kd >= _APR_SCREENING_THRESHOLD

    report: dict = {
        "sequence_length": len(seq),
        "window_size": window_size,
        "calibration": {
            "reference": "Clinical-stage therapeutics (n=13 VH/VHH domains)",
            "cst_mean": _CST_MEAN,
            "cst_std": _CST_STD,
            "screening_threshold_95th": _APR_SCREENING_THRESHOLD,
            "caplacizumab_max_patch": _CAPLACIZUMAB_MAX_PATCH,
        },
        "candidate_max_patch": {
            "mean_hydrophobicity": round(max_patch_kd, 3),
            "z_score": max_z,
            "percentile": max_percentile,
            "vs_caplacizumab": (
                "BETTER"
                if max_patch_kd <= _CAPLACIZUMAB_MAX_PATCH
                else "WORSE"
                if max_patch_kd > _APR_SCREENING_THRESHOLD
                else "ACCEPTABLE"
            ),
        },
        "flagged_patches": flagged_patches,
        "flagged_patch_count": len(flagged_patches),
        "overall_flag": "FAIL" if failed else "PASS",
    }

    if failed:
        report["interpretation"] = (
            f"Candidate max patch ({round(max_patch_kd, 3)}) exceeds the 95th "
            f"percentile ({_APR_SCREENING_THRESHOLD}) of clinical-stage "
            f"therapeutics (z={max_z}, {max_percentile}th percentile). "
            f"The design has a hydrophobic patch worse than >95% of successfully "
            f"manufactured antibodies. Introduce polar substitutions to break "
            f"the patch."
        )
    else:
        report["interpretation"] = (
            f"Candidate max patch ({round(max_patch_kd, 3)}) is within the "
            f"clinical-stage distribution (z={max_z}, {max_percentile}th "
            f"percentile). No aggregation-prone regions exceed the screening "
            f"threshold."
        )

    result = json.dumps(report, indent=2)
    logger.info("scan_aggregation_patches | %s", result)
    return result


@mcp.tool()
def filter_liabilities_by_sasa(
    pdb_path: str,
    sasa_threshold: float = 25.0,
    chain_id: str | None = None,
) -> str:
    """Filter PTM liabilities by solvent-accessible surface area (SASA).

    Loads a PDB structure, computes per-residue SASA with FreeSASA, then
    re-runs scan_structural_liabilities on the extracted chain sequence and
    filters to only those motifs where at least one residue has SASA above
    the threshold. Buried motifs (SASA < 25 Å²) are rarely modified in
    practice and generate false positives that waste agent iterations.

    Designed to be called after predict_vhh_complex_structure has produced
    a PDB file. CPU-only; no GPU required.

    Args:
        pdb_path: Path to a PDB file.
        sasa_threshold: Minimum per-residue SASA (Å²) to consider a residue
            exposed. Default 25.0 Å² (standard developability threshold).
        chain_id: PDB chain to analyse. If None, uses the first chain found.

    Returns:
        JSON string with fields:
            chain_id: str — chain analysed
            sequence: str — sequence extracted from that chain
            sasa_threshold: float
            total_liabilities: int — all motifs found in sequence
            exposed_liabilities: int — motifs with at least one exposed residue
            buried_liabilities: int — motifs fully buried (filtered out)
            exposed: list of liability dicts (same schema as scan_structural_liabilities)
                     each with an added sasa_values field (per-residue SASA in Å²)
            buried: list of fully buried liability dicts
    """
    try:
        import freesasa
    except ImportError:
        return json.dumps(
            {"error": "freesasa not installed. Run: conda install -c conda-forge freesasa"}
        )

    pdb_file = Path(pdb_path)
    if not pdb_file.exists():
        return json.dumps({"error": f"PDB file not found: {pdb_path}"})

    # --- Parse PDB and pick chain ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("vhh", str(pdb_file))
    model = next(structure.get_models())

    chains = list(model.get_chains())
    if not chains:
        return json.dumps({"error": "No chains found in PDB file."})

    if chain_id is not None:
        chain = model[chain_id] if chain_id in model else None
        if chain is None:
            available = [c.id for c in chains]
            return json.dumps({"error": f"Chain '{chain_id}' not found. Available: {available}"})
    else:
        chain = chains[0]
        chain_id = chain.id

    # --- Extract sequence and build residue index map ---
    # Only standard amino acids; skip HETATMs and waters.
    _aa3 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    residues = [r for r in chain.get_residues() if r.get_resname().strip() in _aa3]
    if not residues:
        return json.dumps({"error": f"No standard amino-acid residues found in chain {chain_id}."})

    sequence = "".join(_aa3[r.get_resname().strip()] for r in residues)
    # Map 1-based sequence position → BioPython residue object
    pos_to_residue: dict[int, object] = {i + 1: r for i, r in enumerate(residues)}

    # --- Compute SASA with FreeSASA ---
    freesasa.setVerbosity(freesasa.silent)
    try:
        sasa_result, _classes = freesasa.calcBioPDB(structure)
        residue_areas = sasa_result.residueAreas()
    except Exception as exc:
        return json.dumps({"error": f"FreeSASA calculation failed: {exc}"})

    def _residue_sasa(res) -> float:
        """Return total SASA for a BioPython Residue."""
        chain_key = res.get_parent().id
        res_id = res.get_id()
        # res_id is (hetflag, seqnum, icode); FreeSASA keys are str(seqnum) + icode.strip()
        res_key = str(res_id[1]) + res_id[2].strip()
        try:
            return residue_areas[chain_key][res_key].total
        except (KeyError, AttributeError):
            return 0.0

    # --- Re-run liability scan on extracted sequence ---
    liabilities_raw = json.loads(scan_structural_liabilities(sequence))
    if "error" in liabilities_raw:
        return json.dumps({"error": f"Liability scan failed: {liabilities_raw['error']}"})

    # --- Classify each liability as exposed or buried ---
    exposed: list[dict] = []
    buried: list[dict] = []

    for liability in liabilities_raw.get("liabilities", []):
        start_pos: int = liability["position"]  # 1-based, start of motif
        motif: str = liability["motif"]
        motif_len = len(motif)

        sasa_values: dict[int, float] = {}
        for offset in range(motif_len):
            seq_pos = start_pos + offset
            res = pos_to_residue.get(seq_pos)
            sasa_values[seq_pos] = round(_residue_sasa(res), 2) if res is not None else 0.0

        max_sasa = max(sasa_values.values()) if sasa_values else 0.0
        is_exposed = max_sasa >= sasa_threshold

        entry = {**liability, "sasa_values": sasa_values, "max_sasa": round(max_sasa, 2)}
        (exposed if is_exposed else buried).append(entry)

    report = {
        "chain_id": chain_id,
        "sequence": sequence,
        "sasa_threshold": sasa_threshold,
        "total_liabilities": len(liabilities_raw.get("liabilities", [])),
        "exposed_liabilities": len(exposed),
        "buried_liabilities": len(buried),
        "exposed": exposed,
        "buried": buried,
    }
    result = json.dumps(report, indent=2)
    logger.info(
        "filter_liabilities_by_sasa | chain=%s exposed=%d buried=%d",
        chain_id,
        len(exposed),
        len(buried),
    )
    return result


@mcp.tool()
def predict_vhh_complex_structure(
    vhh_sequence: str,
    antigen_sequence: str | None = None,
    out_dir: str | None = None,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    dry_run: bool = False,
) -> str:
    """Predict the 3D structure of a VHH or VHH-antigen complex using Boltz-2.

    Boltz-2 is the recommended open-source predictor for antibody-antigen complexes
    (MIT license). Key confidence metrics:
      - iptm >= 0.8: high-confidence binding interface
      - iptm >= 0.6: plausible interface, verify experimentally
      - iptm <  0.6: low confidence, consider redesign

    GPU required for practical inference (min 8 GB VRAM). Use dry_run=True to
    validate inputs and generate the input YAML without running inference - useful
    for queueing jobs on a GPU instance or testing the pipeline locally.

    When antigen_sequence is omitted, defaults to the Human PD-1 ectodomain
    (Uniprot Q15116, residues 33-150) - the primary screening target.

    Args:
        vhh_sequence: Single-letter VHH amino acid sequence.
        antigen_sequence: Optional antigen sequence. Defaults to PD-1 ectodomain.
        out_dir: Output directory for Boltz-2 predictions. Defaults to a temp dir.
        accelerator: "gpu" (default), "cpu" (slow), or "tpu".
        recycling_steps: Boltz-2 recycling iterations (default 3).
        dry_run: If True, validate and write input YAML but skip inference.

    Returns:
        JSON string with fields:
            status: "success" | "dry_run" | "error"
            vhh_length: int
            antigen_length: int | None
            input_yaml: str (path to input file)
            output_dir: str | None
            confidence: dict | None (iptm, ptm, complex_plddt, vhh_antigen_iptm)
            structure_path: str | None (path to best .pdb file)
            error: str | None
    """
    from tools.boltz2_structure import PD1_ECTODOMAIN, predict_structure

    antigen = antigen_sequence if antigen_sequence else PD1_ECTODOMAIN

    result = predict_structure(
        vhh_sequence=vhh_sequence,
        antigen_sequence=antigen,
        out_dir=out_dir,
        accelerator=accelerator,
        recycling_steps=recycling_steps,
        dry_run=dry_run,
    )

    output = json.dumps(result, indent=2)
    logger.info("predict_vhh_complex_structure | status=%s", result.get("status"))
    return output


@mcp.tool()
def score_immunogenicity(sequence: str, device: str = "cpu") -> str:
    """Score a VHH sequence for immunogenicity risk using AbLang2.

    AbLang2 (ablang1-heavy model) is trained on the Observed Antibody Space (OAS)
    database of >2 billion antibody sequences. It scores sequences by pseudo-perplexity:
    the mean negative log-likelihood of each residue when masked. Lower perplexity
    means the sequence looks more like a natural human antibody — lower immunogenicity.

    Pseudo-perplexity is computed by masking each position in turn (Salazar et al. 2020).
    This is more accurate than a single forward pass because each residue is scored
    without access to its own identity.

    Risk thresholds (empirically derived from OAS distribution):
      perplexity < 5   Low risk  — sequence is human-like
      5 <= ppl < 10    Moderate  — humanization of flagged positions recommended
      perplexity >= 10 High risk — framework grafting strongly recommended

    Note: Camelid FR2 hallmarks (F37/E44/R45/G47) will naturally score higher than
    human VH equivalents. This is expected and does not indicate manufacturing risk.

    Args:
        sequence: Single-letter amino acid sequence of the VHH.
        device: Inference device — "cpu", "mps" (Apple Silicon), or "cuda".

    Returns:
        JSON string with:
            sequence_length: int
            perplexity: float
            mean_nll: float
            per_residue_nll: list[float]
            risk_level: "low" | "moderate" | "high"
            interpretation: str
            model: str
            reference: str
    """
    from tools.ablang2_immunogenicity import score_sequence

    seq = _clean_sequence(sequence)
    result = score_sequence(seq, device=device)
    output = json.dumps(result, indent=2)
    logger.info(
        "score_immunogenicity | len=%d | perplexity=%.2f | risk=%s",
        len(seq),
        result.get("perplexity", -1),
        result.get("risk_level", "unknown"),
    )
    return output


@mcp.tool()
def optimize_cdr_sequences(
    pdb_path: str,
    vhh_chain: str = "A",
    cdrs_to_redesign: list[str] | None = None,
    num_samples: int = 10,
    temperature: float = 0.20,
    device: str = "cpu",
) -> str:
    """Redesign VHH CDR sequences conditioned on a 3D scaffold using AntiFold.

    AntiFold is a purpose-built antibody inverse folding model trained on SAbDab
    structures. It outperforms ProteinMPNN on CDR sequence recovery benchmarks
    because it was fine-tuned with antibody-specific CDR masking.

    Workflow:
      1. Run predict_vhh_complex_structure to get a PDB file.
      2. Pass structure_path from that output as pdb_path here.
      3. AntiFold holds the framework backbone fixed and samples new CDR sequences.
      4. Run all 4 developability screens on each candidate before advancing.

    Temperature:
      0.10-0.20  Conservative — stays close to wild-type character (recommended for CDR3)
      0.30-0.50  Diverse — explores sequence space more broadly

    Args:
        pdb_path: Path to PDB file from predict_vhh_complex_structure.
        vhh_chain: Chain ID of the VHH in the PDB. Default "A".
        cdrs_to_redesign: CDRs to redesign. Default ["CDR3"]. Options: "CDR1", "CDR2", "CDR3".
        num_samples: Number of candidate sequences to generate. Default 10.
        temperature: Sampling temperature. Lower = more conservative. Default 0.20.
        device: "cpu", "mps" (Apple Silicon), or "cuda". GPU strongly recommended.

    Returns:
        JSON string with:
            status: "success" | "error" | "antifold_not_installed"
            redesigned_sequences: list of {sequence, log_likelihood, header}
            cdrs_redesigned: list[str]
            pdb_path: str
            error: str | None
    """
    from tools.antifold_inverse_fold import optimize_cdrs

    if cdrs_to_redesign is None:
        cdrs_to_redesign = ["CDR3"]

    result = optimize_cdrs(
        pdb_path=pdb_path,
        vhh_chain=vhh_chain,
        cdrs_to_redesign=cdrs_to_redesign,
        num_samples=num_samples,
        temperature=temperature,
        device=device,
    )
    output = json.dumps(result, indent=2)
    logger.info(
        "optimize_cdr_sequences | pdb=%s | cdrs=%s | status=%s",
        pdb_path,
        cdrs_to_redesign,
        result.get("status"),
    )
    return output


if __name__ == "__main__":
    mcp.run(transport="stdio")
