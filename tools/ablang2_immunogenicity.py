"""
ablang2_immunogenicity.py - AbLang pseudo-perplexity immunogenicity scoring.

Scores a VHH sequence for immunogenicity risk using AbLang (heavy chain model),
an antibody-specific language model trained on the Observed Antibody Space (OAS).

Pseudo-perplexity: mask each position in turn, record NLL of true residue,
exponentiate the mean. Lower = more human-like = lower immunogenicity risk.

References:
  Olsen et al., AbLang: an antibody language model for completing antibody sequences.
  Bioinformatics Advances, 2022. https://github.com/oxpig/AbLang
  Salazar et al., Masked Language Model Scoring. ACL, 2020.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger("vhh-screener")

_VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
_MODEL_CACHE: dict[str, Any] = {}


def _get_model(device: str = "cpu") -> Any:
    """Load AbLang heavy chain model, caching after first load (~100 MB download)."""
    import ablang

    key = f"ablang-heavy-{device}"
    if key not in _MODEL_CACHE:
        logger.info("Loading AbLang heavy model (first call - may download ~100 MB)")
        model = ablang.pretrained("heavy", device=device)
        model.freeze()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _build_vocab_map(model: Any) -> dict[str, int]:
    """Build AA → logit-index map from the model's tokenizer vocab."""
    # AbLang vocab_to_token maps AA → token_idx (1-indexed for AAs).
    # The logit tensor has 20 dims for the 20 AAs.
    # Empirically determine the logit index for each AA by checking which
    # logit dimension the tokenizer token maps to via the embedding matrix.
    vocab_to_token: dict[str, int] = model.tokenizer.vocab_to_token
    # Filter to standard 20 AAs only (skip special tokens: <, >, -, *)
    aa_tokens = {aa: idx for aa, idx in vocab_to_token.items() if aa in _VALID_AAS}
    # The model outputs 20 logits corresponding to AA token indices sorted ascending.
    # Sort by token index to get the logit position mapping.
    sorted_aas = sorted(aa_tokens.items(), key=lambda x: x[1])
    return {aa: logit_idx for logit_idx, (aa, _) in enumerate(sorted_aas)}


def score_sequence(
    sequence: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Score a VHH sequence for immunogenicity using AbLang pseudo-perplexity.

    Each position is masked in turn; the NLL of the true residue is recorded.
    Pseudo-perplexity = exp(mean NLL). This is the standard BERT-PLM approach.

    Args:
        sequence: Single-letter amino acid sequence (cleaned, uppercase).
        device: "cpu", "mps" (Apple Silicon), or "cuda".

    Returns:
        dict with perplexity, per_residue_nll, risk_level, interpretation.
    """
    seq = re.sub(r"[^A-Za-z]", "", sequence).upper()
    invalid = set(seq) - _VALID_AAS
    if invalid:
        raise ValueError(f"Non-standard amino acids: {invalid}")

    model = _get_model(device)
    aa_to_logit_idx = _build_vocab_map(model)

    per_residue_nll: list[float] = []

    for i, aa in enumerate(seq):
        # Mask position i with '*' (AbLang mask token)
        masked_seq = seq[:i] + "*" + seq[i + 1 :]
        # likelihood mode: returns logits of shape (1, L+2, 20)
        # positions: 0=<start>, 1..L=residues, L+1=<end>
        logits = model([masked_seq], mode="likelihood")
        pos_logits = logits[0, i + 1, :]  # shape (20,)

        # Stable log-softmax
        shifted = pos_logits - pos_logits.max()
        log_probs = shifted - np.log(np.sum(np.exp(shifted)))

        logit_idx = aa_to_logit_idx.get(aa)
        if logit_idx is None:
            per_residue_nll.append(0.0)
            continue
        nll = float(-log_probs[logit_idx])
        per_residue_nll.append(nll)

    mean_nll = float(np.mean(per_residue_nll))
    perplexity = float(np.exp(mean_nll))

    if perplexity < 5.0:
        risk_level = "low"
        interpretation = (
            "Sequence is human-like. Low immunogenicity risk for therapeutic development."
        )
    elif perplexity < 10.0:
        risk_level = "moderate"
        interpretation = (
            "Sequence diverges somewhat from human antibody repertoire. "
            "Consider humanization of high-NLL positions."
        )
    else:
        risk_level = "high"
        interpretation = (
            "Sequence is significantly non-human. High immunogenicity risk. "
            "Humanization or framework grafting recommended."
        )

    logger.info(
        "score_immunogenicity | len=%d | perplexity=%.2f | risk=%s",
        len(seq),
        perplexity,
        risk_level,
    )

    return {
        "sequence_length": len(seq),
        "perplexity": round(perplexity, 3),
        "mean_nll": round(mean_nll, 4),
        "per_residue_nll": [round(x, 4) for x in per_residue_nll],
        "risk_level": risk_level,
        "interpretation": interpretation,
        "model": "AbLang heavy (ablang v1)",
        "reference": "Olsen et al. 2022, Bioinformatics Advances",
        "note": (
            "Camelid FR2 hallmarks (F37/E44/R45/G47) will score higher than "
            "human VH — expected and does not indicate manufacturing risk."
        ),
    }
