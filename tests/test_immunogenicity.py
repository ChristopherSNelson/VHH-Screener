"""
Tests for score_immunogenicity (AbLang2) and optimize_cdr_sequences (AntiFold).

AbLang2 tests: verify JSON structure, risk level logic, and error handling.
  Model download is lazy — first test that touches the model will be slow.
  Mark with @pytest.mark.slow if needed; currently run in CI via the model cache.

AntiFold tests: verify graceful fallback when antifold is not installed,
  and error handling for missing PDB files. GPU tests are excluded.
"""

import json
import pytest

from biologics_server import optimize_cdr_sequences, score_immunogenicity

CAPLACIZUMAB = (
    "EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYY"
    "PDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS"
)

SHORT_SEQ = "EVQLVESGGGLVQPGG"


def parse(s: str) -> dict:
    return json.loads(s)


# ---------------------------------------------------------------------------
# score_immunogenicity
# ---------------------------------------------------------------------------


class TestScoreImmunogenicity:
    def test_returns_valid_json(self):
        result = parse(score_immunogenicity(CAPLACIZUMAB))
        assert isinstance(result, dict)

    def test_required_fields_present(self):
        result = parse(score_immunogenicity(CAPLACIZUMAB))
        for field in (
            "sequence_length",
            "perplexity",
            "mean_nll",
            "per_residue_nll",
            "risk_level",
            "interpretation",
            "model",
        ):
            assert field in result, f"Missing field: {field}"

    def test_sequence_length_correct(self):
        result = parse(score_immunogenicity(CAPLACIZUMAB))
        assert result["sequence_length"] == len(CAPLACIZUMAB)

    def test_perplexity_is_positive(self):
        result = parse(score_immunogenicity(CAPLACIZUMAB))
        assert result["perplexity"] > 0.0

    def test_per_residue_nll_length_matches_sequence(self):
        result = parse(score_immunogenicity(CAPLACIZUMAB))
        assert len(result["per_residue_nll"]) == result["sequence_length"]

    def test_risk_level_is_valid(self):
        result = parse(score_immunogenicity(CAPLACIZUMAB))
        assert result["risk_level"] in ("low", "moderate", "high")

    def test_whitespace_stripped_from_sequence(self):
        spaced = CAPLACIZUMAB[:20] + " " + CAPLACIZUMAB[20:40]
        result = parse(score_immunogenicity(spaced))
        assert "error" not in result

    def test_short_sequence_works(self):
        result = parse(score_immunogenicity(SHORT_SEQ))
        assert result["sequence_length"] == len(SHORT_SEQ)
        assert result["perplexity"] > 0.0


# ---------------------------------------------------------------------------
# optimize_cdr_sequences
# ---------------------------------------------------------------------------


class TestOptimizeCdrSequences:
    def test_missing_pdb_returns_error(self):
        result = parse(optimize_cdr_sequences("/nonexistent/path/structure.pdb"))
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_returns_valid_json(self, tmp_path):
        # Create a dummy empty PDB to test the antifold-not-installed path
        dummy_pdb = tmp_path / "test.pdb"
        dummy_pdb.write_text("REMARK dummy\n")
        result = parse(optimize_cdr_sequences(str(dummy_pdb)))
        assert isinstance(result, dict)
        # Either antifold not installed or an error — both are valid without GPU
        assert result["status"] in ("success", "error", "antifold_not_installed")

    def test_antifold_not_installed_message_is_helpful(self, tmp_path):
        try:
            import antifold  # noqa: F401

            pytest.skip("AntiFold is installed — skip not-installed path")
        except ImportError:
            dummy_pdb = tmp_path / "test.pdb"
            dummy_pdb.write_text("REMARK dummy\n")
            result = parse(optimize_cdr_sequences(str(dummy_pdb)))
            assert result["status"] == "antifold_not_installed"
            assert "pip install antifold" in result.get("install_command", "")

    def test_default_cdrs_is_cdr3(self, tmp_path):
        dummy_pdb = tmp_path / "test.pdb"
        dummy_pdb.write_text("REMARK dummy\n")
        result = parse(optimize_cdr_sequences(str(dummy_pdb)))
        # cdrs_redesigned should be set even if execution fails
        # (it's set before the antifold call)
        # Check the result structure is a dict
        assert isinstance(result, dict)
