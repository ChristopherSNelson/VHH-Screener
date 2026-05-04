"""
Regression tests for the four deterministic VHH-Screener screening tools.

Each test uses either:
  - Known clinical sequences (Caplacizumab, Pembrolizumab VH) with verified outputs
  - Synthetic sequences constructed so key positions are known exactly
  - Minimal motif-containing strings to test individual liability patterns
"""

import json

import pytest

from biologics_server import (
    _APR_SCREENING_THRESHOLD,
    _CAPLACIZUMAB_MAX_PATCH,
    _CST_MEAN,
    _CST_STD,
    calculate_biophysical_profile,
    scan_aggregation_patches,
    scan_structural_liabilities,
    vhh_hallmark_audit,
)


def parse(result_str: str) -> dict:
    return json.loads(result_str)


# ---------------------------------------------------------------------------
# calculate_biophysical_profile
# ---------------------------------------------------------------------------


class TestCalculateBiophysicalProfile:
    def test_empty_sequence_returns_error(self):
        result = parse(calculate_biophysical_profile(""))
        assert "error" in result

    def test_whitespace_only_returns_error(self):
        result = parse(calculate_biophysical_profile("   \n\t"))
        assert "error" in result

    def test_nonstandard_residues_return_error(self):
        result = parse(calculate_biophysical_profile("EVQLVEXB"))
        assert "error" in result
        assert "Non-standard" in result["error"]

    def test_whitespace_and_digits_stripped(self):
        # Sequence with spaces and digits (e.g. pasted with line numbers) should work
        # "EVQ LVE 123 SGG" → digits and spaces stripped → "EVQLVESGG" = 9 letters
        result = parse(calculate_biophysical_profile("EVQ LVE 123 SGG"))
        assert "error" not in result
        assert result["sequence_length"] == 9

    def test_caplacizumab_passes_all(self, caplacizumab_seq):
        result = parse(calculate_biophysical_profile(caplacizumab_seq))
        assert "error" not in result
        assert result["pI_flag"] == "PASS"
        assert result["gravy_flag"] == "PASS"
        assert result["overall_risk"] == "Low"
        assert result["isoelectric_point"] > 7.5
        assert result["gravy"] <= 0.0

    def test_naive_seed_fails_pi(self, naive_seed):
        result = parse(calculate_biophysical_profile(naive_seed))
        assert "error" not in result
        assert result["pI_flag"] == "FAIL"
        assert result["isoelectric_point"] == pytest.approx(5.18, abs=0.1)

    def test_naive_seed_passes_gravy(self, naive_seed):
        # Naive seed has low pI but is just below 0.0 GRAVY
        result = parse(calculate_biophysical_profile(naive_seed))
        assert result["gravy_flag"] == "PASS"
        assert result["gravy"] <= 0.0

    def test_return_structure_has_required_keys(self, caplacizumab_seq):
        result = parse(calculate_biophysical_profile(caplacizumab_seq))
        for key in (
            "sequence_length",
            "isoelectric_point",
            "gravy",
            "pI_flag",
            "gravy_flag",
            "overall_risk",
            "flags",
        ):
            assert key in result, f"Missing key: {key}"

    def test_flags_populated_on_failure(self, naive_seed):
        result = parse(calculate_biophysical_profile(naive_seed))
        assert len(result["flags"]) >= 1  # At least pI flag

    def test_pembrolizumab_vh_passes_pi(self, pembrolizumab_vh):
        result = parse(calculate_biophysical_profile(pembrolizumab_vh))
        assert result["pI_flag"] == "PASS"


# ---------------------------------------------------------------------------
# scan_structural_liabilities
# ---------------------------------------------------------------------------


class TestScanStructuralLiabilities:
    def test_empty_sequence_returns_error(self):
        result = parse(scan_structural_liabilities(""))
        assert "error" in result

    def test_clean_sequence_passes(self):
        # No NG/NS/NA/DG/NxST motifs
        result = parse(scan_structural_liabilities("EVQLVESGGGLVQPGG"))
        assert result["overall_flag"] == "PASS"
        assert result["liability_count"] == 0

    def test_deamidation_ng(self):
        result = parse(scan_structural_liabilities("AAANGAAA"))
        assert result["liability_count"] >= 1
        motifs = [hit["motif"] for hit in result["liabilities"]]
        assert "NG" in motifs

    def test_deamidation_ns(self):
        result = parse(scan_structural_liabilities("AAANSAAA"))
        motifs = [hit["motif"] for hit in result["liabilities"]]
        assert "NS" in motifs

    def test_deamidation_na(self):
        result = parse(scan_structural_liabilities("AAANAAA"))
        motifs = [hit["motif"] for hit in result["liabilities"]]
        assert "NA" in motifs

    def test_isomerization_dg(self):
        result = parse(scan_structural_liabilities("AAADGAAA"))
        types = [hit["liability_type"] for hit in result["liabilities"]]
        assert "Isomerization" in types

    def test_n_glycosylation_sequon(self):
        # NxS/T where x != P
        result = parse(scan_structural_liabilities("AAANFSTAAA"))
        types = [hit["liability_type"] for hit in result["liabilities"]]
        assert "N-Glycosylation" in types

    def test_n_glycosylation_blocked_by_proline(self):
        # NPST should NOT be flagged — proline blocks glycosylation
        result = parse(scan_structural_liabilities("AAANPSTAAA"))
        glyc = [h for h in result["liabilities"] if h["liability_type"] == "N-Glycosylation"]
        assert len(glyc) == 0

    def test_naive_seed_has_seven_liabilities(self, naive_seed):
        result = parse(scan_structural_liabilities(naive_seed))
        assert result["liability_count"] == 7
        assert result["overall_flag"] == "FAIL"

    def test_positions_are_one_based(self):
        # NG starts at position 4 in 1-based (index 3 in 0-based)
        result = parse(scan_structural_liabilities("AAANGAAA"))
        ng_hit = next(h for h in result["liabilities"] if h["motif"] == "NG")
        assert ng_hit["position"] == 4  # 1-based

    def test_context_field_present_and_formatted(self):
        result = parse(scan_structural_liabilities("AAANGAAA"))
        hit = result["liabilities"][0]
        assert "context" in hit
        # Context should bracket the motif
        assert "[" in hit["context"] and "]" in hit["context"]
        assert hit["motif"] in hit["context"]

    def test_multiple_liabilities_sorted_by_position(self):
        # NG at position 2, DG at position 6
        result = parse(scan_structural_liabilities("ANGAAADG"))
        positions = [h["position"] for h in result["liabilities"]]
        assert positions == sorted(positions)

    def test_return_structure_has_required_keys(self):
        result = parse(scan_structural_liabilities("EVQLVES"))
        for key in ("sequence_length", "liabilities", "liability_count", "overall_flag"):
            assert key in result


# ---------------------------------------------------------------------------
# vhh_hallmark_audit
# ---------------------------------------------------------------------------


class TestVhhHallmarkAudit:
    def test_empty_sequence_returns_error(self):
        result = parse(vhh_hallmark_audit(""))
        assert "error" in result

    def test_fully_camelid_sequence(self, fully_camelid_seq):
        # F37/E44/R45/G47 - constructed sequence with verified positions
        result = parse(vhh_hallmark_audit(fully_camelid_seq))
        assert result["camelid_hallmark_count"] == 4
        assert result["identity"] == "Fully camelid VHH (all 4 hallmarks present)"

    def test_fully_humanized_fr2(self, fully_humanized_fr2_seq):
        # Pembrolizumab VH: V37/G44/L45/W47
        result = parse(vhh_hallmark_audit(fully_humanized_fr2_seq))
        assert result["camelid_hallmark_count"] == 0
        assert result["identity"] == "Fully humanized FR2 (no camelid hallmarks)"

    def test_humanization_warning_for_camelid(self, fully_camelid_seq):
        result = parse(vhh_hallmark_audit(fully_camelid_seq))
        assert "humanization_warning" in result

    def test_no_humanization_warning_for_humanized(self, fully_humanized_fr2_seq):
        result = parse(vhh_hallmark_audit(fully_humanized_fr2_seq))
        assert "humanization_warning" not in result

    def test_short_sequence_returns_error_status(self):
        # Sequence shorter than FR2 start index + max offset (36+10=46)
        result = parse(vhh_hallmark_audit("EVQLVES"))
        # Should have at least one ERROR status in hallmark_audit entries
        statuses = [a.get("status") for a in result["hallmark_audit"]]
        assert "ERROR" in statuses

    def test_audit_has_four_entries(self, fully_camelid_seq):
        result = parse(vhh_hallmark_audit(fully_camelid_seq))
        assert len(result["hallmark_audit"]) == 4

    def test_per_position_fields(self, fully_camelid_seq):
        result = parse(vhh_hallmark_audit(fully_camelid_seq))
        entry = result["hallmark_audit"][0]
        for key in (
            "kabat_position",
            "sequence_index",
            "observed_residue",
            "expected_camelid",
            "expected_human",
            "is_camelid_hallmark",
        ):
            assert key in entry

    def test_suggestion_present_for_camelid_positions(self, fully_camelid_seq):
        result = parse(vhh_hallmark_audit(fully_camelid_seq))
        camelid_entries = [a for a in result["hallmark_audit"] if a.get("is_camelid_hallmark")]
        for entry in camelid_entries:
            assert "suggestion" in entry

    def test_return_structure_has_required_keys(self, fully_camelid_seq):
        result = parse(vhh_hallmark_audit(fully_camelid_seq))
        for key in (
            "sequence_length",
            "framework2_start_index",
            "hallmark_audit",
            "camelid_hallmark_count",
            "identity",
        ):
            assert key in result

    def test_caplacizumab_chimeric(self, caplacizumab_seq):
        # Caplacizumab (7EOW): F37/R45 camelid, G44 human, L47 neither → 2/4
        result = parse(vhh_hallmark_audit(caplacizumab_seq))
        assert result["camelid_hallmark_count"] == 2
        assert "Chimeric" in result["identity"]


# ---------------------------------------------------------------------------
# scan_aggregation_patches
# ---------------------------------------------------------------------------


class TestScanAggregationPatches:
    def test_empty_sequence_returns_error(self):
        result = parse(scan_aggregation_patches(""))
        assert "error" in result

    def test_sequence_too_short_returns_error(self):
        # Fewer than 7 residues (default window)
        result = parse(scan_aggregation_patches("EVQLV"))
        assert "error" in result

    def test_nonstandard_residues_return_error(self):
        result = parse(scan_aggregation_patches("IIIIIIXIIIIIII"))
        assert "error" in result

    def test_caplacizumab_passes_screening(self, caplacizumab_seq):
        result = parse(scan_aggregation_patches(caplacizumab_seq))
        assert result["overall_flag"] == "PASS"
        assert result["candidate_max_patch"]["percentile"] < 95.0

    def test_highly_hydrophobic_fails(self):
        # All isoleucine: KD = 4.5, mean = 4.5 >> threshold 1.934
        result = parse(scan_aggregation_patches("IIIIIIIIIIIIIII"))
        assert result["overall_flag"] == "FAIL"
        assert result["candidate_max_patch"]["z_score"] > 0

    def test_highly_hydrophilic_passes(self):
        # All aspartate: KD = -3.5, mean = -3.5 << threshold
        result = parse(scan_aggregation_patches("DDDDDDDDDDDDDDD"))
        assert result["overall_flag"] == "PASS"
        assert result["candidate_max_patch"]["z_score"] < 0

    def test_naive_seed_fails(self, naive_seed):
        result = parse(scan_aggregation_patches(naive_seed))
        assert result["overall_flag"] == "FAIL"
        assert result["candidate_max_patch"]["percentile"] == pytest.approx(100.0, abs=1.0)

    def test_flagged_patches_structure(self, naive_seed):
        result = parse(scan_aggregation_patches(naive_seed))
        assert result["flagged_patch_count"] > 0
        patch = result["flagged_patches"][0]
        for key in (
            "start_position",
            "end_position",
            "patch_sequence",
            "mean_hydrophobicity",
            "z_score",
            "percentile",
            "suggestion",
        ):
            assert key in patch

    def test_calibration_constants_correct(self):
        # Smoke test: verify the hardcoded calibration constants are sane
        assert _APR_SCREENING_THRESHOLD == pytest.approx(1.971, abs=0.01)
        assert _CAPLACIZUMAB_MAX_PATCH == pytest.approx(1.686, abs=0.001)
        assert _CST_MEAN == pytest.approx(1.456, abs=0.01)
        assert _CST_STD > 0

    def test_z_score_direction(self):
        # A very hydrophilic sequence should have negative z-score (below mean)
        result = parse(scan_aggregation_patches("DDDDDDDDDDDDDDD"))
        assert result["candidate_max_patch"]["z_score"] < 0

    def test_return_structure_has_required_keys(self, caplacizumab_seq):
        result = parse(scan_aggregation_patches(caplacizumab_seq))
        for key in (
            "sequence_length",
            "window_size",
            "calibration",
            "candidate_max_patch",
            "flagged_patches",
            "flagged_patch_count",
            "overall_flag",
            "interpretation",
        ):
            assert key in result

    def test_window_size_configurable(self, caplacizumab_seq):
        result_5 = parse(scan_aggregation_patches(caplacizumab_seq, window_size=5))
        result_7 = parse(scan_aggregation_patches(caplacizumab_seq, window_size=7))
        assert result_5["window_size"] == 5
        assert result_7["window_size"] == 7

    def test_positions_are_one_based(self, naive_seed):
        result = parse(scan_aggregation_patches(naive_seed))
        for patch in result["flagged_patches"]:
            assert patch["start_position"] >= 1
