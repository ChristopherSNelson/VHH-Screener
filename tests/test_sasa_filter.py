"""
Tests for filter_liabilities_by_sasa (biologics_server.py).

Uses PDB 7EOW (Caplacizumab / vWF A1 complex) downloaded from RCSB.
Chain B is the caplacizumab VHH; it contains known PTM motifs (NA at
the framework-CDR3 junction) that should be assessed for solvent exposure.

Tests are skipped if the PDB download fails (network unavailable).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from biologics_server import filter_liabilities_by_sasa


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pdb_7eow(tmp_path_factory) -> Path:
    """Download PDB 7EOW and return local path. Skip if network unavailable."""
    tmp = tmp_path_factory.mktemp("pdb")
    dest = tmp / "7eow.pdb"
    url = "https://files.rcsb.org/download/7EOW.pdb"
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        pytest.skip(f"Could not download 7EOW from RCSB: {exc}")
    return dest


@pytest.fixture(scope="session")
def sasa_result_chain_b(pdb_7eow) -> dict:
    """Run filter_liabilities_by_sasa on 7EOW chain B (caplacizumab VHH)."""
    raw = filter_liabilities_by_sasa(str(pdb_7eow), sasa_threshold=25.0, chain_id="B")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Structural / schema tests (no network needed beyond fixture)
# ---------------------------------------------------------------------------


def test_missing_pdb_returns_error():
    result = json.loads(filter_liabilities_by_sasa("/nonexistent/path/fake.pdb"))
    assert "error" in result


def test_invalid_chain_returns_error(pdb_7eow):
    result = json.loads(filter_liabilities_by_sasa(str(pdb_7eow), chain_id="Z"))
    assert "error" in result
    assert "available" in result["error"].lower() or "not found" in result["error"].lower()


def test_result_schema(sasa_result_chain_b):
    r = sasa_result_chain_b
    assert "error" not in r, f"Unexpected error: {r.get('error')}"
    for key in (
        "chain_id",
        "sequence",
        "sasa_threshold",
        "total_liabilities",
        "exposed_liabilities",
        "buried_liabilities",
        "exposed",
        "buried",
    ):
        assert key in r, f"Missing key: {key}"


def test_chain_id_reported(sasa_result_chain_b):
    assert sasa_result_chain_b["chain_id"] == "B"


def test_threshold_reported(sasa_result_chain_b):
    assert sasa_result_chain_b["sasa_threshold"] == 25.0


def test_sequence_is_protein(sasa_result_chain_b):
    seq = sasa_result_chain_b["sequence"]
    assert len(seq) > 50, "Caplacizumab VHH should be >50 residues"
    assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq), "Non-standard residues in sequence"


def test_liability_counts_consistent(sasa_result_chain_b):
    r = sasa_result_chain_b
    assert r["exposed_liabilities"] + r["buried_liabilities"] == r["total_liabilities"]
    assert len(r["exposed"]) == r["exposed_liabilities"]
    assert len(r["buried"]) == r["buried_liabilities"]


def test_exposed_entry_has_sasa_values(sasa_result_chain_b):
    all_liabilities = sasa_result_chain_b["exposed"] + sasa_result_chain_b["buried"]
    if not all_liabilities:
        pytest.skip("No liabilities found in 7EOW chain B — cannot test sasa_values")
    for entry in all_liabilities:
        assert "sasa_values" in entry, "Each liability must have sasa_values"
        assert "max_sasa" in entry, "Each liability must have max_sasa"
        assert isinstance(entry["sasa_values"], dict)
        assert entry["max_sasa"] >= 0.0


def test_exposed_entries_meet_threshold(sasa_result_chain_b):
    threshold = sasa_result_chain_b["sasa_threshold"]
    for entry in sasa_result_chain_b["exposed"]:
        assert entry["max_sasa"] >= threshold, (
            f"Exposed liability has max_sasa={entry['max_sasa']} < threshold={threshold}"
        )


def test_buried_entries_below_threshold(sasa_result_chain_b):
    threshold = sasa_result_chain_b["sasa_threshold"]
    for entry in sasa_result_chain_b["buried"]:
        assert entry["max_sasa"] < threshold, (
            f"Buried liability has max_sasa={entry['max_sasa']} >= threshold={threshold}"
        )


def test_exposed_liability_inherits_scan_fields(sasa_result_chain_b):
    all_liabilities = sasa_result_chain_b["exposed"] + sasa_result_chain_b["buried"]
    if not all_liabilities:
        pytest.skip("No liabilities in 7EOW chain B")
    for entry in all_liabilities:
        for field in ("liability_type", "motif", "position", "context"):
            assert field in entry, f"Missing scan_structural_liabilities field: {field}"


def test_default_chain_uses_first(pdb_7eow):
    """Omitting chain_id should succeed and return a valid result."""
    result = json.loads(filter_liabilities_by_sasa(str(pdb_7eow)))
    assert "error" not in result
    assert "chain_id" in result


def test_high_threshold_buries_all(pdb_7eow):
    """A threshold of 9999 Å² should bury all liabilities."""
    result = json.loads(
        filter_liabilities_by_sasa(str(pdb_7eow), sasa_threshold=9999.0, chain_id="B")
    )
    assert result["exposed_liabilities"] == 0
    assert result["buried_liabilities"] == result["total_liabilities"]


def test_zero_threshold_exposes_all(pdb_7eow):
    """A threshold of 0 Å² should expose all liabilities."""
    result = json.loads(
        filter_liabilities_by_sasa(str(pdb_7eow), sasa_threshold=0.0, chain_id="B")
    )
    assert result["buried_liabilities"] == 0
    assert result["exposed_liabilities"] == result["total_liabilities"]
