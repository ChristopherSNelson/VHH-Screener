"""
Shared fixtures for VHH-Screener test suite.

Sequences:
- Caplacizumab: first approved VHH therapeutic (anti-vWF)
- Pembrolizumab VH: from agent_loop.py PEMBROLIZUMAB_VH_SEED (PDB 5DK3)
- Naive seed: from agent_loop.py NAIVE_SEED (deliberately bad, 7 liabilities)
- Synthetic camelid/humanized: constructed to have known FR2 hallmark residues
"""

import json

import pytest


@pytest.fixture
def caplacizumab_seq() -> str:
    # Caplacizumab VHH domain - first approved nanobody therapeutic (anti-vWF)
    # Source: PDB 7EOW chain B (crystal structure with vWF A1 domain), His-tag removed
    # biophysics: pI=9.07 PASS, GRAVY=-0.349 PASS
    # APR: max_patch=1.686, ~77th percentile, PASS
    # FR2 tetrad: F37/G44/R45/L47 → 2/4 camelid hallmarks (Chimeric)
    return (
        "EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYY"
        "PDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS"
    )


@pytest.fixture
def naive_seed() -> str:
    # Deliberately bad VHH - 7 liabilities, pI=5.18 FAIL, APR 100th percentile FAIL
    # Includes an intentional space to test whitespace stripping
    return (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSNGYMSNGWVRQAPGKGLEWVSDGISNGGS"
        "TYYAD SVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAAILVCFFDGYWGQGTLVTVSS"
    )


@pytest.fixture
def pembrolizumab_vh() -> str:
    # Pembrolizumab VH - fully humanized FR2 (V37/G44/L45/W47)
    # Has 1 known liability: NG at position ~5
    return (
        "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTN"
        "FNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
    )


@pytest.fixture
def fully_camelid_seq() -> str:
    # Synthetic sequence with confirmed camelid FR2 tetrad
    # index 36=F (Kabat 37), 43=E (Kabat 44), 44=R (Kabat 45), 46=G (Kabat 47)
    return "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWFQAPGKGEREG"


@pytest.fixture
def fully_humanized_fr2_seq() -> str:
    # Pembrolizumab VH: confirmed human FR2 (V37/G44/L45/W47)
    # index 36=V, 43=G, 44=L, 46=W
    return (
        "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTN"
        "FNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
    )


def parse_result(result_str: str) -> dict:
    """Helper to parse tool JSON output."""
    return json.loads(result_str)
