# SCRATCHPAD - 2-Hour Sprint Plan

## Goal
Ship tests, packaging, validation, and example log. All local, no API keys, no GPU.

## Context for Sonnet
- Repo: `~/projects/VHH-Screener`
- Two main files: `biologics_server.py` (561 lines, 4 MCP tools), `agent_loop.py` (843 lines, screening loop)
- No tests, no pyproject.toml, no notebooks yet
- Existing logs: `logs/agent_cot.log` (4076 lines from a real run)
- The 4 tools are pure functions that accept a sequence string and return JSON strings
- CST calibration data is hardcoded in `biologics_server.py` (lines 384-398)

---

## Block 1: Tests (45 min) - `tests/test_biologics_server.py`

### Setup
- Create `tests/` directory with `__init__.py` and `conftest.py`
- Import the 4 tool functions directly from `biologics_server.py`
- No MCP subprocess needed - these are plain Python functions that return JSON strings

### Test sequences to use

**Caplacizumab VHH** (PDB: 7EOW chain B, first approved VHH):
```
QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYNMGWFRQAPGKEREFVSAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS
```
Expected: PASS on APR (40.5th percentile), PASS on GRAVY (hydrophilic), known CST gold standard.

**Naive seed** (from agent_loop.py line 293-296):
```
EVQLVESGGGLVQPGGSLRLSCAASGFTFSNGYMSNGWVRQAPGKGLEWVSDGISNGGSTYYAD
SVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAAILVCFFDGYWGQGTLVTVSS
```
Expected: FAIL on liabilities (has NG, NS motifs), high APR, low pI - the deliberately bad sequence.

**Pembrolizumab VH** (from agent_loop.py line 300-303):
```
QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKF
KNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS
```
Expected: Has NG liability, good pI.

### Test cases per tool

#### `test_calculate_biophysical_profile`
1. `test_empty_sequence` - returns error JSON
2. `test_invalid_residues` - sequence with "X" or "B" returns error
3. `test_caplacizumab_passes` - pI > 7.5, GRAVY <= 0.0, overall_risk = "Low"
4. `test_naive_seed_fails_pi` - pI < 7.5 (the naive seed is engineered to fail)
5. `test_return_structure` - verify all expected JSON keys are present
6. `test_whitespace_stripping` - "EVQ LVE" should work like "EVQLVE"

#### `test_scan_structural_liabilities`
1. `test_clean_sequence` - "EVQLVESGGGLVQ" (no NG/NS/NA/DG/NxST) returns PASS
2. `test_deamidation_ng` - "AANGBB" finds NG at correct position
3. `test_deamidation_ns` - "AANSBB" finds NS
4. `test_deamidation_na` - "AANABB" finds NA
5. `test_isomerization_dg` - "AADGBB" finds DG
6. `test_glycosylation_nxst` - "AANFTBB" finds N-glycosylation sequon
7. `test_glycosylation_npst_excluded` - "AANPSTBB" does NOT flag (P blocks glycosylation)
8. `test_multiple_liabilities` - naive seed returns liability_count >= 3
9. `test_context_field` - each liability has a "context" field with brackets around motif
10. `test_1_based_positions` - positions are 1-based, not 0-based

#### `test_vhh_hallmark_audit`
1. `test_fully_camelid` - sequence with F37/E44/R45/G47 returns camelid_hallmark_count = 4
2. `test_fully_humanized` - sequence with V37/G44/L45/W47 returns camelid_hallmark_count = 0
3. `test_short_sequence` - returns ERROR status for sequences too short
4. `test_identity_string` - "Fully camelid VHH" or "Fully humanized FR2" in identity field
5. `test_humanization_warning_present` - camelid sequences get warning about solubility

#### `test_scan_aggregation_patches`
1. `test_caplacizumab_gold_standard` - max patch percentile ~ 40.5 (within 2.0 tolerance)
2. `test_highly_hydrophobic_fails` - "IIIIIIIIIIIIIII" (all isoleucine) should FAIL
3. `test_highly_hydrophilic_passes` - "DDDDDDDDDDDDDDD" (all aspartate) should PASS
4. `test_short_sequence_error` - fewer than 7 residues returns error
5. `test_z_score_sign` - Caplacizumab z-score should be negative (below mean)
6. `test_flagged_patches_structure` - each flagged patch has start_position, end_position, patch_sequence, suggestion
7. `test_screening_threshold` - verify _APR_SCREENING_THRESHOLD ~ 1.934

### conftest.py fixtures
```python
@pytest.fixture
def caplacizumab_seq():
    return "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYNMGWFRQAPGKEREFVSAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS"

@pytest.fixture
def naive_seed():
    return "EVQLVESGGGLVQPGGSLRLSCAASGFTFSNGYMSNGWVRQAPGKGLEWVSDGISNGGSTYYAD SVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAAILVCFFDGYWGQGTLVTVSS"

@pytest.fixture
def pembrolizumab_vh():
    return "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"

@pytest.fixture
def clean_short_seq():
    return "EVQLVESGGGLVQPGGSLRL"  # No liabilities, long enough for APR
```

---

## Block 2: pyproject.toml (15 min)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vhh-screener"
version = "0.1.0"
description = "Agentic developability screening for VHH nanobody engineering"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
dependencies = [
    "biopython>=1.83",
    "fastmcp>=0.1.0",
    "openai>=1.0.0",
    "matplotlib>=3.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4.0",
    "mypy>=1.10",
]

[project.scripts]
vhh-screen = "agent_loop:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
target-version = "py311"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

Also create `.gitignore` additions if needed (logs/*.log already gitignored? check).

---

## Block 3: Validation notebook (30 min) - `notebooks/cst_validation.ipynb`

### Purpose
Run all 13 CST sequences through the 4 screening tools. Produce a summary table showing each passes the APR threshold. This validates the calibration set.

### CST sequences needed
The 13 sequences must be defined. They are NOT currently in the codebase - only their max-patch scores are (biologics_server.py lines 384-398). Sonnet needs to look up or reconstruct these from public sources.

**Critical**: The actual amino acid sequences for the 13 CSTs must be sourced. Approach:
- Hard-code them in a `data/cst_sequences.py` or `data/cst_sequences.json` file
- Sources: PDB (7EOW for Caplacizumab, 5DK3 for Pembrolizumab, etc.)
- Only the VH/VHH domain, not full heavy chain

### Notebook structure
1. Import tools from biologics_server.py
2. Load CST sequences
3. Run each through all 4 tools
4. Build a pandas DataFrame summary table
5. Assert all 13 are within the 95th percentile (they should be - that's the calibration)
6. Show the distribution plot that matches the hardcoded stats

### Known issue
The CST max-patch scores in biologics_server.py were computed from the actual sequences. If Sonnet uses slightly different sequence boundaries, the numbers may not match exactly. Tolerance of +/- 0.05 on max patch scores is acceptable.

**Sonnet instruction**: If you cannot find the exact PDB sequences, skip the notebook and note what's missing. Do NOT hallucinate sequences. The Pembrolizumab VH and naive seed from agent_loop.py are known-good and should be used for whatever validation you can do.

---

## Block 4: Example run log (15 min)

- The existing `logs/agent_cot.log` (4076 lines) is from a real run
- Sanitize it: strip any API keys or sensitive info (there shouldn't be any - the key is only in env vars)
- Copy to `examples/example_run.log`
- Add a brief `examples/README.md` explaining what the log shows

**Sonnet instruction**: Read the first and last 50 lines of `logs/agent_cot.log` to verify it's a complete, successful run. If it shows a passing design at the end, it's good to commit. If it's a partial/failed run, note that in the examples README.

---

## Block 5: Tool registry + cleanup (15 min)

### Tool registry in agent_loop.py
Currently tools are hardcoded in two places:
1. `TOOLS` list (lines 105-202) - OpenAI function-calling schema
2. `TOOL_DISPATCH` dict (lines 207-220) - maps name to callable

Refactor to a single registry pattern:
```python
TOOL_REGISTRY = {
    "scan_structural_liabilities": {
        "fn": scan_structural_liabilities,
        "params": {"sequence": {"type": "string", "description": "..."}},
        "description": "...",
    },
    # ...
}

# Auto-generate TOOLS list and TOOL_DISPATCH from TOOL_REGISTRY
```

This is a nice-to-have. If time is short, skip it. The tests and pyproject.toml are higher priority.

---

## AWS Resources (for future session, not this sprint)

### What to request now
1. **Service**: EC2
2. **Region**: us-west-2 (best GPU availability)
3. **Quota increase**: "Running On-Demand G and VT instances" from 0 to 4 vCPUs
4. **Instance type**: `g5.xlarge` (1x NVIDIA A10G, 24 GB VRAM, 4 vCPU, 16 GB RAM)
5. **Cost**: ~$1.006/hr on-demand, ~$0.35/hr spot
6. **AMI**: Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04) - `ami-0c76f473` or similar

### What this unlocks
- **Boltz-2**: VHH-antigen complex structure prediction (~2 GB model, needs GPU for inference in <5 min)
- **AntiFold**: CDR inverse folding (ESM-based, needs GPU)
- **AbLang2**: Humanness scoring via OAS-perplexity (GPU preferred but CPU possible)
- **FreeSASA**: CPU-only, no GPU needed - can run locally on M1 today

### How to request
```bash
# AWS Console path:
# Service Quotas > Amazon EC2 > Running On-Demand G and VT instances
# Request increase to 4 (enough for 1x g5.xlarge)

# Or via CLI:
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-DB2F2B96 \
  --desired-value 4 \
  --region us-west-2
```

Typical approval time: 15 min to 48 hours depending on account age and region.

### Alternative: Lambda Labs or RunPod
If AWS quota is slow, Lambda Labs ($0.80/hr for A10G) or RunPod ($0.44/hr for A10G spot) are viable alternatives with no quota wait. Both offer PyTorch pre-installed.

---

## Execution order for Sonnet 4.6

1. **pyproject.toml** first (15 min) - enables `pytest` config and `pythonpath`
2. **tests** (45 min) - the main deliverable
3. **Run tests, fix failures** (built into test block)
4. **Example log** (15 min) - quick win
5. **Validation notebook** (30 min) - if CST sequences are available
6. **Tool registry** (15 min) - only if time remains

## Verification
After each block, run:
```bash
python -m pytest tests/ -x -q
ruff check . --fix
ruff format .
```

## Definition of done
- `pytest` passes with 20+ tests
- `pyproject.toml` exists with correct deps
- `examples/example_run.log` committed
- All files pass `ruff check` and `ruff format`
