# Handoff - 2026-05-03 (session 2)

## What was done this session

- Fixed all three blocking biogrill issues:

  1. **PD-1 ectodomain sequence** (`tools/boltz2_structure.py`): The original sequence
     was a chimeric non-PD-1 sequence - the first ~63 residues matched PD-1 but the
     second half was unrelated. Replaced with the verified 118 AA sequence from PDB 4ZQK
     chain B (PD-1/PD-L1 co-crystal), which contains the C93S engineering mutation
     standard in PD-1 expression constructs.

  2. **Caplacizumab fixture** (`tests/conftest.py`): The old sequence cited PDB 1KXV
     (anti-lysozyme VHH - wrong). Updated to the verified sequence from PDB 7EOW chain B
     (caplacizumab/vWF A1 crystal structure). Cascaded changes:
     - `_CAPLACIZUMAB_MAX_PATCH`: 1.357 → 1.686
     - `_CST_MEAN`: 1.431 → 1.456
     - `_CST_STD`: 0.306 → 0.313
     - `_APR_SCREENING_THRESHOLD`: 1.934 → 1.971
     - `camelid_hallmark_count` assertion: 3 → 2 (F37/R45 camelid, G44 human, L47 neither)
     - Dashboard Caplacizumab reference line: 40.5 → 76.9 percentile
     - SCRATCHPAD.md PDB citation corrected (1KXV → 7EOW)

  3. **README tool count**: "five deterministic tools" → "four sequence-level screening
     tools" in the loop description; test count description updated to explicitly distinguish
     sequence-level tools from Boltz-2 structure predictor.

- All 69 tests pass after changes.

## What's next (priority order)

1. **Benchmarks** - run agent_loop N times, report success rate / iterations / cost per
   design. Needed to demonstrate agent performance to Phylo/Biomni audience.
2. **Validation notebook** - run all 13 CST sequences through the pipeline; produce a
   summary table. Now feasible since the calibration is scientifically correct.
3. **5th roadmap tool** - FreeSASA (CPU-only, works on M1 today) for SASA-aware liability
   filtering (only flag surface-exposed PTM motifs, SASA > 25 A^2).
4. **Tool registry pattern** - decouple tool discovery from hardcoded imports in agent_loop.
5. Verify 5DK3 PDB citation in SCRATCHPAD.md - it doesn't contain PD-1; may need to
   update other references that cite it as a PD-1 source.

## Key decisions

- Used PDB 4ZQK chain B for PD-1 ectodomain (not 5DK3 which has no PD-1 chain).
- Used PDB 7EOW chain B for Caplacizumab (crystallographically verified).
- Recalibrated entire APR distribution when Caplacizumab score changed; kept the same
  13-sequence reference set, only corrected the wrong Caplacizumab entry.
- README "five tools" issue resolved by clarification in text, not by integrating Boltz-2
  into agent_loop.py (requires GPU, not suitable for M1 development loop).

## Current blockers

- None. All three biogrill blocking issues resolved.
- Boltz-2 integration into agent_loop.py requires a remote GPU instance (g5.xlarge or
  similar). AWS EC2 quota request may be needed (see SCRATCHPAD.md Block 5).

## Lesson learned

**Never derive biological sequences manually.** Always download from Uniprot/PDB API and
grep. LLMs are unreliable at sequence fidelity - the first attempt at fixing PD1_ECTODOMAIN
derived the corrected sequence from the broken original, preserving the wrong second half.
The correct fix required `curl`-ing the actual FASTA from RCSB and grepping.
