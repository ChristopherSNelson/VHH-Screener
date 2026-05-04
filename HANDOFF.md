# Handoff - 2026-05-03 (session 3)

## What was done this session

- **PDB 5DK3 citation verified**: 5DK3 is a pembrolizumab Fab crystal structure; chain B
  residues 1-120 exactly match the PEMBROLIZUMAB_VH_SEED in agent_loop.py. Citation is
  correct. Comments updated to "chain B, residues 1-120" in agent_loop.py and conftest.py.
  HANDOFF task closed.

- **Benchmark runner** (`benchmark.py`, new file):
  - Runs agent_loop N times for a given seed / model combination
  - Prints live per-run result (pass/fail, iters, cost, final metrics)
  - Prints summary table: pass rate, mean iters, stddev, mean cost/run, cost/passing design
  - Cross-seed comparison when multiple seeds specified
  - Saves JSON to `logs/benchmark_<seed>_<model>_<timestamp>.json`
  - CLI: `--seed {naive,pembrolizumab,none,all}`, `--n N`, `--model MODEL`

- **RunResult dataclass** added to `agent_loop.py`:
  - `run_screening_loop` now returns `RunResult` instead of `None`
  - Objective pass/fail: all 4 tools run on final sequence after loop ends (not agent self-report)
  - `suppress_plot=True` for headless benchmark runs
  - `seed_label` param for RunResult labelling
  - `__main__` exits with code 0/1 (CI-friendly)

- **Tool registry** (`agent_loop.py`):
  - Replaced separate `TOOLS` list + `TOOL_DISPATCH` dict with a single `TOOL_REGISTRY`
  - `TOOLS` and `TOOL_DISPATCH` auto-derived; adding a new tool = one registry entry

- **FreeSASA SASA-aware liability filter** (`biologics_server.py`, 5th MCP tool):
  - `filter_liabilities_by_sasa(pdb_path, sasa_threshold=25.0, chain_id=None)`
  - Loads PDB with BioPython, computes per-residue SASA (Lee-Richards, FreeSASA)
  - Re-runs `scan_structural_liabilities` on extracted chain sequence
  - Splits liabilities into `exposed` (SASA >= threshold) and `buried`; each entry gets
    `sasa_values` dict and `max_sasa`
  - CPU-only; works on M1 today. Designed for use after Boltz-2 structure prediction.
  - freesasa installed via `conda install -c conda-forge freesasa` (pip wheel broken on ARM)
  - 14 new tests in `tests/test_sasa_filter.py` using PDB 7EOW downloaded as session fixture;
    skipped automatically if network unavailable

- **README updated**: new SASA tool section, benchmark usage, architecture diagram updated,
  test count 69 → 83, roadmap SASA item removed (shipped), items renumbered.

- **83 tests**, all passing.

## What's next (priority order)

1. **Run benchmarks** - now that `benchmark.py` exists, run `--seed all --n 5` and commit
   the JSON output + a summary table to README or docs/. This is the key demo artefact for
   the Phylo/Biomni audience.
2. **Validation notebook** - run all 13 CST sequences through the pipeline; produce a
   summary table showing each passes the APR threshold. CST sequences must be sourced from
   PDB/patents (do NOT hallucinate). Feasible now that calibration is correct.
3. **Inverse folding** - AntiFold for CDR sequence optimization conditioned on Boltz-2
   structure. Requires GPU (g5.xlarge).
4. **Immunogenicity** - AbLang2/AntiBERTa2 OAS-perplexity scoring; BigMHC for MHC
   presentation. CPU-feasible for AbLang2.
5. **Search strategy** - MCTS-based mutation exploration; multi-agent adversarial debate.

## Key decisions

- Benchmark pass/fail is determined objectively (all 4 tools on final sequence), not from
  the agent's self-assessment ("I am satisfied with this design"). This is more rigorous and
  reproducible across models.
- FreeSASA installed via conda-forge, not pip - pip wheel fails to build on ARM/M1. Noted
  in pyproject.toml as a dependency but install must be done with conda.
- SASA filter is NOT added to agent_loop.py TOOL_REGISTRY - it requires a PDB file that
  doesn't exist in the sequence-only loop. It's an MCP tool for interactive/post-structure use.
- freesasa.calcBioPDB returns 2 values (result, classes) in the conda-forge version, not 3
  as the docs imply. residue_areas obtained via result.residueAreas().

## Current blockers

- GPU required for Boltz-2 inference and AntiFold. AWS EC2 quota increase for g5.xlarge
  may be needed (see SCRATCHPAD.md Block 5). Lambda Labs / RunPod are alternatives.
- Validation notebook blocked on sourcing all 13 CST sequences; do not hallucinate them.
