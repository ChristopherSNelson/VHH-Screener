# CLAUDE.md — Bioinformatics & ML/AI for Biopharma

> Adapted from Boris Cherny's Claude Code setup for a computational biology workflow
> on an M1 Pro MacBook (16 GB RAM).
>
> **For Claude**: Sections marked *[AGENT RULES]* are directives you must follow.
> **For the human**: Sections marked *[OPERATOR NOTES]* are reminders for the user's workflow.
> **Shared**: Unmarked sections apply to both.
>
> **Self-updating rule** *[AGENT RULES]*: After any correction or mistake, propose a specific
> addition to the Mistakes log at the bottom of this file. The user will end corrections with:
> "Now update CLAUDE.md so you don't make that mistake again."

---

## Environment constraints *[AGENT RULES]*

- **Machine**: Apple M1 Pro, 16 GB unified memory, macOS. All resource recommendations must respect this.
- **Parallelism budget**: Do not suggest more than 2 concurrent sessions or parallel subagents. Prefer sequential work with clean handoffs.
- **ARM/Apple Silicon awareness**: Default to `osx-arm64` / `linux-aarch64` for conda, Docker `--platform linux/arm64`, and `tensorflow-macos` / `tensorflow-metal` when TF is needed. Before recommending any pip/conda package, verify it has an ARM-native wheel. Flag Rosetta fallbacks explicitly.
- **Memory discipline**: Before recommending a tool or approach, estimate peak memory. Prefer streaming/chunked I/O (e.g., `pysam`, `dask`, `polars`) over full in-memory loads. If a dataset exceeds ~6 GB on disk, propose a chunked or out-of-core strategy before writing code.
- **Docker**: Assume `colima` or Docker Desktop for Mac. Pin `--platform linux/arm64` in Dockerfiles. Keep images slim — no CUDA layers unless the user specifies a remote GPU target.

---

## `.claude/` directory manifest *[AGENT RULES]*

This project includes a `.claude/` configuration directory. Be aware of its contents:

### Slash commands (`.claude/commands/`) — user-invoked only

These are triggered by the user typing `/command-name` in the terminal. You cannot invoke them yourself, but you should understand what they do so you can suggest the right one when relevant.

| Command | What it does |
|---|---|
| `/commit-push-pr` | Stages, commits, pushes, and opens a PR in one shot. Uses inline bash to pre-compute git status. |
| `/quick-commit` | Stage all changes and commit with a descriptive message. |
| `/test-and-fix` | Runs the test suite; if failures occur, attempts to fix them. |
| `/review-changes` | Reviews uncommitted changes and suggests improvements. |
| `/grill` | Adversarial code review — stress-tests the implementation. |
| `/techdebt` | Scans for and proposes cleanup of technical debt. |
| `/worktree` | Sets up git worktrees for parallel Claude sessions. |

When suggesting a workflow to the user, reference these by name (e.g., "you could run `/test-and-fix` now"). Do not attempt to execute them as bash commands.

### Subagents (`.claude/agents/`) — you can delegate to these

You can spawn these as subagents when appropriate. Limit to 1 concurrent subagent on this machine.

| Agent | When to use it |
|---|---|
| `code-simplifier` | After implementation is complete and tests pass. Simplifies and cleans up the code. |
| `code-architect` | For design reviews and architectural decisions before implementation. |
| `verify-app` | Thorough end-to-end application testing. Use after major changes. |
| `build-validator` | Ensures the project builds correctly for deployment. |
| `oncall-guide` | Diagnosing and resolving production issues. |

### Settings (`.claude/settings.json`)

Pre-allowed permissions for common safe commands so they don't prompt for approval each time. Customize this for your stack — typical allows include: `git`, `conda`, `mamba`, `pip`, `python`, `pytest`, `ruff`, `mypy`, `nextflow`, `snakemake`, `docker`, `colima`, `samtools`, `bcftools`, `bedtools`, `gh` (GitHub CLI). Also includes a PostToolUse hook that auto-formats code after Write/Edit operations.

### Skills (`.claude/skills/`)

| Skill | What it does |
|---|---|
| `mcp-maker` | Scaffolds a new FastMCP server from a spec, registers it with Claude Code, and runs a smoke test. Use when a task needs structured access to an external API. See `.claude/skills/mcp-maker/SKILL.md`. |

Add more skills over time for recurring workflows (e.g., `nf-pipeline-scaffold`, `eda-anndata`, `deseq2-from-counts`).

### MCP servers (`.claude/mcp-servers/`)

Project-local MCP servers created by the `mcp-maker` skill live here. Each is a self-contained Python file using FastMCP. Disposable servers can be deleted after a project; reusable ones should be committed to git.

### Adapting for new projects

When setting up a new project, consider which commands and agents to keep, remove, or add. Domain-specific additions to consider:
- `/validate-pipeline` — dry-run the Nextflow/Snakemake pipeline and report errors.
- `/check-data-schema` — validate input data against expected schema.
- A `pipeline-reviewer` agent — reviews workflow logic, resource allocations, and container configs.

---

## Domain rules — Bioinformatics *[AGENT RULES]*

- **File formats**: Know the difference between BAM/CRAM, VCF/BCF, BED, GFF3/GTF, FASTQ, AnnData (.h5ad), and when to use each. Don't confuse 0-based (BED) with 1-based (VCF, GFF) coordinates — ever.
- **Reference genomes**: Always confirm which build (hg38/GRCh38 vs. hg19/GRCh37 vs. T2T-CHM13) before writing any genomic pipeline code. Never assume.
- **Bioinformatics stack preferences**:
  - Alignment: `minimap2`, `STAR` (RNA-seq), `bwa-mem2`
  - Variant calling: `DeepVariant`, `GATK HaplotypeCaller`, `bcftools`
  - Differential expression: `DESeq2` (R), `pydeseq2` (Python)
  - Single-cell: `scanpy` (Python), `Seurat` (R)
  - Methylation: `bismark`, `methyldackel`, custom QC (cf. ImmuneMethylTools)
  - Workflow management: `Nextflow` (preferred) or `Snakemake`
- **Statistical rigor**: Always apply multiple-testing correction (Benjamini-Hochberg unless stated otherwise). Report effect sizes alongside p-values. Flag when sample sizes are too small for the proposed analysis.
- **Reproducibility**: Pin every tool version. Use conda environment YAML or Docker for dependency management. Set random seeds. Record CLI invocations in a run log.

---

## Domain rules — ML/AI for Life Sciences *[AGENT RULES]*

- **Framework preferences**: PyTorch first (better ARM/MPS support), scikit-learn for classical ML, XGBoost/LightGBM for tabular. Only suggest TensorFlow if there's a compelling pretrained model reason.
- **Apple MPS acceleration**: Use `device = torch.device("mps")` where supported. Test for MPS compatibility — not all ops are implemented. Fall back gracefully to CPU with a warning.
- **Model size awareness**: With 16 GB unified memory, we can comfortably fine-tune models up to ~1–2B parameters at reduced precision. For larger models, recommend quantized inference (GGUF/llama.cpp/MLX) or suggest offloading to a cloud GPU. Don't propose loading a 7B float16 model locally without discussing this.
- **Data leakage**: In any train/val/test split involving biological data, ensure no patient/sample leakage across splits. Group splits by patient ID, cell line, or batch — not random row sampling — unless independence is justified.
- **Molecular ML**: For protein/small-molecule tasks, prefer ESM-2 embeddings (protein), ChemBERTa or molecular fingerprints (small molecules). Mention AlphaFold/OpenFold for structure prediction but don't reinvent them.
- **Evaluation**: Use domain-appropriate metrics. AUROC and AUPRC for imbalanced binary classification (common in drug response). Pearson/Spearman + RMSE for continuous outcomes. Don't report accuracy on imbalanced datasets without context.

---

## Coding standards *[AGENT RULES]*

- **Python**: 3.11+. Type hints on all function signatures. Google-style docstrings. `ruff` for linting and formatting. `pytest` for tests.
- **R**: Tidyverse style when in R. Use `renv` for dependency management.
- **Nextflow/Snakemake**: Each process/rule should be independently testable. Parameterise reference paths and resource allocations — don't hardcode.
- **No `any` in TypeScript** without explicit approval (keeping Boris's rule — relevant for any dashboard/viz work).
- **Git**: Conventional commits (`feat:`, `fix:`, `refactor:`, `data:`, `pipeline:`). Feature branches off `main`. Squash-merge PRs.

---

## Verification loops *[AGENT RULES]*

After completing any code change, run the applicable verification commands before reporting success:

```bash
# Core checks — run all that apply to the current project
python -m pytest tests/ -x -q          # Run tests, stop on first failure
ruff check . --fix                      # Lint and auto-fix
ruff format .                           # Format
mypy src/ --ignore-missing-imports      # Type check
nextflow run main.nf -profile test      # Pipeline smoke test (if Nextflow project)
```

Do not report a task as complete unless verification passes. If tests fail, fix and re-run. If a pipeline exists, do a dry-run or test-profile run. If training an ML model, confirm that training loss decreases and validation metrics are sane before declaring success.

---

## Workflow philosophy *[AGENT RULES]*

- **Plan before executing**: For any complex task, produce a plan first (input data format, expected output, tool versions, resource estimates) and confirm it with the user before writing implementation code. A good plan enables one-shot execution.
- **When things go sideways**: Stop, reassess, and re-plan. Do not keep pushing a broken approach. For pipeline debugging, re-examine assumptions about input data before touching code.
- **Subagents**: Limit to 1 concurrent subagent on this machine. Good uses: code simplification after implementation, architecture review, generating a test suite from a spec.
- **Self-correcting CLAUDE.md**: After any correction or mistake, propose an update to this file with a rule to prevent repeating it. Keep rules specific and actionable.
- **Handoff discipline**: When the user ends a session, or when context is getting long (15+ turns), proactively offer to write a `HANDOFF.md` summarizing: what was done, what's left, key decisions, current blockers. This lets the next session pick up cleanly.

## Workflow philosophy *[OPERATOR NOTES]*

- **Plan mode first**: Start sessions in plan mode (shift+tab twice). Iterate on the plan, then switch to auto-accept for execution.
- **Slash commands for the human**: Use `/commit-push-pr`, `/quick-commit`, `/test-and-fix`, `/review-changes` for your own inner-loop workflows. These are user-invoked shortcuts — Claude doesn't call them, but they save you repeated typing. Build domain-specific ones as needed (e.g., `/validate-pipeline`, `/check-data-schema`).
- **When to override**: If Claude proposes an approach that doesn't fit the project, switch back to plan mode and redirect rather than trying to patch mid-execution.

---

## Biopharma context *[AGENT RULES]*

- We work at the intersection of computational biology and drug discovery/development.
- Audiences for our code and analyses include:  ML/AI bioinformaticians especially, bench scientists, translational researchers, clinical data scientists, and occasionally regulatory reviewers.
- Documentation should be clear enough for a biologist to understand the methods section, and precise enough for a bioinformatician to reproduce.
- When discussing mRNA, immunotherapy, or any therapeutic modality — be precise about mechanism. Don't conflate checkpoint inhibitors with CAR-T with bispecifics with ADCs. Get the biology right.
- For anything touching patient data or clinical datasets: assume HIPAA/GxP constraints apply. Never log PHI. Use synthetic or de-identified data for development.

---

## MCP server policy *[AGENT RULES]*

Do not assume any MCP servers are preloaded. This setup uses on-demand MCP creation to avoid paying context-window tax for unused tool descriptions.

- **No preloaded MCPs by default**. The only exception is PubMed if the user has installed it — it's low-overhead and useful mid-session for checking published methods.
- **When an MCP would help**: If a task would benefit from structured access to an external API (e.g., Ensembl REST, UniProt, a LIMS, GEO/SRA), suggest creating one using the `mcp-maker` skill rather than telling the user to go find a premade integration.
- **Use the `mcp-maker` skill** (located at `.claude/skills/mcp-maker/SKILL.md`) to scaffold a FastMCP server, register it with Claude Code, and test it — all in one workflow.
- **Lifecycle**: MCPs created during a project should be committed to `.claude/mcp-servers/` if they'll be reused, or treated as disposable if they were session-specific. Suggest cleanup when a project wraps up.
- **Credentials**: Never hardcode API keys in MCP server source. Use environment variables loaded from `~/.env` or project `.env` (which must be in `.gitignore`).
- **Transport**: Default to `stdio` for local MCPs. Only use SSE/HTTP if the server needs to be shared or accessed remotely.

## MCP server policy *[OPERATOR NOTES]*

- To install the Anthropic life sciences marketplace: `/plugin marketplace add anthropics/life-sciences`
- Available official plugins: `pubmed`, `biorender`, `synapse`, `10x-genomics`, `wiley-scholar-gateway`, `nextflow-development`, `single-cell-rna-qc`, `scvi-tools`
- Install individually: `/plugin install pubmed@life-sciences`
- For anything not covered by an official plugin, use `/mcp-maker` or ask Claude to invoke the skill.

---

## Usage economy — stretching a Pro account

This setup runs on a Claude Pro subscription, not unlimited API.

### Token-saving rules *[AGENT RULES]*

- **Keep outputs compact**: When the user asks to explore data (e.g., "show me the first 10 rows"), limit output. Do not dump large stdout into tool results — it fills the context window fast.
- **Do not re-read files unnecessarily**: If you wrote a file within the last few turns, reference it by name. Do not `cat` it back unless the user asks.
- **Batch work**: When a task has multiple parts (write code, write tests, write docstring), do them in one turn rather than spreading across multiple rounds.
- **Proactive session management**: If the conversation has exceeded ~15 turns, or if you notice your context is heavy with dead-end debugging, suggest to the user: "This session is getting long. Want me to write a HANDOFF.md and we start fresh?" Do not wait to be asked.
- **Subagents for isolation**: For self-contained tasks (code simplification, test generation from a spec), prefer spawning a subagent over extending the main session. The subagent runs in its own context and doesn't carry the main session's history.
- **Front-load context in files, not conversation**: Reference `CLAUDE.md`, `CONTEXT.md`, `HANDOFF.md` rather than re-explaining constraints conversationally.

### Guardrail alerts *[AGENT RULES]*

Proactively warn the user when you detect any of the following. Do not wait to be asked — flag it inline, briefly, as it comes up.

| Signal | What to say |
|---|---|
| **Context bloat** | "Heads up — this session is getting long (~N turns). Want me to write a HANDOFF.md and start fresh?" |
| **Token-heavy output** | Before running a command that will produce large stdout (>100 lines), ask: "This will dump a lot of output. Want me to truncate or summarise instead?" |
| **Scope creep** | If a single prompt is asking for 3+ loosely related things: "This is turning into multiple tasks. Want me to batch them, or should we do them one at a time so I can verify each?" |
| **Complexity escalation** | If a file is exceeding ~300 lines, or a function exceeds ~50 lines, or a plan has >8 steps: "This is getting complex. Want me to break it into smaller pieces / use a subagent / simplify the design?" |
| **Stability risk** | Before making a change that could break the pipeline or an existing test suite: "This touches [X critical path]. I'll run verification before and after — just flagging the risk." |
| **Resource limits** | Before recommending a tool or operation that will stress 16 GB RAM (large model load, big join, full-genome indexing): "This may push memory. Here's a lighter alternative: [X]. Want to try that first?" |
| **Diminishing returns** | If a debug loop has gone >3 iterations without progress: "We've been circling on this. Want me to re-plan from scratch, or should we write up what we know in HANDOFF.md and come back fresh?" |

- **Opus for thinking**: Plan mode, architecture decisions, debugging subtle biology/stats, reviewing scientific soundness, writing CLAUDE.md rules.
- **Sonnet for doing**: Executing a locked plan, boilerplate generation, scaffolding, writing tests from a spec, refactoring, docstrings, git operations. Switch with `/model sonnet` mid-session.
- **Rule of thumb**: Plan mode → Opus. Auto-accept execution with a solid plan → Sonnet stretches your budget 3–5x.

### Context window hygiene *[OPERATOR NOTES]*

Long sessions burn usage exponentially (each message re-sends full history). Start a new session when:

- You've been going 15+ turns on the same task.
- Claude starts repeating itself, forgetting instructions, or hallucinating file contents.
- You've finished a distinct unit of work (pipeline passes tests → commit → fresh session for next feature).
- The context is cluttered with dead-end debugging threads.
- You're switching domains (e.g., Nextflow pipeline → PyTorch model).

**Before ending a session**, tell Claude to:
1. Write a `HANDOFF.md`: what was done, what's left, key decisions, current blockers.
2. Propose any `CLAUDE.md` updates from lessons learned.
3. Commit WIP to a branch.

Then start fresh. Claude reads `CLAUDE.md` + `HANDOFF.md` on startup and picks up without re-paying for the old context.

**Scratchpad pattern**: For multi-step tasks, maintain a `SCRATCHPAD.md` with the current plan and progress. Cheaper than Claude re-deriving the plan from conversation history.

### Daily rhythm *[OPERATOR NOTES]*

1. **Morning / fresh session**: Plan mode (Opus). Lay out today's work. Reference yesterday's `HANDOFF.md`.
2. **Execution blocks**: Switch to Sonnet. Execute in focused chunks. Commit after each.
3. **Review / debug**: Opus if something subtle breaks. Sonnet for straightforward test failures.
4. **End of day**: Claude writes `HANDOFF.md`, proposes `CLAUDE.md` updates, commits. Kill the session.

---

## Mistakes log

### Never derive biological sequences manually
When fixing or sourcing a protein sequence (UniProt, PDB chain, etc.), always download the
FASTA directly from the API (`curl https://rest.uniprot.org/uniprotkb/Q15116.fasta` or
`curl https://www.rcsb.org/fasta/entry/7EOW/download`) and grep/extract from the result.
Do not derive sequences by counting characters in existing code or reconstructing from
memory. LLMs have poor sequence fidelity - a manually derived "fix" can silently preserve
the original bug in different positions.

### Always verify pass/fail logic against actual tool response fields
Before using `result["nested"]["field"]` for pass/fail, confirm the field actually exists
in the tool's JSON response. Defaulting to `False` when a key is missing silently causes
every run to fail. Use a direct numeric comparison (e.g. `percentile < 95.0`) instead of
relying on a `passed` flag that may not be populated.

### Do not use `tool_choice="required"` for agentic design loops
With `tool_choice="required"`, the LLM jumps directly to making tool calls without a
reasoning step. For generate-and-screen loops (propose sequence → screen → critique →
mutate), use a text-only API call first (no `tools=` arg) so the model can reason and
write a new sequence. Then run screening functions locally. This prevents the "same sequence
every iteration" failure mode where the model re-screens without mutating.

### Do not use `ablang2` to load ablang1 models — use `ablang` directly
`ablang2` v0.2.1 wraps ablang1-heavy but has an internal tokenizer bug (`w_extra_tkns`
keyword argument) that breaks all forward passes. Use `ablang.pretrained("heavy")`
from the standalone `ablang` package instead. Do not suggest upgrading to ablang2
until the upstream tokenizer mismatch is fixed.

### Derive vocab→logit mapping from model internals at runtime
For language models where the logit dimension ordering is unclear, build the
AA→logit-index map from `model.tokenizer.vocab_to_token` at runtime (sort by token
index, enumerate). Do not hardcode index mappings — they can silently drift across
model versions.

<!-- Add entries here as they happen. Format: date, what went wrong, rule added. -->
<!-- Example:
- 2026-03-07: Claude used hg19 coordinates with an hg38 reference. Rule: always confirm genome build before writing pipeline code.
-->

---

## Project-specific overrides

<!-- Per-project settings go here. Copy this CLAUDE.md into each project root and customize below. -->
<!-- Example:
- This project uses Snakemake, not Nextflow.
- Reference genome: GRCh38 (Ensembl release 110).
- Primary language: R (Seurat v5 for single-cell).
-->

### Project-specific overrides

#### Role & Persona
You are acting as a Senior Computational Biologics Engineer at an agentic drug discovery startup. Your mindset is screening-first: every design must pass deterministic developability QC before advancing.

#### Core Scientific Domain Knowledge
- Targets: Nipah Virus G protein, Human PD-1 (Pembrolizumab epitope). The system is target-agnostic — new targets can be added without changing the screening tools.
- Binding strategy: Zero-shot approach inspired by the Escalante 180-line strategy (https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/).
- Scaffold: Camelid VHH (Nanobodies).
- VHH Hallmark Tetrad: Always monitor FR2 positions 37, 44, 45, and 47 (Kabat/Chothia).
- Liabilities: Deamidation (NG, NS, NA), Isomerization (DG), N-glycosylation (N-X-S/T). Zero tolerance in CDRs.
- Biophysics: pI > 7.5 and GRAVY ≤ 0.0 are hard requirements.
- APR: No hydrophobic patch exceeding the 95th percentile of clinical-stage therapeutics (1.934 mean KD/residue, 7-residue window).

#### Implemented Components
- `biologics_server.py` — FastMCP server with four deterministic tools: `calculate_biophysical_profile`, `scan_structural_liabilities`, `vhh_hallmark_audit`, `scan_aggregation_patches`. All return structured JSON.
- `scan_aggregation_patches` — Clinically-calibrated APR scanner. 7-residue sliding window over Kyte-Doolittle, scored as z-scores/percentiles against a reference distribution of 13 clinical-stage VH/VHH domains. Screening threshold: 95th percentile (1.934 mean KD/residue). Gold standard: Caplacizumab (max patch = 1.357, 40.5th percentile).
- `agent_loop.py` — Developability screening loop using Together AI (DeepSeek V3) via OpenAI-compatible API. Includes per-iteration cost tracking, 4-panel developability dashboard (pI, GRAVY, liability count, APR percentile; auto-opens on macOS). CoT logged to `logs/agent_cot.log`. Features: auto-profiling (runs missing tools after each iteration for complete metrics), carry-forward + back-fill for missing data points, seed sequences (`--seed naive|pembrolizumab|none`, default: naive).
- API provider: Together AI (US-hosted, `api.together.xyz`). Default model: `deepseek-ai/DeepSeek-V3`. Configurable via `MODEL_ID` env var.
- API key: `TOGETHER_API_KEY` env var (set in `~/.zshrc`, never committed).

#### SOTA Tool Preferences (for roadmap implementation)
- Structure prediction: Boltz-2 (not ESMFold or AlphaFold-Multimer). Best open-source accuracy for antibody-antigen complexes, MIT license.
- Antibody inverse folding: AntiFold (not ProteinMPNN). Purpose-built for antibody CDR sequence recovery.
- Humanness / immunogenicity scoring: AbLang2 or AntiBERTa2 (not ESM-2). Trained on antibody repertoires, perplexity reflects actual immunogenicity risk.
- MHC presentation prediction: BigMHC or PRIME 2.0. Mass-spec ground truth over binding-affinity-only models like NetMHCpan.

#### Coding Standard
1. Tool-first design: Every tool returns structured JSON that a downstream LLM can reason over.
2. Deterministic over generative: Use regex and BioPython for liability scanning, not LLM inference. Ground truth for screening.
3. Generate → screen loop: The Generator is explicitly critiqued by the screening tools. No design passes if any liability is detected.

#### Formatting & Tone
- Green terminal output for the agent's internal reasoning (Chain of Thought).
- Documentation uses terms like "Developability Screening" and "Developability Constraints."
- Reference the Escalante blog post in docstrings of any binding prediction logic.
- README tone: academic, rigorous, minimal inline bold.

#### Constraints
- If a user asks for a design, always run `vhh_hallmark_audit` first to check if the framework is properly humanized vs. stable camelid-style.

#### Known Issues (resolved)
- DeepSeek V3 position indexing: Previously struggled to locate residues by position number alone. Fixed by adding a `context` field to `scan_structural_liabilities` output (e.g. `"...TISRD[NS]KNTLY..."`). The model now correctly identifies and mutates the target residue.
- DeepSeek V3 pI charge engineering: The model understands it needs Lys/Arg substitutions but takes several iterations to actually apply them to the sequence. This is a model reasoning issue, not a tool issue.

#### Future Directions
1. ~~Structural screening: Boltz-2 for VHH-antigen complex prediction, FreeSASA for SASA-aware liability context.~~ **DONE** - `predict_vhh_complex_structure` and `filter_liabilities_by_sasa` are both implemented.
2. Inverse folding: AntiFold for CDR sequence optimization pre-conditioned on 3D scaffold coordinates. Prerequisite (Boltz-2) is done.
3. Immunogenicity: AbLang2/AntiBERTa2 for OAS-perplexity humanness scoring, BigMHC/PRIME 2.0 for MHC presentation prediction.
4. Search strategy: MCTS-based mutation exploration instead of linear loop. Multi-agent Generator vs. Screener adversarial debate.
5. Spatially-resolved aggregation: SAP mapping on solvent-exposed surface, aligned with TDC/TAP benchmarks.
