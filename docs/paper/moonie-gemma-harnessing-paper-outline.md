# Moonie Gemma Harnessing Paper Outline

Working title:

**Harnessing Local Gemma for Tool-Using Knowledge Work: Controller Dependence, Tool Contracts, and Replayable Live Evaluation**

## Core Claim

Local-first agent quality is a systems problem. On Moonie's current Gemma-on-MLX surface, task readiness can look solved while raw tool-protocol behavior remains dependent on controller repair, fallback planning, and argument normalization.

## Paper Shape

1. Abstract
   - State the problem: top-line task readiness hides controller dependence in local tool-using agents.
   - State the method: paired contracted/no-directive runs, helper ablations, exact replay, CLI-live replay, and catalog/prompt interventions.
   - State the main result: the final tool-turn directive is causal for raw protocol behavior; catalog shaping can move visual routing, but exact argument fidelity remains unresolved.

2. Introduction
   - Why local-first agent evaluation needs more than pass/fail task readiness.
   - Why Gemma/MLX is a useful target: local, inspectable, reproducible, and sensitive to harness design.
   - Contributions:
     - a benchmark-backed distinction between readiness, strict interface compliance, recovered execution, and controller burden
     - replayable exact-call packets for observed failures
     - CLI-live replay as an operator-visible discriminator
     - a visual tool-catalog intervention line with positive and negative results

3. Experimental System
   - Moonie runtime: packaged workflows, tool specs, controller repair/fallback, approvals, sandboxed live runs, artifacts.
   - Models/backends: local MLX Gemma row, HF specialist rows, reference rows.
   - Surfaces:
     - aligned `32 / 26` comparison surface
     - H1f/H1h/H1i/H1j/H1k slices
     - exact tool-directive probe
     - exact replay and CLI-live replay

4. Metrics
   - Real-world readiness.
   - Strict interface score.
   - Recovered execution.
   - Raw planning clean rate.
   - Controller repair/fallback/argument repair rates.
   - Exact tool-call match and executable visual match.
   - Why these metrics must be reported together.

5. Main Results
   - Readiness parity hides controller burden.
   - Removing controller repair/fallback/argument repair exposes causal dependence.
   - The final tool-turn directive is causal on exact replay: contracted `7 / 8`, no-directive `0 / 8`.
   - Packaged workflows can saturate and wash out raw one-turn failures.
   - CLI-live replay keeps the failure shape visible to the operator.

6. Visual Harnessing Case Study
   - Wave-three visual initiation recovers entry but not the filter case.
   - Wave-four/wave-five prompt wording is negative or weak.
   - `visual_role_catalog_v1` moves routing from wrong-tool/no-call to argument mismatch.
   - `visual_role_catalog_argument_hints_v2` reaches `2 / 3` focused visual live exactness but loses executable form-target recovery.
   - `visual_role_catalog_split_selector_hints_v3` is negative evidence: broader prose preserves latest-filter exactness but regresses readback shape and does not earn live replay.

7. Threats To Validity
   - Internal benchmark and local runtime only.
   - Focused replay packets are failure-conditioned, not population estimates.
   - Packaged workflows and exact replay measure different task pressures.
   - Some visual executor aliases allow executable non-exact matches, so exactness and executability must remain separate.
   - Gemini CLI is currently an external design/reference baseline, not a same-harness replacement.

8. Reproducibility
   - Evidence ledger: `results/reports/publication_evidence_ledger/ledger.md`.
   - Readiness audit: `results/reports/publication_readiness_audit/publication_readiness_audit.md`.
   - Main report: `results/reports/mlx_tool_contract_harnessing/report.md`.
   - Regeneration commands:

```bash
uv run python scripts/build_mlx_tool_contract_report.py
uv run python scripts/build_publication_evidence_ledger.py
uv run python scripts/audit_publication_readiness.py
uv run pytest tests/test_mlx_tool_contract_report.py tests/test_publication_evidence_ledger.py tests/test_publication_readiness_audit.py -q
```

## Current Paper-Ready Figures

- `h1h_h1i_controller_burden.svg`
- `tool_probe_contract_gap.svg`
- `exact_probe_replay_gap.svg`
- `live_replay_focus_gap.svg`
- `tool_catalog_profile_probe_gate.svg`
- `visual_catalog_argument_hints_live_candidate_replay_gate.svg`

## Next Evidence Needed Before Submission

- A fresh hard visual slice beyond the three current focused visual replay cases.
- A v4 schema-local or executor-grounded selector intervention that does not use broad prose.
- Repeated-seed or repeated-run variance for the strongest exact-replay claims.
- A clean table separating population-style benchmark claims from failure-conditioned replay claims.
