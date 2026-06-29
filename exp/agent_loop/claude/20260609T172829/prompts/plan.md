# Plan Phase

## Task
Optimize ESG Stage 4 (verification_timeline classification) Codex prediction to beat baseline ST4 Macro-F1 = 0.5109 on data/benchmarks/val_test.json.

Baseline prompt: exp/integrated_stage_predictions/best_comb/detail_method/prompts/stage4_balance_rule_v1.txt (gpt-5.5)
Per-class baseline (val_test): already=0.454, within_2_years=0.222, between_2_and_5_years=0.575, more_than_5_years=0.603, N/A=0.701

Key weaknesses:
- within_2_years: P=0.143 R=0.500 — massively over-predicted as between_2_and_5_years
- already: P=0.621 R=0.358 — recall too low, completed events classified as future

Constraints:
- No post-processing (no keyword rules, no rule-based post-process)
- Data-only runtime: only `data` field allowed; no evidence_string, promise_string, etc.
- Selection split: data/benchmarks/val_public.json
- Gate split: data/benchmarks/val_test.json
- Training data for few-shot mining: data/raw_data/vpesg_4k_train_1000.json (data field + labels only)
- Model: gpt-5.5 (CODEX_MODEL env var)

## Your job
Write a method-level plan for loop {loop_id} (target: 3).

## Required reading
- CLAUDE.md in /workspace/esg_contest/
- docs/methods.md
- All prior plans in this workspace: loops/loops*/plans/*.md (build a Novelty Check)
- The baseline prompt: exp/integrated_stage_predictions/best_comb/detail_method/prompts/stage4_balance_rule_v1.txt
- Training data label distribution: analyze data/raw_data/vpesg_4k_train_1000.json for ST4 class distribution

Current best_artifact metrics:
- ST4 Macro-F1 (val_test) = 0.5109
- already=0.454, within_2_years=0.222, between_2_and_5_years=0.575, more_than_5_years=0.603, N/A=0.701
- No ST4 entry in docs/loops/agent_loop_state.json yet; baseline is the codex prompt above

## Plan requirements
1. One distinct method family — not just a hyperparameter tweak of the existing prompt.
2. Multiple variants or ablations inside this loop (test a grid, not one point).
3. **Novelty Check**: name closest prior loop and explain method-level difference. (This is the first ST4 loop, so note that prior loops covered ST1 and ST2 only.)
4. Baseline, primary metric, secondary metrics.
5. Acceptance and rejection thresholds (score_first_promote gate): must define ST4 val_test threshold AND which gate baseline artifact this is compared against.
6. Data-use boundaries.
7. Failure modes and abandonment criteria.
8. A `## Parallel Subtasks` section at the end.

## Output rules
- Raw markdown only. Start with `# Loop {loop_id} — …`.
- No preamble, no fence, no commentary outside the plan content.
- Write the file to loops/loops{loop_id}/plans/{loop_id:03d}_agent_loop_plan.md before finishing.
