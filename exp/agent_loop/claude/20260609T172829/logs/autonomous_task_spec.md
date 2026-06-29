# Autonomous Task Spec

Generated: 2026-06-09T17:28:29Z

```
Task spec (autonomous):
  Objective:   ST4 Codex verification_timeline optimization (ESG contest)
  Baseline:    exp/integrated_stage_predictions/best_comb/detail_method/prompts/stage4_balance_rule_v1.txt
               ST4 val_test Macro-F1 = 0.5109
               Per-class: already=0.454, within_2_years=0.222, b2_5y=0.575, m5y=0.603, N/A=0.701
  Primary:     ST4 Macro-F1 on val_test   accept > 0.5109   reject ≤ 0.5109
  Secondary:   val_public ST4 Macro-F1 must also improve; generalization gap ≤ 0.05
  Data-use:    input = data field only; forbidden = evidence_string, promise_string, post-processing
  Constraints: no keyword rules; no rule-based post-process; model = gpt-5.5
  target_loop: 3
  mode:        autonomous
```

## Source
- Spec file: docs/plans/0610_optimize_st4.md
- Prior loops covered: ST1 (exp/agent_loop/claude/20260608T152150), ST2 (exp/agent_loop/claude/20260608T174056)
- This is the first ST4 optimization loop run
