# Reflect Phase

## Task
Optimize ESG Stage 4 (verification_timeline) Codex prompt to beat ST4 Macro-F1 = 0.5109 on val_test.

## Your job
Evaluate loop {loop_id} and decide whether to promote the result.

## Required reading
- loops/loops{loop_id}/plans/
- loops/loops{loop_id}/dev/
- loops/loops{loop_id}/exp/
- docs/methods.md — check if ST4 module section needs updating

## Evaluation checklist
Answer each item explicitly (yes/no + evidence):

1. Did the val_public ST4 Macro-F1 exceed the acceptance threshold defined in the plan?
2. Did the val_test (gate) ST4 Macro-F1 beat 0.5109?
3. Is the generalization gap (val_public − val_test) within 0.05?
4. Is the method data-use compliant (prompt uses only data field at runtime; no post-processing)?
5. Was val_test excluded from selection/tuning (split discipline maintained)?
6. Does this method represent a genuinely new family vs. prior ST4 loops in this workspace?
7. Does docs/methods.md ST4 section need updating?
8. What concrete change should the next loop try? (specific, actionable)

## Promotion rules (CLAUDE.md gate)
A result may be promoted (accepted) only if ALL of the following hold:
- val_test ST4 Macro-F1 > 0.5109
- val_public ST4 Macro-F1 also improved (no overfit)
- Data-only compliance: runtime path uses only `data` field
- No post-processing (no keyword rule, no rule-based decision)
- Reflect record explicitly verifies every gate item

If promoted:
- Update docs/methods.md ST4 section
- Update docs/loops/agent_loop_state.json with new ST4 best artifact and scores

## Verdict
End with exactly one of:
  Verdict: accept
  Verdict: reject
  Verdict: defer

## Output rules
- Raw markdown only. Start with `# Loop {loop_id} — Reflect`.
- No preamble, no fence.
- Write record to loops/loops{loop_id}/reflect/{loop_id:03d}_agent_loop_reflect.md before finishing.
