# Reflect Phase

## Task
Read `prompts/_task.md` (in this workspace) in full — it is the authoritative
task spec. ESG Stage 1 optimization. Blind gate: beat the baseline ST1 Macro-F1
on `data/benchmarks/val_test.json`. Runtime input is `data` only; promise_string
offline-only for synthesis.

## Your job
Evaluate loop {loop_id} and decide whether to promote the result.

## Required reading
- `prompts/_task.md` in this workspace
- /workspace/esg_contest/CLAUDE.md (Promotion Gate section)
- loops/loops{loop_id}/plans/
- loops/loops{loop_id}/dev/
- loops/loops{loop_id}/exp/
- /workspace/esg_contest/docs/methods.md (check if Stage 1 section needs updating)

## Evaluation checklist
Answer each item explicitly (yes/no + evidence):

1. Did the primary metric exceed the acceptance threshold on the dev split?
2. Did the blind-gate metric (val_test.json Macro-F1) beat the baseline?
3. Is the generalization gap (dev − blind) within the allowed limit?
4. Is the method data-use compliant — runtime uses ONLY `data`, and did
   promise_string stay strictly offline (synthesis only)?
5. Was blind/test data (val_public.json, val_test.json) excluded from tuning and
   threshold design (split discipline)?
6. Does the method introduce a genuinely new family vs prior loops?
7. Does docs/methods.md (or agent_loop_state) need updating?
8. What should the next loop try? (concrete, specific advice — e.g. next
   synthesis source, mix weight, when to introduce LLM-RAG fallback)

## Promotion gate (CLAUDE.md)
Promotion is score-first and reflect-gated. Do NOT promote unless every gate item
in the plan's score_first_promote section is explicitly verified and passed, the
blind-gate Macro-F1 beats baseline, and the runtime path is data-only. If any
required gate item is missing or fails, the verdict must be reject or defer.

## Verdict
End with exactly one of:
  Verdict: accept
  Verdict: reject
  Verdict: defer

If `accept` → update /workspace/esg_contest/docs/methods.md (Stage 1 section) to
record the accepted method, and note the promoted artifact path so the
orchestrator can update best_artifact.

## Output rules
- Raw markdown only. Start with `# Loop {loop_id} — Reflect`.
- No preamble, no fence.
- Write the record to loops/loops{loop_id}/reflect/{loop_id}_agent_loop_reflect.md
  before finishing.
