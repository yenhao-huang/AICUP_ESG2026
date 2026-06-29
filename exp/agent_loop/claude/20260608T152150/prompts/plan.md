# Plan Phase

## Task
Read `prompts/_task.md` (in this workspace) in full — it is the authoritative
task spec. Summary: optimize ESG Stage 1 (Promise Identification, Yes/No,
Macro-F1) via a BERT classifier trained on synthetic + real data, with an
optional low-confidence LLM-RAG fallback introduced in a later loop. Blind gate:
beat the baseline Macro-F1 on `data/benchmarks/val_test.json`. Runtime input is
`data` only; promise_string is allowed OFFLINE for synthetic-data generation only.

## Your job
Write a method-level plan for loop {loop_id} (target: {target_loop}).

## Required reading
- `prompts/_task.md` in this workspace
- /workspace/esg_contest/CLAUDE.md (problem definition, data-use, promotion gate)
- /workspace/esg_contest/docs/methods.md (Stage 1 section)
- All prior plans: loops/loops*/plans/*.md (build a Novelty Check)

## Plan requirements
1. One distinct method family — not just a hyperparameter point.
2. Multiple variants or ablations inside this loop (e.g. synthesis source,
   mix weight, loss, threshold — a real grid, not one point).
3. **Novelty Check**: name the closest prior loop and the method-level difference.
4. Baseline (cite the val_test.json baseline number; if not yet measured, the
   first variant must measure it), primary metric, secondary metrics.
5. `score_first_promote` gate: baseline artifact, candidate artifact path,
   primary weighted/Macro-F1 threshold on the blind gate, tolerances, and
   data-only compliance requirement for the promoted runtime path.
6. Data-use boundaries (runtime = `data` only; promise_string offline-only).
7. Failure modes and abandonment criteria.
8. A `## Parallel Subtasks` section at the end listing parallelisable parts
   (format: `- id: X  phase: dev|exp|both  description: …`), or `(none)`.

## Output rules
- Raw markdown only. Start with `# Loop {loop_id} — …`.
- No preamble, no fence, no commentary outside the plan content.
- Write the file to loops/loops{loop_id}/plans/{loop_id}_agent_loop_plan.md
  before finishing.
