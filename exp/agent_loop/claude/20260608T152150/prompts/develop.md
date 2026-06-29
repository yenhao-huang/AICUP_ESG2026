# Develop Phase

## Task
Read `prompts/_task.md` (in this workspace) in full — it is the authoritative
task spec. ESG Stage 1 optimization: BERT classifier on synthetic + real data,
blind gate on `data/benchmarks/val_test.json`. Runtime input is `data` only;
promise_string is allowed OFFLINE for synthetic-data generation only.

## Your job
Implement the plan for loop {loop_id}.

## Required reading
- `prompts/_task.md` in this workspace
- /workspace/esg_contest/CLAUDE.md
- loops/loops{loop_id}/plans/

## Implementation rules
- Use Edit, Write, and Bash tools to make ACTUAL file changes. Reuse existing
  code (core/train/train_bert.py, core/eval/eval_bert.py, core/e2e/stage1.py,
  configs) — do not reinvent training/eval harness.
- Runtime data-use rule: inference path uses only `data`. promise_string may be
  used ONLY in offline synthesis scripts, never in the runtime/inference path.
- Tuning/threshold search must use only the dev split
  (data/benchmarks/test.json) — never val_public.json or val_test.json.
- Follow /workspace/esg_contest/docs/rules/ for any new file path/name; if a
  rule conflicts, pick the closest compliant path and note it.
- Synthetic data and trained checkpoints must go to sensible repo paths
  (e.g. data/generated/..., models/...); record exact paths in the dev record.

## Dev record
After implementing, write a dev record. Include:
- Every changed or created file and the key changes made
- Design decisions and trade-offs
- How dev/blind split discipline was maintained
- Data-use compliance notes (explicitly confirm promise_string did not leak
  into the runtime path)
- Any blocked items

## Output rules
- Raw markdown only. Start with `# Loop {loop_id} — Dev`.
- No preamble, no fence.
- Implement the changes with Edit/Write/Bash BEFORE writing the record.
- Write the record to loops/loops{loop_id}/dev/{loop_id}_agent_loop_dev.md
  before finishing.
