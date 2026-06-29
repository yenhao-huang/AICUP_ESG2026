# Develop Phase

## Task
Optimize ESG Stage 4 (verification_timeline) Codex prompt to beat ST4 Macro-F1 = 0.5109 on val_test.

## Your job
Implement the plan for loop {loop_id}.

## Required reading
- CLAUDE.md in /workspace/esg_contest/
- loops/loops{loop_id}/plans/

## Implementation rules
- Use Edit, Write, and Bash tools to make ACTUAL file changes.
- Runtime data-use rule: prompt must only use the `data` field (no post-processing).
- New prompt files go to: configs/prompt/stage4/codex/ (named clearly per variant).
- If training few-shot examples: mine only from data/raw_data/vpesg_4k_train_1000.json using `data` + ground-truth labels (offline mining OK; runtime prompt uses data-only examples).
- No tuning against val_test.json; use val_public.json for selection decisions.

## Dev record
After implementing, write a dev record. Include:
- Every created/changed file and key changes
- How few-shot examples were selected (if applicable)
- Design decisions and trade-offs
- Data-use compliance confirmation
- Any blocked items

## Output rules
- Raw markdown only. Start with `# Loop {loop_id} — Dev`.
- No preamble, no fence.
- Write dev record to loops/loops{loop_id}/dev/{loop_id:03d}_agent_loop_dev.md before finishing.
