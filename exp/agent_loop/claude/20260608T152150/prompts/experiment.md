# Experiment Phase

## Task
Read `prompts/_task.md` (in this workspace) in full — it is the authoritative
task spec. ESG Stage 1 optimization. Blind gate: beat the baseline ST1 Macro-F1
on `data/benchmarks/val_test.json`. Runtime input is `data` only.

## Your job
Run the experiments defined in the plan for loop {loop_id}.

## Required reading
- `prompts/_task.md` in this workspace
- loops/loops{loop_id}/plans/
- loops/loops{loop_id}/dev/

## Protocol
1. If the val_test.json baseline number is not yet recorded in a prior loop,
   measure the baseline model
   (models/exp4_optimize2_highconf_yes_balanced_no_large) ST1 Macro-F1 on
   data/benchmarks/val_test.json FIRST and record it as the gate baseline.
2. Tune thresholds/hyperparameters ONLY on the dev split
   (data/benchmarks/test.json). Report primary metric + per-class breakdown.
3. Run the SAME tuned method (no re-tuning) on data/benchmarks/val_public.json
   (test/report split) and on data/benchmarks/val_test.json (blind gate).
4. Compute generalization gap = dev metric − blind metric.
5. Check against the acceptance thresholds and score_first_promote gate from the
   plan.
6. If an experiment cannot run, record the exact blocker (command + error).

## Experiment record
Write a record. Include:
- Every command run (exact, copy-pasteable)
- Real stdout/stderr excerpts (NOT fabricated)
- All metric values for every variant (dev / val_public / val_test Macro-F1 +
  per-class F1)
- Artifact paths (checkpoints, prediction CSVs)
- Gate-check table (pass / fail / Δ vs baseline for each threshold)

## Output rules
- Raw markdown only. Start with `# Loop {loop_id} — Experiment`.
- No preamble, no fence.
- Use the Bash tool to run commands. Record ACTUAL stdout/stderr and real metric
  values — do not fabricate numbers.
- Write the record to loops/loops{loop_id}/exp/{loop_id}_agent_loop_exp.md
  before finishing.
