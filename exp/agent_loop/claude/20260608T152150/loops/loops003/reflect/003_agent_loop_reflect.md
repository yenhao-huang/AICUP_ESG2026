# Loop 003 — Reflect (promotion decision: a0 → new ST1 baseline)

User-directed promotion of the loop001 a0 real-only recipe BERT as the new
Stage 1 baseline. This reflect verifies every CLAUDE.md Promotion Gate item.

## Checklist (yes/no + evidence)
1. Primary metric beats threshold on selection set? **Yes** — val_public a0
   0.803069 > exp4 0.7873 (+0.0158).
2. Blind-gate metric beats baseline? **Yes** — val_test a0 0.810977 > BASE_GATE
   0.802383 (+0.008594, strict).
3. Generalization gap acceptable? **Yes** — val_public→val_test gap +0.0079
   (positive; no overfit to selection).
4. Data-use compliant? **Yes** — a0 uses NO synthetic data, so promise_string is
   irrelevant; runtime is a single `data`-only BERT forward pass; no rule
   postprocess. Training source is fully auditable canonical data.
5. Split discipline? **Yes** — selection on val_public, gate on val_test (both
   clean, mutually disjoint); contaminated test.json retired.
6. New family vs prior loops? **N/A for promotion** — this promotes the
   conservative real-only arm whose validity loop001/002 reflects established;
   it is the honest recipe win, not a new method claim.
7. methods.md / state need updating? **Yes — done this session** (Module A
   default ckpt, Per-Stage table, feature_list.json, workspace state.json,
   docs/loops/agent_loop_state.json created).
8. Next loop advice: pursue BERT-only calibration / per-class threshold on the
   weak No-F1 (a0 No-F1 ≈ 0.69) as a distinct family; report all future lifts
   relative to the a0 base on the blind set, not just over exp4. The loop002
   LLM-RAG fallback must NOT ride this clearance — it needs its own gate (its
   marginal gate lift over a0 was ~1 row).

## External review (CLAUDE.md mandatory)
`loops/loops003/reflect/external_claude_review_a0_promote.md` —
VERDICT: conditional support-promote. Conditions: (a) store review [done],
(b) integrated weighted re-check ≥ baseline [done: 0.5916→0.6101, +0.0185, no
stage regresses], (c) methods.md + state updates same session [done], (d) keep
test.json retired [done]. **All conditions satisfied.**

## Promotion gate result
Every gate item passes (see exp gate-check table). Integrated weighted strictly
improves with no stage regression. Promotion is valid under the current problem
definition and data-use rules.

Verdict: accept
