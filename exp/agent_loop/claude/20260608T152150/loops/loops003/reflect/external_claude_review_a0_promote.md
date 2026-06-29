# External Claude Review — a0 recipe Stage 1 promotion

Command: claude --dangerously-skip-permissions -p "<prompt in /tmp/a0_promote_review_prompt.txt>"
Timestamp (UTC): 20260609T011056Z
Model: 2.1.168 (Claude Code)

---

# External Review — Loop 001 a0 (real-only) ST1 promotion

I read CLAUDE.md, methods.md Stage 1, and the loop001/002 plan/exp/reflect records, and **re-verified every headline number directly from the raw eval JSON** (not the markdown). All figures reproduce exactly. Review stored at `loops/loops001/reflect/001_claude_review_a0_promotion.md` — which itself satisfies the mandatory Claude-review gate item that forced loop001/002 to *defer*.

## Verified numbers (from raw `st1.macro_f1`)
| metric | value |
|---|---|
| a0 val_test (blind gate) | **0.8109766** |
| BASE_GATE (exp4 val_test) | 0.802383 → **Δ +0.008594, strict** |
| a0 val_public (selection) | 0.8030692 |
| exp4 val_public | 0.7872834 → **Δ +0.015786** |

Candidate ckpt exists (1.30 GB); eval config is `cascade=False`, `stages=['st1']`, input = `data`; train file is 1000 real rows (Yes 814 / No 186, no synthesis).

## Answers to the five questions

**1. Data-use compliance — PASS.** a0 runs no synthesis, so `promise_string` is irrelevant to it (confirmed: never in training/runtime/prompt/rule/feature). Runtime is a single `data`-only BERT forward pass; no derived fields, no rule postprocess. Bonus: a0 also clears the stale methods.md L206–207 caveat ("training/export path needs audit") that hangs over the exp4 checkpoint — a0's training source is fully auditable canonical data.

**2. Score-first gate — PASS (strict).** Blind val_test 0.810977 > BASE_GATE 0.802383, strictly. Selected on val_public (both splits clean, 0/500 train-overlap, mutually disjoint); gate touched once; nothing tuned on the gate. Key point: a0 is the **lower** of the two val_public-tied arms on the blind gate (0.8110 vs a3_b1's 0.8234), so promoting a0 does *not* exploit the gate — it's the conservative, synthesis-free choice both prior reflects explicitly recommend, with the gain correctly attributed to the recipe rather than synthesis.

**3. Downstream — ST1-only artifact swap, but re-score integrated.** ST2–4 checkpoints are byte-identical, so for the standalone no-cascade ST1 metric there's no downstream concern. **Caveat:** per methods.md's integrated-scoring section, changing ST1 *predictions* mechanically moves integrated ST2/ST3/ST4 via the prediction-error penalty even with identical downstream models — so "ST2–4 Δ=0" holds for *artifacts* but not for *integrated* scores. Direction is favorable (more-accurate ST1 → fewer flips → ST3/ST4 hold or improve), but it should be **measured once**, not asserted as 0.

**4. methods.md update — YES.** Stage 1 Module A default checkpoint → `models/loop001_st1_a0_real_only/best_st1.pt`; recipe + data-only + scores note; same change in the Per-Stage Selectable Options table; add the data-use note that `test.json` is retired (200/200 leaked), `val_public` is canonical selection, `val_test` the locked gate. Create `docs/loops/agent_loop_state.json` (absent) with BASE_GATE 0.802383 + new pointer. Leave Module B (hybrid) unchanged.

**5. Risks/conditions.** Margin is modest (+0.0086, a few rows) but real and improves No-F1. Don't smuggle loop002's LLM-RAG pipeline under this clearance — that needs its own gate. Conditions: (a) store this review, (b) re-score the integrated weighted once and confirm ≥ baseline, (c) update methods.md + state file same session, (d) keep test.json retired.

REVIEW VERDICT: **conditional** — support-promote on the score and data-only merits (strict blind-gate pass, clean selection, fully data-only, auditable real-only recipe); conditioned on storing this review (done), a one-time integrated weighted-score re-check, and the methods.md / agent_loop_state updates.
