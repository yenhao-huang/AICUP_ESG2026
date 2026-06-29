---
name: loop-analysis
description: Produce a fixed-format markdown analysis of agent-loop runs (plan/dev/exp/reflect) for ESG-contest style controllers such as `exp/claude_agent_loop` or `exp/agent_loop_controller`. Use when the user asks to "分析 loops X-Y", "整理 loops", "撰寫 loop 觀察", or otherwise summarize multiple loop iterations into a single observation document.
---

# Loop Analysis Skill

Produce a uniform, scan-friendly markdown writeup of N loop iterations from an
agent-loop controller (e.g. `exp/claude_agent_loop/loops/loopsNNN/`).

## When to use

Invoke this skill when the user asks to analyze, summarize, or compare multiple
agent-loop iterations — typical phrasings include "分析 loops X-Y", "整理 loop
觀察", "loops X 到 Y 寫到 …", "把 loop 進度整理一下". Each loop directory is
expected to contain four phase subdirectories: `plans/`, `dev/`, `exp/`,
`reflect/`.

## Inputs to confirm with the user

Before writing, make sure you know:

1. **Loop range** — explicit start and end loop IDs (e.g. 1–7).
2. **Source directory** — usually
   `exp/<controller>/loops/loopsNNN/` (claude_agent_loop or agent_loop_controller).
   If only one controller's `loops/` directory is present, infer it.
3. **Output path** — where to write the result. Default to
   `docs/plans/<controller>_loop_observe/loops_<start>_<end>_<stage>_analysis.md`
   if not given.

If any of these are ambiguous, ask once, then proceed.

## Required reading per loop

For each loop in range, read:

- `loopsNNN/plans/NNN_agent_loop_plan.md` — method family, gates, novelty check
- `loopsNNN/dev/NNN_agent_loop_dev.md` — what was actually implemented
- `loopsNNN/exp/NNN_agent_loop_exp.md` — variant grid, metrics, verdict draft
- `loopsNNN/reflect/NNN_agent_loop_reflect.md` — verdict, gate table, next-loop advice

If a phase file is missing, mark that loop as "pending" or "incomplete" in every
section rather than skipping.

Also read once at the start:

- The controller README (e.g. `exp/<controller>/README.md`) for context.
- The most recent reflect's stated baseline metrics, so the Baseline-vs-Methods
  table compares against the same numbers each loop references.

## Output format (mandatory — keep this exact section order)

Always emit these six top-level sections, in this order, with these headings:

```
# <title>
## 0. TOC
## 1. Baseline vs Methods
## 2. Plan / Dev / Exp / Reflect 對齊狀況
## 3. Gate 通過 / 失敗總表
## 4. 下一步建議
## 5. 方法介紹
```

Use Traditional Chinese in headings and prose by default; keep metric names,
artifact paths, and rule identifiers in their original form (usually English or
mixed). Do not invent metrics — only report what the loop records actually
contain.

### Section 0 — TOC

A simple anchor list of the five subsequent sections, plus a one-line "範圍"
note stating loop range, controller, and primary optimization target
(e.g. "Stage 3 evidence_quality on top of best_comb"). Example:

```markdown
## 0. TOC

範圍：loops 001–007、controller `exp/claude_agent_loop`、Stage 3
(`evidence_quality`) on `best_comb`。

1. [Baseline vs Methods](#1-baseline-vs-methods)
2. [Plan / Dev / Exp / Reflect 對齊狀況](#2-plan--dev--exp--reflect-對齊狀況)
3. [Gate 通過 / 失敗總表](#3-gate-通過--失敗總表)
4. [下一步建議](#4-下一步建議)
5. [方法介紹](#5-方法介紹)
```

### Section 1 — Baseline vs Methods

Lead with a 2-line baseline block (artifact path + the 4-5 primary metrics).
Then a single table comparing each loop's best variant against the baseline.
Required columns:

| Loop | Method family | Best variant | Primary metric | Δ vs baseline | Weighted | Verdict |

- Use the loop's own "best variant" name (the one its exp record bolds).
- "Primary metric" is whatever the plan declared as primary (usually
  `evidence_quality F1` for Stage 3 loops).
- Δ is signed and rounded to the precision the exp record uses.
- Verdict is the final reflect verdict (`accept` / `reject` / `defer` /
  `pending`).
- If a loop is incomplete, fill cells with `—` and mark Verdict as `pending`.
- **Sort rows by Δ vs baseline descending** (largest improvement first). Loops
  with no Δ (pending / incomplete) go at the bottom. Loop column is for
  identification only — do not order by loop ID. This makes "what worked best"
  the first thing the reader sees.

Optionally add a second smaller table for reference comparators that loops cite
but didn't run themselves (e.g. `score_first_greedy`, `families_abcd_full`),
labelled "Reference comparators".

### Section 2 — Plan / Dev / Exp / Reflect 對齊狀況

One row per loop. Required columns:

| Loop | Planned method | Implemented method | 對齊 | 備註 |

- "對齊" is one of: `aligned` / `pivoted` / `partial` / `stale reflect` /
  `incomplete`.
- "備註" explains the mismatch in ≤ 25 字 if not `aligned`, otherwise leave `—`.
- Capture known pivots even when the experiment still produced numbers
  (e.g. loop 006 planned TACMAR, ran OKPR).
- Flag any duplicate or stale reflect record and which one supersedes.

### Section 3 — Gate 通過 / 失敗總表

A wide table, one row per loop, one column per gate item the plans share.
Standard column set when analyzing Stage 3 best_comb loops:

| Loop | EQ F1 ≥ accept | Weighted ≥ accept | ev_status non-regress | promise_status non-regress | timeline non-regress | Cascade=0 | Correct-after ≥ floor | Data-only |

Cell values: `PASS`, `FAIL (Δ)`, or `—`. Include the shortfall in parentheses
for FAIL cells where possible (e.g. `FAIL (−0.005)`).

If the loops target a different stage, replace the metric columns with that
stage's gates but keep the structure: primary metric, composite metric, all
non-regression metrics, structural gates (cascade, data-only), precision/
diagnostic gates.

End the section with a one-line summary: "X / Y loops fully pass; Z loops in
defer zone; W rejected."

### Section 4 — 下一步建議

Pull from the most recent reflect's "Concrete Modification Advice" (or
equivalent) for the next loop. Present as a numbered list of 3–6 items, each
1–2 sentences. Where the recent reflect mandates a method, label it
**(mandatory)** or **(fallback)** as the reflect did.

If the latest loop is incomplete (no reflect yet), pull guidance from the
previous loop's reflect and explicitly note "依 loop N-1 reflect".

### Section 5 — 方法介紹

One subsection per loop, in chronological order. Heading format:
`### Loop NNN — <方法家族縮寫> (<完整英文名稱>)`

Each subsection must contain four labeled bullet groups (use exactly these
labels):

- **計畫家族**：what the plan proposed (method family, key hypothesis, # of
  variants).
- **執行結果**：best variant name, primary metric, weighted, key counts (changed
  rows, correct-after).
- **關鍵發現**：3–5 bullets pulled from exp "Key Findings" / reflect "Key
  Findings". Prefer findings that falsify a hypothesis or carry forward to
  later loops.
- **Verdict**：final reflect verdict in bold, plus one sentence on why
  (e.g. "差 acceptance gate −0.005"). Note any plan→dev→exp pivot or stale
  reflect.

Keep each subsection ≤ ~25 lines. Do not paste raw variant grids — summarize.

## Cross-loop hygiene rules

- **Never invent numbers.** If the exp and dev records disagree, report both
  and tag which is treated as authoritative (usually exp).
- **Cite paths sparingly.** Inline path mentions OK; do not dump large file
  listings.
- **Watch for stale reflect files.** If `reflect/NNN_*.md` describes a
  different method than `exp/NNN_*.md`, flag it in Section 2 and use the exp
  record for Section 1 numbers.
- **Cross-check memory.** If auto-memory contains a "loop NNN accepted"
  entry but the reflect on disk says `reject`/`defer`, the memory likely refers
  to a different controller's loop NNN. Note this in Section 2 备注 and trust
  the on-disk reflect for this analysis.
- **Use the verbatim verdict word** (`accept` / `reject` / `defer`) from the
  reflect's last line, even if you think the analysis disagrees.

## Suggested workflow

1. Resolve loop range, controller, and output path.
2. For each loop, read four phase files. Capture: best variant name + primary
   metric, weighted score, verdict, planned vs implemented method, gate
   table, key findings, next-loop advice.
3. Build Section 1 table first (it's the densest and reveals discrepancies).
4. Build Section 3 gate table second — it forces you to check every metric
   against every threshold, surfacing FAILs you might miss in prose.
5. Build Section 2 alignment table while the phase comparisons are fresh.
6. Build Section 4 from the latest reflect only; do not aggregate across
   loops (that's Section 5's job).
7. Write Section 5 last; this is where per-loop nuance lives.
8. Write the file at the resolved output path.
9. Report to the user: file path, # loops analyzed, # accept/defer/reject, and
   one sentence on the most important open issue.

## What NOT to do

- Do not write a "summary" paragraph at the very top before Section 0; the TOC
  is the entry point.
- Do not include narrative analysis of why the loops collectively did/didn't
  work outside Section 5 (and even there, scope it to per-loop findings). The
  橫向觀察 of cross-loop patterns belongs at the END of Section 5 only if it
  adds something not already in the gate table or alignment table.
- Do not include code blocks unless quoting a small command from an exp record
  (≤ 3 lines) that's critical to understanding the result.
- Do not commit the output file or run any git commands unless the user
  explicitly asks.
