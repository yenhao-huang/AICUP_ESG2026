# Stage 4 test_add_context — Results

Data-use: probe only (context / image / promise_string exceed `data`-only).
Not promotable as a data-only runtime path. Vanilla = data-only baseline
(= `configs/prompt/stage4/codex/boundary_rules_v4.txt`).

Backend: Codex gpt-5.5, CONC=8. Scorer: `score_st4.py` (full-coverage Macro-F1
over the 4 live timeline classes). GT labels used for offline scoring only.

## 100-row screen (`data/val_yes.100.json`, 99 rows) — 2026-06-17

| variant     | Macro-F1 | Δvanilla | acc    | already | within_2y | between_2_5 | more_than_5 |
| ----------- | -------- | -------- | ------ | ------- | --------- | ----------- | ----------- |
| vanilla     | 0.6440   | —        | 0.5455 | 0.5783  | 1.0000    | 0.5128      | 0.4848      |
| add_promise | 0.6012   | -0.0428  | 0.6061 | 0.6667  | 0.6667    | 0.5867      | 0.4848      |
| add_context | 0.5761   | -0.0679  | 0.5758 | 0.6522  | 0.6667    | 0.5152      | 0.4706      |
| all         | 0.5574   | -0.0866  | 0.5758 | 0.6517  | 0.5714    | 0.5217      | 0.4848      |
| add_image   | 0.5564   | -0.0876  | 0.5859 | 0.6667  | 0.5714    | 0.5634      | 0.4242      |

### Read

- Macro-F1 ranking is dominated by a **2-row `within_2_years` artifact**: vanilla
  gets both right (F1 1.0); any enrichment missing one drops it to ~0.57-0.67,
  and equal-weighted macro turns that single row into a ~0.1 swing. The headline
  Macro-F1 is **not trustworthy at this sample size**.
- Setting that class aside, every enrichment **raises accuracy** (0.5455 ->
  0.576-0.606) and improves the populated classes (`already` 0.58 -> 0.65-0.67;
  `between_2_5` best under add_promise 0.587).
- `add_promise` is the least-harmful / best-accuracy enrichment. `add_image` and
  `all` are worst, and the page image specifically hurts `more_than_5_years`
  (0.48 -> 0.42) — table/near-term years on the page likely mislead long-horizon
  promises.

### Decision

100-row screen is **inconclusive on the primary metric**. A full 491-row run
(`SET=full`) is required for a verdict (within_2_years = 11 rows; already 227,
between 144, more5 109 are solid). Pending user sign-off on the full run.
