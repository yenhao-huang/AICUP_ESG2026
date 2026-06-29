#!/usr/bin/env python3
"""Stage3 fallback: merge multitask-BERT predictions with pre-computed codex preds.

BERT (multitask) is primary. This does NOT call codex live — it indexes into a
full codex prediction file (codex already predicted all rows offline) and uses
the codex label only for the BERT rows whose confidence is below threshold.

Flow
----
1. Collect BERT rows whose max-class probability (parsed from
   `evidence_quality_reason`) < --conf-threshold, excluding rows already gated to
   "N/A" (those keep BERT / stay N/A).
2. For each collected low-confidence id, look it up in the full codex prediction
   file and overwrite evidence_quality with the codex label (source =
   "codex_fallback"). High-confidence rows keep BERT; a low-conf id missing from
   the codex file keeps its BERT label (counted + warned).

Output schema = stage3 OUTPUT_COLUMNS (id, evidence_quality,
evidence_quality_raw, evidence_quality_source, evidence_quality_reason).

Usage
-----
    python .../submit_5/submit/scripts/bert_fallback_codex_for_stage3.py \
        --bert-pred  .../stage3/tmp/bert_raw.csv \
        --codex-pred .../stage3/tmp/stage3_codex_predictions_merge.csv \
        --output     .../stage3/tmp/bert_codex.csv \
        --conf-threshold 0.76
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT))

from core.human.predict.stage3 import schema  # noqa: E402
from core.human.predict.stage3.pred_by_bert_gemma import bert_confidence  # noqa: E402

FIELDNAMES = list(schema.OUTPUT_COLUMNS)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bert-pred", type=Path, required=True,
                    help="Multitask BERT stage3 CSV (id + evidence_quality + evidence_quality_reason).")
    ap.add_argument("--codex-pred", type=Path, required=True,
                    help="Full codex stage3 CSV predicting all rows (id + evidence_quality).")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--conf-threshold", type=float, default=0.76,
                    help="BERT max-prob below this routes the row to the codex label (default 0.76).")
    ap.add_argument("--quality-col", default="evidence_quality")
    ap.add_argument("--reason-col", default="evidence_quality_reason")
    args = ap.parse_args()

    bert_rows = read_csv(args.bert_pred)
    codex_by_id = {str(r["id"]).strip(): r for r in read_csv(args.codex_pred)}

    n_total = len(bert_rows)
    n_gated = n_lowconf = n_applied = n_missing = 0
    out_rows: list[dict] = []

    for r in bert_rows:
        rid = str(r.get("id", "")).strip()
        quality = str(r.get(args.quality_col, "")).strip()
        row = {c: r.get(c, "") for c in FIELDNAMES}
        row["id"] = rid

        if quality == "N/A":
            n_gated += 1                                   # gated -> keep BERT (N/A)
            out_rows.append(row)
            continue

        conf = bert_confidence(r.get(args.reason_col, ""))
        if conf < args.conf_threshold:
            n_lowconf += 1
            codex = codex_by_id.get(rid)
            if codex is not None:
                row[args.quality_col] = str(codex.get(args.quality_col, "")).strip()
                row["evidence_quality_raw"] = str(codex.get("evidence_quality_raw", ""))
                row["evidence_quality_reason"] = str(codex.get("evidence_quality_reason", ""))
                row["evidence_quality_source"] = "codex_fallback"
                n_applied += 1
            else:
                n_missing += 1                             # low-conf but no codex row -> keep BERT
                if not row.get("evidence_quality_source"):
                    row["evidence_quality_source"] = "bert"
        else:
            if not row.get("evidence_quality_source"):
                row["evidence_quality_source"] = "bert"    # high-conf -> keep BERT

        out_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"total rows           : {n_total}", flush=True)
    print(f"gated N/A (kept BERT): {n_gated}", flush=True)
    print(f"low-conf (< {args.conf_threshold}) : {n_lowconf}", flush=True)
    print(f"  -> codex applied   : {n_applied}", flush=True)
    print(f"  -> missing in codex: {n_missing}", flush=True)
    print(f"high-conf (kept BERT): {n_total - n_gated - n_lowconf}", flush=True)
    print(f"Output               : {args.output}", flush=True)

    if n_missing:
        print(f"WARNING: {n_missing} low-conf ids absent from codex file", file=sys.stderr)


if __name__ == "__main__":
    main()
