#!/usr/bin/env python3
"""Build submit-skill analysis.md for submission_13 (blind test2000, no labels).

Per stage: class_distribution (from submission.csv), score_distribution (BERT/
soft-vote/multitask stages), fallback_change_rate (ST1 Gemma only).
"""
import csv, json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent.parent          # submission_13/
sub = {r["id"]: r for r in csv.DictReader(open(D / "submission.csv"))}
N = len(sub)
out = []
def w(s=""): out.append(s)

BINS = [("<0.50", lambda c: c < 0.5), ("0.50-0.60", lambda c: 0.5 <= c < 0.6),
        ("0.60-0.70", lambda c: 0.6 <= c < 0.7), ("0.70-0.80", lambda c: 0.7 <= c < 0.8),
        ("0.80-0.90", lambda c: 0.8 <= c < 0.9), (">=0.90", lambda c: c >= 0.9)]

def score_block(confs, label, pct_base=None):
    n = len(confs); base = pct_base if pct_base else n
    w(f"--- score_distribution ({label}) ---")
    for name, f in BINS:
        c = sum(1 for x in confs if f(x))
        w(f"  {name:<12} {c:>6}  ({100*c/base:.1f}%)" if base else f"  {name:<12} {c:>6}")
    if confs:
        w(f"  median confidence    {statistics.median(confs):.6f}")
    w()

def class_block(col, label, na_label=None, active_pct=False):
    cnt = Counter(sub[i][col] for i in sub)
    w(f"--- class_distribution (final) ---")
    na = cnt.get("N/A", 0)
    active = N - na
    for k in sorted(cnt):
        if k == "N/A":
            continue
        denom = active if active_pct else N
        suffix = " of active" if active_pct else ""
        w(f"  {k:<28} {cnt[k]:>6}  ({100*cnt[k]/denom:.1f}%{suffix})")
    if na:
        w(f"  {(na_label or 'N/A'):<28} {na:>6}")
    w(f"  {'TOTAL':<28} {N:>6}")
    w()

sep = "=" * 60

# ================= ST1 =================
w(sep); w("ST1: promise_status"); w(sep); w()
class_block("promise_status", "ST1")
# score: soft-vote raw (avg members), N=all 2000
by = defaultdict(list)
for r in csv.DictReader(open(D / "stage1/tmp/softvote_raw.members.csv")):
    by[r["id"]].append((float(r["score_yes"]), float(r["score_no"])))
st1c = [max(sum(a for a, _ in m)/len(m), sum(b for _, b in m)/len(m)) for m in by.values()]
score_block(st1c, f"soft-vote raw, N=all {len(st1c)}")
# fallback (Gemma)
s = json.load(open(D / "stage1/summary.json"))
w(f"--- fallback_change_rate (Gemma, tau={s['conf_threshold']}) ---")
w(f"  total rows           {s['total_rows']:>6}")
w(f"  kept BERT (conf>=tau){s['kept_bert_rows']:>6}")
w(f"  escalated to Gemma   {s['escalated_rows']:>6}")
w(f"  changed by Gemma     {s['changed_by_llm']:>6}  ({100*s['changed_by_llm']/max(s['escalated_rows'],1):.1f}% of escalated)")
w(f"  overall change rate  {100*s['change_rate']:.2f}%")
w(f"    Yes->No            {s['yes_to_no']:>6}")
w(f"    No->Yes            {s['no_to_yes']:>6}")
w()

# ================= ST2 =================
w(sep); w("ST2: evidence_status"); w(sep); w()
class_block("evidence_status", "ST2", na_label="N/A (ST1=No)", active_pct=True)
# score from softvote_gated postprocess_reason, active = filter_passed==yes
st2c = []
for r in csv.DictReader(open(D / "stage2/softvote_gated.csv")):
    if r.get("evidence_status") != "N/A":   # active = ST1=Yes; filter_passed is all-yes, unusable
        m = re.findall(r"score_(?:yes|no)=([0-9.eE+\-]+)", r.get("postprocess_reason", ""))
        if m:
            st2c.append(max(float(v) for v in m))
score_block(st2c, f"soft-vote, gated active N={len(st2c)}")
w("(no fallback -- ST2 is soft-vote ensemble)"); w()

# ================= ST3 =================
w(sep); w("ST3: evidence_quality"); w(sep); w()
class_block("evidence_quality", "ST3", active_pct=True)
# score from bert_multitask_gated reason, active = real label
st3c = []
for r in csv.DictReader(open(D / "stage3/bert_multitask_gated.csv")):
    if r.get("evidence_quality") in ("Clear", "Not Clear", "Misleading"):
        m = re.findall(r"score_(?:clear|not_clear|misleading)=([0-9.eE+\-]+)", r.get("evidence_quality_reason", ""))
        if m:
            st3c.append(max(float(v) for v in m))
score_block(st3c, "multitask, active rows w/ score")
w("(no fallback -- ST3 is bert_multitask)"); w()

# ================= ST4 =================
w(sep); w("ST4: verification_timeline"); w(sep); w()
class_block("verification_timeline", "ST4", active_pct=True)
w("(no score section -- ST4 is codex add-context)")
w("(no fallback -- codex add-context, v6 prompt, ST1-gated)"); w()

txt = "\n".join(out)
(Path(__file__).resolve().parent / "analysis.md").write_text(txt)
print(txt)
