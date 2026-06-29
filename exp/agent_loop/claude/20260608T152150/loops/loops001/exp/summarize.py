"""Summarize loop001 ST1 eval JSONs: macro-F1 + per-class (No/Yes) F1/recall."""
import json, glob, sys
from sklearn.metrics import f1_score, precision_recall_fscore_support

def row(path):
    d = json.load(open(path))["predictions"]["st1"]
    p, l = d["predictions"], d["labels"]
    macro = f1_score(l, p, average="macro", zero_division=0)
    pr, rc, f1, sup = precision_recall_fscore_support(l, p, labels=[0, 1], zero_division=0)
    return macro, f1[0], rc[0], sup[0], f1[1], rc[1], sup[1]

def main(globpat):
    files = sorted(glob.glob(globpat))
    print(f"{'arm':22s} {'macroF1':>8s} {'No_F1':>7s} {'No_R':>6s} {'No_n':>5s} {'Yes_F1':>7s} {'Yes_R':>6s} {'Yes_n':>6s}")
    rows = []
    for f in files:
        arm = f.split('/')[-1].replace('_dev.json', '').replace('.json', '')
        try:
            m, nf, nr, nn, yf, yr, yn = row(f)
            rows.append((arm, m))
            print(f"{arm:22s} {m:8.4f} {nf:7.4f} {nr:6.4f} {nn:5d} {yf:7.4f} {yr:6.4f} {yn:6d}")
        except Exception as e:
            print(f"{arm:22s} ERR {e}")
    if rows:
        best = max(rows, key=lambda x: x[1])
        print(f"\nBEST DEV ARM: {best[0]}  macroF1={best[1]:.4f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "exp/agent_loop/claude/20260608T152150/loops/loops001/exp/dev_eval/*_dev.json")
