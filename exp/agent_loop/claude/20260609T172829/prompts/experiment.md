# Experiment Phase

## Task
Optimize ESG Stage 4 (verification_timeline) Codex prompt to beat ST4 Macro-F1 = 0.5109 on val_test.

## Your job
Run the experiments defined in the plan for loop {loop_id}.

## Required reading
- loops/loops{loop_id}/plans/
- loops/loops{loop_id}/dev/

## Eval harness
Stage4 codex prediction script: core/human/predict/stage4/pred_by_codex.py

Key env vars:
- DATA: path to benchmark JSON (val_public.json for selection, val_test.json for gate)
- STAGE4_PROMPT: path to the new prompt file
- CODEX_MODEL: gpt-5.5
- STAGE1_CSV: use cached ST1 predictions to skip re-running ST1:
  - val_public: data/benchmarks/val_public.json (use existing ST1 run or run fresh)
  - val_test: exp/continue/eval_val_for_stage1_bert-codex-rag_stage2_codex_stage3_bert_stage4_codex/output/stage1/stage1_bert_predictions.csv
    (IMPORTANT: verify this CSV's split before using it as val_test ST1 input)

Scoring: after predictions, compute ST4 Macro-F1 with:
  python3 -c "
  import json, csv
  from collections import defaultdict
  with open('data/benchmarks/val_test.json') as f:
      gt = {r['id']: r for r in json.load(f)}
  with open('<pred_csv>') as f:
      preds = {r['id']: r for r in csv.DictReader(f)}
  labels = set(r.get('verification_timeline','') for r in gt.values())
  tp,fp,fn = defaultdict(int),defaultdict(int),defaultdict(int)
  for id_,gt_r in gt.items():
      g = gt_r.get('verification_timeline','')
      p = preds.get(id_,{}).get('verification_timeline','')
      if g==p: tp[g]+=1
      else: fp[p]+=1; fn[g]+=1
  f1s=[]
  for l in sorted(set(list(tp)+list(fn))):
      pr=tp[l]/(tp[l]+fp[l]) if tp[l]+fp[l]>0 else 0
      rc=tp[l]/(tp[l]+fn[l]) if tp[l]+fn[l]>0 else 0
      f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0
      print(f'{l}: P={pr:.3f} R={rc:.3f} F1={f1:.3f}')
      f1s.append(f1)
  print(f'Macro-F1: {sum(f1s)/len(f1s):.4f}')
  "

## Protocol
1. Run each prompt variant on val_public (selection split). Report per-class F1.
2. Select best variant on val_public.
3. Run ONLY the best variant on val_test (gate split). No re-tuning after seeing val_test.
4. Compare val_test result against gate threshold from the plan.
5. Record generalization gap = val_public - val_test macro-F1.

## Experiment record
Include:
- Every command run (exact, copy-pasteable)
- Real stdout/stderr excerpts
- All metric values per variant (val_public)
- Final gate result on val_test
- Gate-check table (pass/fail/Δ per threshold)
- Artifact paths (prompt file paths, pred CSV paths)

## Output rules
- Raw markdown only. Start with `# Loop {loop_id} — Experiment`.
- Use Bash tool; record ACTUAL output — do not fabricate numbers.
- Write record to loops/loops{loop_id}/exp/{loop_id:03d}_agent_loop_exp.md before finishing.
