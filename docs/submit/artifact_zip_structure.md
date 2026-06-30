# Submit Artifact Zip Structure

Archive file:

```text
aicup_esg2026_submit_artifacts.zip
```

When uploaded to Google Drive, the archive may be split into parts to avoid
large single-file upload failures:

```text
aicup_esg2026_submit_artifacts.zip.part-000
aicup_esg2026_submit_artifacts.zip.part-001
...
aicup_esg2026_submit_artifacts.zip.parts.txt
```

Rebuild the zip after downloading all parts:

```bash
cat aicup_esg2026_submit_artifacts.zip.part-* > aicup_esg2026_submit_artifacts.zip
```

This zip is built for `scripts/submit.sh`. Extract it at the repository root so
the paths below land in their expected locations.

```bash
unzip aicup_esg2026_submit_artifacts.zip
```

## Top-Level Contents

```text
artifact_manifest.txt
artifact_manifest_details.tsv
configs/
data/
models/
```

## Required Data

```text
data/raw_data/vpesg4k_test_2000.json
```

Used by all submit stages as the default `DATA` input.

## Required Prompt

```text
configs/prompts/stage4/boundary_rules_v4.txt
```

Used by `scripts/predict/predict_codex_for_stage4.sh`.

## Required Models

Stage 1 ensemble checkpoints:

```text
models/submission/stage1/*/best_st1.pt
```

Stage 2 ensemble checkpoints:

```text
models/submission/stage2/*/best_st2.pt
```

Stage 3 multitask checkpoint:

```text
models/submission/stage3/w0_2_0_3_0_5/best_multitask_st3.pt
```

Gemma Stage 1/2 fallback adapter:

```text
models/submission/st12_fallback/gemma4_st12_mix/adapter_config.json
models/submission/st12_fallback/gemma4_st12_mix/adapter_model.safetensors
models/submission/st12_fallback/gemma4_st12_mix/README.md
models/submission/st12_fallback/gemma4_st12_mix/tokenizer.json
models/submission/st12_fallback/gemma4_st12_mix/tokenizer_config.json
models/submission/st12_fallback/gemma4_st12_mix/train_meta.json
models/submission/st12_fallback/gemma4_st12_mix/training_args.bin
```

Gemma base model:

```text
models/gemma/base/unsloth-gemma-4-12b/
```

## Notes

- Symlinks are dereferenced when building the zip, so the archive contains real
  files rather than links to local `/models/...` paths.
- Gemma adapter `checkpoint-*` training directories are intentionally excluded.
  The submit predictor only needs the adapter root files listed above.
- The zip also includes `artifact_manifest.txt` and
  `artifact_manifest_details.tsv` for auditing the packaged paths and sizes.

## scripts/data/dowanload_data_model.sh 介紹

  現在流程是：

  1. 用 gdown --folder 下載 Google Drive folder 到 cache。
  2. 如果找到 *.zip.parts.txt，會讀 manifest。
  3. 驗證 part_count、每片大小、總大小。
  4. 依序把 aicup_esg2026_submit_artifacts.zip.part-* 重組成
     zip。
  5. 驗證重組後是有效 zip。
  6. 解壓並安裝到 data/、models/。
  7. 如果 data/ 或 models/ 已存在，仍會直接 error stop。

  也更新了 README.md，說明現在會 rebuild split zip shards。