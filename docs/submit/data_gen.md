# Submit Training Data Generation

This note summarizes, at a high level, how the submit training data is built for
Stage 1, Stage 2, and Stage 3. It focuses on data sources, synthesis strategy,
and final label statistics.

## Overview

LLM synthesis model: `qwen3.6-35b-a3b.gguf` with Q4 quantization.

Canonical synthesis scripts from `exp/agent_loop/claude/20260608T152150/loops/loops001`:

```bash
python3 core/service/data/synthesis/synthesize_st1_data_only.py
python3 core/service/data/synthesis/synthesize_st1_a2_promise.py
python3 core/service/data/synthesis/build_synth_st1_a4_pdf.py
python3 core/service/data/synthesis/synthesize_st2_evidence.py
```

These scripts reproduce the agent-loop synthesis methods: A1 data-only
paraphrase, A3 data+promise positive generation, A2 promise-string-derived
positive generation, A4 deterministic PDF/data-derived hard negatives, and
Stage 2 supported/unsupported evidence synthesis.
Synthetic generation outputs are written under `results/data_synthesis/`.
The final submit datasets under `data/synthesis_data/` are selected/frozen
outputs built from these synthesis sources plus real train/validation rows.

| Stage | Target | Final training pool | Main idea |
| --- | --- | ---: | --- |
| Stage 1 | `promise_status` | 2,500 rows | Combine real train data, real validation data, and synthetic positive commitment examples. |
| Stage 2 | `evidence_status` | 2,548 rows | Combine real train data, real validation data, and balanced synthetic evidence examples. |
| Stage 3 | `evidence_quality` | 1,998 rows after filtering | Use real train + validation data for multitask learning, excluding extremely rare `Misleading` rows. |

## Stage 1

Stage 1 predicts whether a report sentence contains a corporate commitment
(`promise_status`). The final training pool is built from three sources:

| Source | Records | `promise_status` distribution |
| --- | ---: | --- |
| Original train data | 1,000 | `Yes=814`, `No=186` |
| Original validation data | 1,000 | `Yes=813`, `No=187` |
| Synthetic commitment examples | 500 | `Yes=500`, `No=0` |

Final pool summary:

| Records | `promise_status` distribution |
| ---: | --- |
| 2,500 | `Yes=2127`, `No=373` |

How synthesis works:

1. Select positive real examples where `promise_status=Yes`.
2. Use `data + promise_string` as the LLM input. `data` provides the report
   context, and `promise_string` marks the promise to preserve.
3. Ask an LLM to rewrite the context into a fluent, self-contained
   commitment-style sentence. The rewritten sentence should preserve the same
   corporate promise while varying the surface form.
4. Use the generated sentence as synthetic `data` and set
   `promise_status=Yes`.
5. Generate a 300-row unique synthetic pool.
6. Oversample the 300-row pool with replacement to obtain 500 synthetic
   positives, then merge them with the original train and validation rows.

Example:

| Field | Text |
| --- | --- |
| Parent `data` | 國巨集團深知人才全球化趨勢，將更著重於跨國人才培育，多數公司藉由海外輸入優秀人才，國巨集團則是希望由本地輸出優秀人力到海外分公司。國巨未來人才發展策略將是透過招募、留才、培育，達到歷練不同功能別職務、跨國及跨文化的管理能力。 |
| Parent `promise_string` | 國巨集團深知人才全球化趨勢，將更著重於跨國人才培育， |
| Synthetic `data` | 國巨集團深知人才全球化趨勢，將更著重於跨國人才培育，未來將透過招募、留才與培育，輸出優秀本地人才至海外分公司，歷練不同功能別職務、跨國及跨文化的管理能力。 |
| Synthetic label | `promise_status=Yes` |

## Stage 2

Stage 2 predicts whether a promised ESG claim has supporting evidence
(`evidence_status`). The final training pool is built from:

| Source | Records | `evidence_status` distribution |
| --- | ---: | --- |
| Original train data | 1,000 | `Yes=677`, `No=137`, `N/A=186` |
| Original validation data | 1,000 | `Yes=668`, `No=145`, `N/A=187` |
| Synthetic evidence examples | 548 | `Yes=274`, `No=274` |

Note: the original train file stores Stage 2 not-applicable rows as empty
string `""`, while the validation file stores them as `N/A`; both mean the row
has `promise_status=No`, so Stage 2 is not applicable.

Final pool summary:

| Records | `evidence_status` distribution | `promise_status` distribution |
| ---: | --- | --- |
| 2,548 | `Yes=1619`, `No=556`, `N/A=373` | `Yes=2175`, `No=373` |

How synthesis works:

### Synthetic Yes Examples

1. Select real rows where `promise_status=Yes` and `evidence_status=Yes`.
2. Use `promise_string + evidence_string` as the generation input.
3. Generate a new sentence that states the promise and includes the supporting
   action or evidence.
4. Set the synthetic label to `evidence_status=Yes`.

Example:

| Field | Text |
| --- | --- |
| Parent `data` | 為達成科學基礎減量目標倡議組織 (SBTi) 訂定的 2030 年短期及 2050 年長期減碳目標，範疇三的減量是最具影響力也最難推動的部分，因此台達透過「在地化管理」、「永續原物料採購」及「價值鏈碳足跡減量」三種執行作法，達成綠色低碳化供應鏈的落實。 |
| Parent promise | 為達成科學基礎減量目標倡議組織 (SBTi) 訂定的 2030 年短期及 2050 年長期減碳目標， |
| Parent evidence | 台達透過「在地化管理」、「永續原物料採購」及「價值鏈碳足跡減量」三種執行作法， |
| Synthetic `data` | 為達成減碳目標，公司已規劃並執行以下措施：台達透過「在地化管理」、「永續原物料採購」及「價值鏈碳足跡減量」三種執行作法。 |
| Synthetic label | `evidence_status=Yes` |

### Synthetic No Examples

The synthetic `No` rows are generated in two ways:

1. Remove evidence: start from a supported row, remove the concrete evidence,
   and keep only a generic promise-like statement.
2. Rewrite: start from an original no-evidence row and lightly rewrite it while
   preserving its unsupported nature.
3. Set the synthetic label to `evidence_status=No`.

The synthetic `No` split is:

```text
remove evidence = 233 rows (85.0% of synthetic No)
rewrite         =  41 rows (15.0% of synthetic No)
```

Example: evidence removed from an originally supported row.

| Field | Text |
| --- | --- |
| Parent `data` | 短期 (1-3 年)：為掌握數位金融商機並推行低碳服務，本行積極發展行動 APP 及網路金融服務，衍生軟體採購、專利費用及製卡費等支出，並計入當期的其他業務及管理費用，導致營業活動現金流出增加，影響短期財務績效。 |
| Parent promise | ，未來將持續關注並動態調整應對方案。 |
| Parent evidence | 本行積極發展行動 APP 及網路金融服務，衍生軟體採購、專利費用及製卡費等支出， |
| Synthetic `data` | 公司高度關注公司治理議題，期望逐步改善並回應利害關係人期待。 |
| Synthetic label | `evidence_status=No` |

Example: original no-evidence row.

| Field | Text |
| --- | --- |
| Parent `data` | 藥華醫藥以病友為中心，致力增進醫護人員疾病知識，強化醫病關係以協助患者適切使用藥物，並透過產業培力幫助更多病患使用高品質藥物。 |
| Parent promise | 藥華醫藥以病友為中心，致力增進醫護人員疾病知識，強化醫病關係以協助患者適切使用藥物，並透過產業培力幫助更多病患使用高品質藥物。 |
| Parent evidence | None |
| Synthetic `data` | 整體而言，藥華醫藥以病友為中心，致力增進醫護人員疾病知識，強化醫病關係以協助患者適切使用藥物，並透過產業培力幫助更多病患使用高品質藥物。 |
| Synthetic label | `evidence_status=No` |

The final synthetic rows are balanced across the Stage 2 target:

```text
synthetic evidence_status:
Yes = 274
No  = 274
```

The synthetic evidence examples are then merged with the original train and
validation rows to form the final Stage 2 training pool.

## Stage 3

Stage 3 predicts evidence quality. Unlike Stage 1 and Stage 2, it does not add
synthetic rows. It uses the original train and validation data as the multitask
training pool:

| Source | Records | `evidence_quality` distribution |
| --- | ---: | --- |
| Original train data | 1,000 | `Clear=552`, `Not Clear=124`, `Misleading=1`, `""=323` |
| Original validation data | 1,000 | `Clear=566`, `Not Clear=101`, `Misleading=1`, `N/A=332` |

Before training, the two `Misleading` rows are removed because the class is too
rare to support a stable split. The remaining 1,998 rows are used for multitask
BERT training with:

- main target: `evidence_quality`
- auxiliary targets: `promise_status`, `evidence_status`

Final Stage 3 source distribution before filtering:

```text
evidence_quality:
Clear      = 1118
Not Clear  =  225
N/A        =  332
""         =  323
Misleading =    2
```

Final Stage 3 train/validation split after filtering:

| Split | Records | `evidence_quality` distribution |
| --- | ---: | --- |
| Train | 1,598 | `Clear=894`, `Not Clear=180`, `N/A=266`, `""=258` |
| Val | 400 | `Clear=224`, `Not Clear=45`, `N/A=66`, `""=65` |
