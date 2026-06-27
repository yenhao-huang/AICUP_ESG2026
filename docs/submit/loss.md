# Submit Loss Functions

This note records the loss functions used by the current Stage 1, Stage 2, and
Stage 3 submit training recipes.

## Stage 1

- Training entrypoint: `scripts/train/train_ensemble_models_for_stage1.sh`
- Trainer: `core/service/train/train_bert.py`
- Task label: `promise_status` (`No=0`, `Yes=1`)
- Submit recipe loss: focal loss
- Parameters:
  - `LOSS=focal`
  - `CLASS_WEIGHTS=4.0,1.0`
  - `FOCAL_GAMMA=3.0`
  - `LOSS_TAG=focal_g3_w4`

Formula:

$$
\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

Variables:

- `p_t`: model probability assigned to the true class.
- `alpha_t`: class weight for the true class.
- `gamma`: hard-example focusing factor; larger values reduce the loss from
  easy samples more aggressively, so hard examples take a larger share of the
  update.

For the submit recipe, `alpha = [4.0, 1.0]` for `[No, Yes]` and
`gamma = 3.0`.

Design pattern:

Stage 1 is an imbalanced binary classification task: `No` examples are much
rarer than `Yes` examples, but predicting them correctly is important for the
downstream cascade. Focal loss is used through two mechanisms:

- Rare-class weighting: `alpha=[4.0,1.0]` makes `No` samples contribute about
  four times as much as `Yes` samples before the focal term is applied.
- Hard-example focusing: `(1 - p_t)^gamma` downweights easy samples that the
  model already predicts with high confidence. With `gamma=3.0`, the loss
  focuses more strongly on examples with low true-class probability, which are
  usually the boundary or confusing cases.

In practice, this avoids letting the majority `Yes` class dominate the gradient
and pushes the ensemble to pay more attention to rare or difficult `No`
instances.

## Stage 2

- Training entrypoint: `scripts/train/train_ensemble_models_for_stage2.sh`
- Trainer: `core/service/train/train_bert.py`
- Task label: `evidence_status` (`No=0`, `Yes=1`)
- Submit recipe loss: plain cross entropy
- Parameters:
  - `LOSS=ce`
  - `LOSS_TAG=ce_e5`

Formula:

$$
\mathrm{CE}(p_t) = -\log(p_t)
$$

Variables:

- `p_t`: model probability assigned to the true class after softmax.

No explicit class weights are passed by the current Stage 2 submit wrapper.

Design pattern:

Stage 2 uses plain cross entropy based on empirical study. We tried `ce`,
`wce`, focal loss, and ASL loss; `ce` gave the best result for the Stage 2
submit ensemble, so the final recipe keeps the simplest cross-entropy objective.

## Stage 3

- Training entrypoint: `scripts/train/train_multitaskbert_for_stage3.sh`
- Trainer: `core/service/train/train_multitaskbert_stage3.py`
- Model: multitask BERT with `st1`, `st2`, and `st3` heads
- Stage 3 task label: `evidence_quality`
- Submit recipe loss: task-weighted sum of weighted cross entropy losses
- Parameters:
  - `ST1_LOSS=weighted_ce`
  - `ST2_LOSS=weighted_ce`
  - `ST3_LOSS=weighted_ce`
  - `ST3_CLASS_WEIGHTS=1,8,30`
  - `TASK_WEIGHTS=0.2,0.3,0.5`

Overall formula:

$$
\mathcal{L} = w_{st1}\mathcal{L}_{st1} + w_{st2}\mathcal{L}_{st2} + w_{st3}\mathcal{L}_{st3}
$$

Per-task loss:

$$
\mathcal{L}_{stk}
= \frac{1}{|\mathcal{B}_{stk}|}
  \sum_{i \in \mathcal{B}_{stk}}
  w^{(stk)}_{y_i}
  \left[-\log p^{(stk)}_{i,y_i}\right]
$$

Variables:

- `L_st1`, `L_st2`, `L_st3`: head losses for `promise_status`,
  `evidence_status`, and `evidence_quality`.
- `w_st1`, `w_st2`, `w_st3`: task weights that control each head's contribution
  to the total loss.
- `stk`: one of `st1`, `st2`, or `st3`.
- `B_stk`: applicable rows for task `stk` in the batch. Rows masked by
  `ignore_index` are excluded from this set.
- `y_i`: gold label for row `i` under task `stk`.
- `p^{(stk)}_{i,y_i}`: softmax probability assigned by the `stk` head to the
  gold label of row `i`.
- `w^{(stk)}_{y_i}`: class weight for the gold label under task `stk`.

Implementation detail:

Each `L_st*` is implemented as
`CrossEntropyLoss(weight=class_weights, ignore_index=ignore)`. Rows where a task
is not applicable are masked with `ignore_index`, so they do not contribute
gradient for that task. Stage 3 uses manual class weights `[1, 8, 30]` for its
three evidence-quality classes. Stage 1 and Stage 2 use `weighted_ce`, so their
class weights are computed from the training-label inverse frequency.

Design pattern:

Stage 3 has two additional problems beyond ordinary class imbalance:

- It depends on upstream task structure: evidence quality is meaningful only
  after the promise/evidence gates are satisfied.
- The main Stage 3 decision is `Clear` vs. `Not Clear`; `Misleading` is too rare
  to be treated as a primary loss-design target.

The multitask loss addresses the first problem by training `st1`, `st2`, and
`st3` heads together on a shared encoder. The auxiliary `st1` and `st2`
objectives provide more supervision to the encoder, while the largest task
weight remains on `st3` (`0.5`) because Stage 3 quality is the target checkpoint
selection metric.

The Stage 3 class weights `[1,8,30]` mainly address the `Clear` / `Not Clear`
imbalance. `Misleading` remains in the implementation class space, but it is not
the main optimization target because the class is extremely sparse and unstable
for loss-driven tuning. The masking with `ignore_index` preserves the cascade
semantics: non-applicable rows can still train upstream heads, but they do not
create incorrect Stage 3 gradients.

## Gemma LoRA Fine-Tune

- Training entrypoint: `scripts/train/train_gemma_for_stage12.sh`
- Trainer: `core/service/train/train_gemma4.py`
- Config: `configs/train/gemma4_st12_mix.yml`
- Method: QLoRA SFT with `trl.SFTTrainer`, PEFT LoRA, and 4-bit quantization
- Task target: one JSON object containing `promise_status`, `promise_str`,
  `evidence_status`, and `evidence_str`
- Submit recipe loss: completion-only autoregressive cross entropy

Formula:

$$
\mathcal{L}_{\mathrm{Gemma}}
= -\frac{1}{|\mathcal{T}|}
  \sum_{t \in \mathcal{T}}
  \log P_\theta(y_t \mid x, y_{<t})
$$

Variables:

- `x`: prompt tokens built from the system instruction and raw `data` field.
- `y_t`: target JSON token at decoding step `t`.
- `y_<t`: previous target JSON tokens before step `t`.
- `T`: completion-token positions that belong to the JSON target.
- `P_theta(y_t | x, y_<t)`: Gemma probability assigned to the correct next
  target token.

Implementation detail:

`completion_only_loss=True` masks the prompt tokens, so the prompt is used as
conditioning context but does not contribute to the loss. The loss is computed
only on the generated JSON completion. `promise_string` and `evidence_string`
are used only to build the target JSON and never enter the prompt.

Design pattern:

This fine-tune treats Stage 1 and Stage 2 as a structured generation problem
instead of two independent classifiers. The model reads the report text and
learns to generate the exact JSON output expected by the downstream predictors.
LoRA changes the trainable parameter pattern, not the loss: the base Gemma
weights stay frozen/quantized, and only the low-rank adapter parameters are
updated by the completion-token cross entropy. This keeps the fine-tune smaller
while still teaching the model the task-specific output schema and extraction
behavior.
