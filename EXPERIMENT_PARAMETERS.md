# Experiment Parameter Guide

This document lists the parameter changes used for Layer Select, KAN Select,
and the final ablation. It is intended as a compact reproduction guide. The
full model matrix is available in `configs/model_matrix.tsv`.

## 1. Common setup

All descriptor-conditioned pretraining experiments use the same graph
Transformer backbone and optimization setup. Only the flags listed in the
stage-specific sections are changed.

```text
Pretraining dataset        nmrshiftdb configuration with ZINC15 inputs
Pretraining epochs         300
Pretraining seed           42
Graph Transformer layers   8
Attention heads            8
Hidden dimension           256
Dense layers per FFN       2
Descriptor branches        8
TPSA boundaries            20, 40, 60, 80, 100, 120, 140 A^2
MolLogP boundaries         -1, 0, 1, 2, 3, 4, 5
KAN grid size              5
KAN spline order           3
```

The fixed boundaries produce regions 0 to 7. Intervals are left-closed and
right-open. Values below the first boundary enter region 0. Values at or above
the final boundary enter region 7.

The common descriptor-conditioned flags are:

```bash
--use_prior_moe \
--num_experts 8 \
--expert_property <tpsa-or-logp> \
--moe_layer_mode <mode>
```

Following MoE terminology in the implementation, the computation branches are
named experts in command-line flags and source identifiers.

## 2. Layer Select

### Question

Where should descriptor-conditioned computation be introduced in the
eight-layer graph Transformer?

### Parameters changed

Layer Select uses MolLogP branches with MLP functions. The only architectural
parameter changed across the three runs is `--moe_layer_mode`.

| Run | Flag | One-based Transformer layers |
| --- | --- | --- |
| Last | `--moe_layer_mode last` | 8 |
| Odd | `--moe_layer_mode odd` | 1, 3, 5, 7 |
| Even | `--moe_layer_mode even` | 2, 4, 6, 8 |

The remaining branch flags are fixed:

```bash
--use_prior_moe \
--num_experts 8 \
--expert_property logp \
--expert_type ffn
```

Example:

```bash
cd MoleSG/pretrain

python train_total.py \
  --gpu 0 \
  --seed 42 \
  --epochs 300 \
  --experiment_name p2_mofe_ffn_logp_last \
  --use_prior_moe \
  --num_experts 8 \
  --expert_property logp \
  --expert_type ffn \
  --moe_layer_mode last
```

Replace the experiment name and layer mode with `odd` or `even` for the other
two runs. The historical screening result selected `last`, so all subsequent
KAN and final-ablation models use only layer 8. Test summaries were visible
during this exploratory design stage.

## 3. KAN Select

### Question

Do different TPSA-defined or MolLogP-defined regions benefit from different
nonlinear branch functions?

### Parameters changed

KAN Select starts from the all-MLP MoFE-last configuration. One branch at a
time is replaced by KAN:

```bash
--expert_type mixed \
--moe_layer_mode last \
--kan_expert_indices <x>
```

where `<x>` is independently set to `0`, `1`, ..., `7`. TPSA and MolLogP are
screened separately, giving 16 pretraining models in total.

Example for TPSA branch 3:

```bash
cd MoleSG/pretrain

python train_total.py \
  --gpu 0 \
  --seed 42 \
  --epochs 300 \
  --experiment_name p4_tpsa_mofe_last_k3 \
  --use_prior_moe \
  --num_experts 8 \
  --expert_property tpsa \
  --expert_type mixed \
  --moe_layer_mode last \
  --kan_expert_indices 3
```

For MolLogP, change:

```text
--experiment_name p4_logp_mofe_last_k3
--expert_property logp
```

Each kx model was compared with the matched all-MLP MoFE-last baseline for the
same descriptor during exploratory development. The historical comparison
used the equal-weight mean ROC-AUC across the eight classification datasets. A
branch was assigned KAN when its kx macro ROC-AUC exceeded the matched all-MLP
value. These test-derived choices document model construction and are not
independent final test evidence.

The resulting assignments are:

| Descriptor | KAN branches | MLP branches |
| --- | --- | --- |
| TPSA | 0, 1, 2, 3, 5 | 4, 6, 7 |
| MolLogP | 1, 2, 3, 4 | 0, 5, 6, 7 |

The screening configurations and checkpoint names are listed under
`phase=kan_select` in `configs/model_matrix.tsv`.

## 4. Final ablation

The final ablation keeps `--moe_layer_mode last` fixed. TPSA and MolLogP are
evaluated independently.

| Role | Branch assignment | Branch function | ConLoss |
| --- | --- | --- | --- |
| Base | None | Vanilla FFN | No |
| I1 | TPSA or MolLogP | All MLP | No |
| I1+I2 | TPSA or MolLogP | Selected KAN | No |
| I1+I3 | TPSA or MolLogP | All MLP | Yes |
| Full | TPSA or MolLogP | Selected KAN | Yes |
| all-KAN control | TPSA or MolLogP | All KAN | No |

### Base

Do not add descriptor-conditioned flags:

```bash
python train_total.py \
  --gpu 0 \
  --seed 42 \
  --epochs 300 \
  --experiment_name p3_base_vanilla
```

### I1 with all-MLP branches

```bash
--use_prior_moe \
--num_experts 8 \
--expert_property <tpsa-or-logp> \
--expert_type ffn \
--moe_layer_mode last
```

### I1+I2 with selected KAN branches

TPSA:

```bash
--use_prior_moe \
--num_experts 8 \
--expert_property tpsa \
--expert_type mixed \
--moe_layer_mode last \
--kan_expert_indices 0,1,2,3,5
```

MolLogP:

```bash
--use_prior_moe \
--num_experts 8 \
--expert_property logp \
--expert_type mixed \
--moe_layer_mode last \
--kan_expert_indices 1,2,3,4
```

### I1+I3 with ConLoss

Start from the I1 flags and add:

```bash
--use_contrastive_expert_loss \
--contrastive_coff 0.1 \
--contrastive_temperature 0.1
```

### Full model

Start from the descriptor-specific I1+I2 flags and add:

```bash
--use_contrastive_expert_loss \
--contrastive_coff 0.1 \
--contrastive_temperature 0.1
```

### all-KAN control

Use:

```bash
--expert_type kan
```

No `--kan_expert_indices` value is required because all eight branches use
KAN.

## 5. Downstream finetuning

The historical development snapshot used `train_graph.py`. It is retained so
that the included legacy tables remain traceable. Submission-grade reruns use
`train_graph_evalfix.py`, which performs checkpoint selection on complete,
deterministic validation splits and does not construct the test split during
finetuning.

The corrected eight-layer checkpoint-aligned command is:

```bash
cd MoleSG/Downstream

python train_graph_evalfix.py \
  --mode finetune \
  --gpu 0 \
  --seed 42 \
  --fold 5 \
  --dataset <dataset> \
  --pretrained_ckpt_path <checkpoint> \
  --experiment_name <result-name-ending-in-evalfix> \
  --backbone_profile pretrain_aligned \
  --encoder_layers_override 8 \
  --strict_backbone_load \
  <matching-model-flags>
```

`<matching-model-flags>` must exactly reproduce the pretraining architecture.
For example, the TPSA selected-KAN model requires:

```bash
--use_prior_moe \
--num_experts 8 \
--expert_property tpsa \
--expert_type mixed \
--moe_layer_mode last \
--kan_expert_indices 0,1,2,3,5
```

The five scaffold runs use seeds 42, 43, 44, 45, and 46. The datasets are:

```text
bbbp tox21 toxcast sider clintox bace hiv muv
```

Memory-stable settings used for the largest datasets were:

```text
toxcast  --num_workers 0
muv      --num_workers 0
hiv      --num_workers 0 --batch_size_override 12
```

The KAN screening stage retained its historical downstream configuration for
exploratory model selection. The final main-table evaluation uses the strict
eight-layer alignment above. These two protocols are reported separately and
are not combined into one average.

Corrected layer placement and KAN selection use `--mode finetune` and read only
full validation results. After placement and branch assignments are frozen,
final test evaluation uses `--mode final_test` together with
`--require_model_definition_lock` and `--model_definition_lock <file>`. The
test mode loads the corrected fold checkpoints and evaluates each test row
exactly once. Test results are not used for architecture selection.

## 6. Result aggregation

For each dataset, endpoint labels equal to `-1` are masked. ROC-AUC is
calculated for endpoints containing both classes, then averaged within each
fold. The five folds are summarized by their arithmetic mean and sample
standard deviation. The overall macro ROC-AUC gives equal weight to the eight
dataset means.

Metric formulas and code locations are listed in:

```text
tables_v2/metric_definitions.csv
```
