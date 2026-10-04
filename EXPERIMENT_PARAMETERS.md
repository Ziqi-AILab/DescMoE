# Experiment parameters

## Corrected main panel

`configs/final_model_matrix.tsv` defines the 16 configurations. `paper_label`
contains the DescMoE manuscript names. The command wrapper reads the original
configuration fields rather than selecting flags from filenames.

| Comparison | Pretraining change | Downstream behavior |
| --- | --- | --- |
| Dense Base | No descriptor input, target, or branch | Dense N8 encoder |
| Concatenation, each axis | `--descriptor_use concat` | Same standardized descriptor input |
| Auxiliary prediction, each axis | `--descriptor_use auxiliary --descriptor_aux_coeff 0.1` | No descriptor regression loss |
| DescMoE, each axis | `--use_prior_moe --expert_type ffn --moe_layer_mode last` | Same fixed thresholds |
| DescMoE + ConLoss, each axis | Above plus `--use_contrastive_expert_loss --contrastive_coff 0.1 --contrastive_temperature 0.1` | No ConLoss |
| Quantile assignment, each axis | `--assignment_scheme quantile`, ZINC-fitted boundaries | Frozen pretraining boundaries |
| Stable random, each axis | `--assignment_scheme stable_random` | Same canonical molecule rule |
| Stable random + ConLoss, each axis | Random IDs also define positives | Same random branch IDs, no ConLoss |
| Learned molecule router, shared | `--use_standard_moe --router_granularity molecule --moe_top_k 1 --moe_aux_loss_coeff 0.01` | Router and balancing loss retained in training |

All conditional models use eight MLP branches, one selected branch per molecule
at layer 8. The first seven layers are shared. Descriptor concatenation is also
at layer 8, but `moe_layer_mode=none` because it uses a shared FFN rather than MoE.
Auxiliary prediction adds a pooled descriptor head, not a conditional FFN.

TPSA boundaries are 20,40,60,80,100,120,140 square angstroms. MolLogP boundaries
are -1,0,1,2,3,4,5. Boundaries belong to the region on their right, with IDs 0 to 7.
Descriptor standardization uses ZINC mean and **population SD** (`ddof=0`).
This differs from **sample SD** (`ddof=1`) used to summarize five downstream runs.

Pretraining uses the existing `nmrshiftdb` option preset name, but the actual
corpus is the ZINC15 250K cache. The backbone is N8, width 256, eight heads,
two dense transformations, ScaleNorm, exponential graph-distance kernel and
dropout zero. Batch size is 32, seed 42, epochs 300, warmup 30 epochs and
learning-rate factor 0.2. See the two `utils.py` files for the scheduler.

Finetuning uses `--backbone_profile pretrain_aligned --encoder_layers_override 8
--strict_backbone_load`. Dataset-specific prediction heads and learning-rate
factors are retained. Maximum epochs are 150, warmup 30 and patience 20.
HIV uses batch size 12. Other batch sizes come from downstream `get_options`.
Train shuffles and drops its final incomplete batch. Validation/test do neither.
Fold labels 1 to 5 mean repeated balanced scaffold splits with seeds 42 to 46,
not disjoint cross-validation folds. Proportions are approximately 80/10/10.

## Historical SI configuration switches

Do not mix these runs with the corrected 16-model panel.

- Layer screening changes `--moe_layer_mode last|odd|even`. With N8, `last`
  means layer 8, `odd` means layers 1,3,5,7 and `even` layers 2,4,6,8.
- Single-KAN screening keeps `last` and sets `--expert_type mixed
  --kan_expert_indices k`, with k from 0 to 7. Other branches remain MLP.
- Historical selected-KAN allocations were TPSA `0,1,2,3,5` and MolLogP `1,2,3,4`.
- All-KAN uses `--expert_type kan`. KAN grid size is 5 and spline order is 3.
- P6 stage-specific random assignment and token routing are legacy controls.
  P8 stable random and molecule routing are the controls in the corrected panel.

The legacy matrix records model identities. Its original downstream depth
defaults and incomplete-batch evaluation differ from the corrected protocol.
Running those architectures through the new full-coverage runner is a new
evaluation, not a claim of reproducing a historical numeric table.
