# Methods and implementation

All paths are relative to this repository. Public CLI names containing `expert`
are retained for checkpoint compatibility. Manuscript prose uses `branch`.

| Method | Implementation | Configuration or detail |
| --- | --- | --- |
| ZINC graph/SMILES corruption | `MoleSG/Data_process/data_geometirc_maeratio.py`, `load_data_from_mol_mask` | Stored NPZ cache, same source rows |
| Eight-layer graph encoder | `MoleSG/pretrain/transformer_graph.py`, `make_model`, `EncoderLayer` | N=8, width=256, heads=8, ScaleNorm |
| Fixed region assignment | `MoleSG/pretrain/prior_moe.py`, `assign_expert_ids`, `PriorMoEFFN.forward` | Threshold equality enters the right region |
| Shared descriptor concatenation | `transformer_graph.py`, `DescriptorConditionedFeedForward` | Only layer 8, 257 to 256 to 256, Mish after both transformations |
| Auxiliary descriptor prediction | `transformer_graph.py`, `GraphTransformer.forward`; `train_total.py`, `model_train` | Masked pooling, linear head, standardized target, 0.1 MSE |
| Descriptor mean/scale and quantiles | `scripts/prepare_matched_control_assignments.py`, `prepare_zinc`, `quantile_edges` | ZINC population SD, NumPy linear quantiles |
| Stable random molecule rule | `MoleSG/Data_process/stable_random_assignment.py` | Canonical SMILES bytes, axis code, seed 42, SeedSequence/PCG64 |
| Random occupancy thresholds | Same module, `occupancy_matched_score_edges` | Fit on ZINC only, keep identical keys together |
| Molecule learned router | `MoleSG/pretrain/standard_moe.py`, `StandardMoEFFN` | Post-attention masked pooling, Linear(256,8), top1 probability retained |
| ConLoss | `MoleSG/pretrain/prior_moe.py`, `expert_contrastive_loss`; `train_total.py`, `model_train` | Before graph/SMILES fusion, coefficient/temperature 0.1 |
| Corrected training and validation | `MoleSG/Downstream/train_graph_evalfix.py`, `train_one_fold`, `evaluate_model` | Patience 20, full deterministic validation |
| Repeated balanced scaffold splits | `MoleSG/Downstream/utils.py`, `scaffold_split` | Seeds 42 to 46, include_chirality=False |
| Pretrained weight loading | `MoleSG/Downstream/train_graph.py`, `_load_pretrained_checkpoint` | N8 strict loading, no discarded encoder layers |
| Sample row identity | `MoleSG/Downstream/dataset_graph.py`; `train_graph_evalfix.py`, `verify_coverage` | `<dataset>:<source_row_index>`, not deduplicated structures |
| Final-test separation | `scripts/freeze_panel.py`; `train_graph_evalfix.py`, `require_frozen_lock` | Complete validation first, no test-based model selection |
| Missing-label ROC-AUC | `analysis/build_evalfix_canonical.py`, `roc_auc` | Observed labels, both classes, endpoint mean |
| Paired contrasts and bootstrap | `analysis/build_p0_corrected_panel.py`, `paired_contrasts`; `build_evalfix_canonical.py`, `hierarchical_ci` | Dataset then seed resampling, 10,000 draws, seed 42 |
| CPU references | `scripts/run_cpu_descriptor_ecfp_baselines.py` | ECFP4 radius 2, 2048 bits, validation-selected logistic regression |

## Details that affect the mathematical description

- A branch MLP is linear, Mish, linear. The original shared dense FFN and the
  concatenation FFN also apply the configured output Mish. Their output
  nonlinearities should not be described as identical.
- The graph mask is computed from nonzero input node features. During
  pretraining, graph-masked atoms have zero features. They are excluded from
  ConLoss and auxiliary pooling as well as padding. Downstream graph inputs
  are not graph-masked.
- ConLoss pools graph encoder output before fusion with SMILES. Positive
  labels are the same IDs used by the conditional branches. Its denominator
  contains all nonself molecules. Only anchors with positives are averaged.
  The 4096-row subsampling option is inactive at batch size 32.
- Learned routing pools the final layer's FFN input, not the final graph
  output. Its load loss is `0.01 * 8 * sum(assignment_fraction * mean_probability)`.
  That training loss is used in pretraining and downstream training.
- ConLoss and descriptor MSE are pretraining losses only. Descriptor input
  concatenation and the branch assignment rules persist downstream.
- Region masks are broadcast over padded tensors. One branch is selected per
  molecule, but the implementation does not promise to skip every padded
  tensor operation.
- Quantile edges and stable-random score thresholds remain fixed downstream.
  Stable-random occupancy matching refers to the pretraining distribution,
  not to a guarantee of matching downstream region counts.

## Historical material

Layer and KAN configuration switches are documented in
`EXPERIMENT_PARAMETERS.md`. They do not change the declared 16-model panel.
The older `tables_v2/` metrics and manuscript snapshot remain historical.
