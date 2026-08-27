# Descriptor-Guided Chemical Space Partitioning

This repository accompanies the JCIM manuscript on descriptor-guided chemical
space partitioning for molecular representation learning. The implementation
uses fixed intervals of topological polar surface area (TPSA) or
Wildman-Crippen MolLogP to assign each molecule to one computation branch in
the final layer of an eight-layer graph Transformer. The branch functions can
be multilayer perceptrons (MLPs), Kolmogorov-Arnold networks (KANs), or a
descriptor-specific mixture of both. A descriptor-aware contrastive loss
(ConLoss) is available during pretraining.

The repository contains the method implementation, experiment configurations,
the full-coverage downstream evaluation entry point, canonical tables used by
the development manuscript, and deterministic plotting scripts. Molecular
datasets, model checkpoints, raw predictions, node embeddings, and scheduler
logs are intentionally excluded because of their size or redistribution terms.

## Repository structure

```text
MoleSG/
  Data_process/                 ZINC15 graph preprocessing
  pretrain/                     pretraining and branch implementations
  Downstream/                   legacy and full-coverage finetuning entry points
analysis/                       canonical aggregation and plotting scripts
configs/                        model matrix and downstream task definitions
tables_v2/                      canonical development-snapshot tables
data/                           external data instructions
checkpoints/                    external checkpoint layout
raw_results/                    external raw-result layout
EXPERIMENT_PARAMETERS.md        exact Layer Select, KAN Select, and ablation flags
paper_v3.0_all_in_one.tex       manuscript and Supporting Information snapshot
```

## Evidence status

The included canonical downstream tables reproduce the historical development
snapshot. All 440 planned eight-layer fold files were present, but the original
validation and test loaders used `drop_last=True`. The historical values are
therefore retained as development evidence rather than a corrected final
benchmark. Layer placement and KAN allocation were also exploratory design
steps in which test summaries were visible.

`MoleSG/Downstream/train_graph_evalfix.py` implements the corrected protocol.
Validation and test loaders are deterministic and use `drop_last=False`.
Checkpoint selection uses the complete validation split. Final test evaluation
is a separate mode and requires a frozen model-definition file. Corrected
results are not included until that evaluation is complete.

## Environment

Create the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate MoleSG
```

The manuscript calculations use RDKit 2022.09.5. The environment also pins the
PyTorch, PyTorch Geometric, NumPy, SciPy, scikit-learn, Transformers, and
tokenizers versions used by the project.

## Reproduce the included figures

The publication data figures can be regenerated from the included canonical
CSV files without molecular datasets, checkpoints, or a GPU:

```bash
python analysis/plot_figures_v2.py
```

Outputs are written to `figures/`. Figure files are not tracked in this public
repository. The canonical tables document the legacy development snapshot
described above. The optional overview panel is skipped when its visual assets
are absent. All quantitative result figures are still generated.

The chemistry example in the overview figure can be independently checked with
the same descriptor-assignment function used by the model:

```bash
python analysis/build_verified_overview_chemistry.py
```

This calculation verifies ibuprofen using RDKit 2022.09.5. Its TPSA is
37.30 square angstroms and its MolLogP is 3.0732. Under the fixed project
boundaries, these values enter zero-based TPSA region 1 and MolLogP region 5.

## External data

The repository does not redistribute ZINC15 or MoleculeNet data. After obtaining
the datasets under their original licenses, use this layout:

```text
MoleSG/Data/zinc15/zinc15_250K.csv
MoleSG/Data/zinc15/expert_ids.npz
MoleSG/Data/zinc15/zinc15_0.25_geo/preprocess/
MoleSG/Downstream/Data/<dataset>/preprocess/<dataset>.pickle
```

The eight classification datasets are BBBP, Tox21, ToxCast, SIDER, ClinTox,
BACE, HIV, and MUV. Details are in `data/README.md` and
`configs/downstream_tasks.tsv`.

## Descriptor assignment audit

After external ZINC15 files are available, recompute TPSA and MolLogP for every
source row and compare them with the stored branch assignments:

```bash
python analysis/audit_pretraining_descriptor_assignments.py
```

Set `MOLESG_ROOT` if the `MoleSG` directory is stored outside the repository:

```bash
MOLESG_ROOT=/path/to/MoleSG \
python analysis/audit_pretraining_descriptor_assignments.py
```

## Pretraining

All main models use eight graph Transformer layers, eight attention heads, a
hidden dimension of 256, and 300 pretraining epochs. TPSA boundaries are 20,
40, 60, 80, 100, 120, and 140 square angstroms. MolLogP boundaries are -1, 0,
1, 2, 3, 4, and 5.

Example for the final-layer TPSA all-MLP model:

```bash
cd MoleSG/pretrain
python train_total.py \
  --gpu 0 \
  --seed 42 \
  --epochs 300 \
  --experiment_name p3_tpsa_mofe_last_ffn \
  --use_prior_moe \
  --num_experts 8 \
  --expert_property tpsa \
  --expert_type ffn \
  --moe_layer_mode last
```

For the TPSA selected-KAN model, use `--expert_type mixed` and
`--kan_expert_indices 0,1,2,3,5`. For MolLogP, use
`--kan_expert_indices 1,2,3,4`. Add the following flags for ConLoss:

```bash
--use_contrastive_expert_loss \
--contrastive_coff 0.1 \
--contrastive_temperature 0.1
```

All parameter changes and checkpoint names are listed in
`EXPERIMENT_PARAMETERS.md` and `configs/model_matrix.tsv`.

## Corrected downstream finetuning

The corrected entry point separates validation-based checkpoint selection from
final test evaluation. A typical finetuning command is:

```bash
cd MoleSG/Downstream
python train_graph_evalfix.py \
  --mode finetune \
  --dataset bbbp \
  --seed 42 \
  --fold 5 \
  --experiment_name p3_tpsa_mofe_last_ffn_n8_evalfix \
  --pretrained_ckpt_path ../pretrain/Model/p3_tpsa_mofe_last_ffn/compt/periodic_latest.pth \
  --backbone_profile pretrain_aligned \
  --encoder_layers_override 8 \
  --strict_backbone_load \
  --use_prior_moe \
  --num_experts 8 \
  --expert_property tpsa \
  --expert_type ffn \
  --moe_layer_mode last
```

The model-specific flags must match pretraining exactly. Five scaffold runs use
seeds 42 through 46. The corrected runner stores stable sample identifiers and
checks that every validation or test row is evaluated exactly once.

Final test evaluation is intentionally guarded. It should be run only after the
model definition has been frozen and the corrected checkpoint has been selected
using full validation. See `EXPERIMENT_PARAMETERS.md` for the complete protocol.

## Rebuild canonical tables from external raw artifacts

The included tables are sufficient to reproduce the supplied figures. To
rebuild them from raw predictions, embeddings, and pretraining logs, provide the
external paths through environment variables:

```bash
PAPER101_PROJECT_ROOT=/path/to/paper_101 \
MOLESG_ROOT=/path/to/MoleSG \
MOLESG_RESULT_ROOT=/path/to/MoleSG/Downstream/Result \
python analysis/build_analysis_v2.py
```

Embedding analysis uses the same `MOLESG_RESULT_ROOT` variable:

```bash
MOLESG_RESULT_ROOT=/path/to/MoleSG/Downstream/Result \
python analysis/analyze_embeddings_v2.py
```

Raw artifact layouts are documented in `raw_results/README.md` and
`checkpoints/README.md`.

## Lightweight checks

Run the repository checks with:

```bash
python -m compileall -q MoleSG analysis tests
python tests/test_descriptor_assignment.py
python tests/test_model_components.py
python scripts/check_repository.py
```

The checks do not download data or run GPU training.

## Manuscript correspondence

The mapping between model names, ablation roles, descriptors, branch functions,
and checkpoint paths is in `configs/model_matrix.tsv`. Metric definitions are in
`tables_v2/metric_definitions.csv`. The figure and table source mapping is in
`docs/figure_table_manifest.csv`.
