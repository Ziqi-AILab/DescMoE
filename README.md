# DescMoE

Code for **DescMoE: Physicochemical Descriptor-Guided Mixture-of-Experts
Pretraining for Molecular Property Prediction**.

DescMoE uses molecular descriptors to assign molecules to computation branches
during molecular pretraining and downstream property prediction.

## Overview

DescMoE partitions molecules by topological polar surface area (TPSA) or
Wildman-Crippen MolLogP. Each descriptor is studied in a separate configuration.
Fixed descriptor intervals select one of eight multilayer perceptron (MLP)
branches in the final graph Transformer layer. An optional descriptor-aware
contrastive loss (ConLoss) groups representations from the same region during
pretraining.

## Architecture

```text
Molecular graph -> Shared graph layers 1 to 7 -> Graph layer 8 -> Prediction
                                               Attention
                                               Descriptor-selected MLP
```

| Component | Configuration |
| --- | --- |
| Graph encoder | 8 Transformer layers, hidden size 256, 8 attention heads |
| DescMoE | Replaces only the layer-8 FFN with 8 MLP branches |
| Assignment | One branch for all valid nodes of a molecule |
| ConLoss | Pretraining only, weight 0.1, temperature 0.1 |
| Downstream transfer | Finetune the full graph encoder and task head |

Dense Base, concatenation, auxiliary prediction, quantile assignment, stable
random assignment and learned routing are comparison models. Their settings
are listed in [Experiment parameters](EXPERIMENT_PARAMETERS.md).

## Installation

```bash
git clone https://github.com/Ziqi-AILab/DescMoE.git
cd DescMoE
conda env create -f environment.yml
conda activate MoleSG
```

The environment specifies Python 3.10 and RDKit 2022.09.5. GPU execution requires
a compatible CUDA driver. GPU extension packages may need platform-specific
installation. For statistics only, use the smaller installation below.

## Data Preparation

Datasets, processed caches and weights are not included. Exact training
reproduction requires the original processed row order and ZINC15 cache from
the authors. See [data formats](data/README.md) and
[checkpoint locations](checkpoints/README.md). Set the paths to your external
files and choose an empty output directory.

```bash
export ZINC_CSV=/path/to/zinc15_250K.csv
export ZINC_CACHE=/path/to/zinc15_0.25_geo/preprocess
export DOWNSTREAM_DATA=/path/to/Downstream/Data
export RUN=/path/to/descmoe_run

python scripts/prepare_matched_control_assignments.py \
  --zinc-csv "$ZINC_CSV" --downstream-data "$DOWNSTREAM_DATA" \
  --output-dir "$RUN/control_data" \
  --zinc-output "$RUN/control_data/descriptor_controls.npz"
```

Preparation runs on local CPU. Fixed intervals remain predefined. Descriptor
statistics, quantile boundaries and stable-random thresholds are fitted on
ZINC15 and then reused downstream.

## Training and Evaluation

The examples use TPSA DescMoE (`p3_tpsa_mofe_last_ffn`). Select another `model`
from [the final configuration table](configs/final_model_matrix.tsv) for
MolLogP, ConLoss or a comparison model. `paper_label` gives the manuscript name.

**`run_model.py` prints commands by default.** Add `--execute` only inside an
allocated Slurm GPU job. The wrapper does not submit jobs. Use one GPU and your
cluster's partition, QoS and time settings. Data preparation, freezing and
statistics run on local CPU, without Slurm.

### 1. Pretraining

```bash
python scripts/run_model.py --phase pretrain \
  --model p3_tpsa_mofe_last_ffn --zinc-cache "$ZINC_CACHE" --run-root "$RUN"
```

Pretraining uses ZINC15 250K, 300 epochs, batch size 32 and seed 42. To reuse
completed weights during finetuning, pass `--pretrain-root /path/to/Model`.

### 2. Validation-based finetuning

```bash
python scripts/run_model.py --phase finetune \
  --model p3_tpsa_mofe_last_ffn --dataset bbbp \
  --downstream-data "$DOWNSTREAM_DATA" --run-root "$RUN"
```

Each executed job runs five repeated balanced scaffold splits with seeds 42
to 46. Complete validation ROC-AUC selects the checkpoint with patience 20.
Finetuning does not evaluate test data. Validation and test loaders use
`shuffle=False, drop_last=False`.

### 3. Freeze the panel and evaluate test

First complete finetuning for **all 16 configurations and all eight datasets**
(`bbbp`, `tox21`, `toxcast`, `sider`, `clintox`, `bace`, `hiv`, `muv`). The freeze
command requires all **640 validation outputs and corresponding checkpoints**.
The single BBBP example above is not sufficient.

```bash
python scripts/freeze_panel.py --run-root "$RUN"
python scripts/run_model.py --phase final_test \
  --model p3_tpsa_mofe_last_ffn --dataset bbbp \
  --downstream-data "$DOWNSTREAM_DATA" --run-root "$RUN"
```

After freezing, execute final test for every model-dataset combination in GPU
jobs. Each uses its validation-selected checkpoint and the frozen definition.
The wrapper refuses to overwrite existing test outputs.

## Results Reproduction

### Rebuild the reported statistics without a GPU

```bash
python -m pip install -r requirements-analysis.txt
python scripts/summarize_reported_results.py --check
```

The included tables contain 640 neural runs and 200 logistic-regression
reference runs. This command rebuilds dataset means, sample SDs, macro ROC-AUC,
paired contrasts and 10,000-draw bootstrap intervals in `runs/rebuilt_tables/`.
No molecular data or model weights are needed. See [included results](results/corrected/README.md).

### Run CPU references and summarize new experiments

```bash
python scripts/run_cpu_descriptor_ecfp_baselines.py \
  --data-root "$DOWNSTREAM_DATA" --output-root "$RUN/cpu_baselines"
python analysis/build_p0_corrected_panel.py \
  --lock "$RUN/model_definition_frozen.json" --result-root "$RUN/results" \
  --cpu-folds "$RUN/cpu_baselines/cpu_baseline_fold_auc.csv" \
  --output-dir "$RUN/summary"
```

CPU references use the same scaffold splits and validation-only regularization
selection. Run aggregation after all neural test outputs are complete.

## Project Structure

```text
MoleSG/     Pretraining, graph encoder and downstream implementation
configs/    Model configurations and descriptor statistics
scripts/    Data preparation, training wrappers and result checks
analysis/   Prediction aggregation and paired statistics
results/    Lightweight final tables and separate historical SI results
tests/      CPU component and workflow checks
docs/       Method mapping and reproduction notes
```

See [parameter switches](EXPERIMENT_PARAMETERS.md),
[method-to-code mapping](docs/METHOD_CODE_MAP.md) and
[reproduction notes](docs/REPRODUCTION_NOTES.md) for tests, historical screens
and checkpoint limitations.

## Acknowledgments

This implementation builds on MoleSG's graph and SMILES pretraining framework.
The upstream MoleSG names are retained in the source tree.
