# paper101

Code and lightweight results for **Physicochemical Descriptors as Features,
Auxiliary Targets, and Branch Selectors in Molecular Pretraining**.

The corrected comparison contains 16 neural configurations across eight
classification datasets and five repeated scaffold splits (640 test runs).
Five logistic-regression reference families contribute another 200 runs.
TPSA and MolLogP are studied separately. The configurations include a dense
backbone, descriptor concatenation, auxiliary prediction, fixed and quantile
assignment, cross-stage stable random assignment, molecule-level learned
routing, and ConLoss variants. Historical KAN and layer screens are SI material.

## Check the reported statistics without a GPU

```bash
python -m pip install -r requirements-analysis.txt
python scripts/summarize_reported_results.py --check
```

This rebuilds dataset means, sample SDs, macro ROC-AUC, paired contrasts, and
10,000-iteration hierarchical bootstrap intervals from `results/corrected/`.
Outputs go to `runs/rebuilt_tables/`. It does not download molecular data or
run a model. Included CSV values retain their original precision.

## Environment and external files

```bash
conda env create -f environment.yml
conda activate MoleSG
```

The source uses the MoleSG architecture. The supplied environment records
the project versions, including RDKit 2022.09.5. CUDA training needs a compatible
driver. GPU extension packages may need platform-specific installation.

Datasets, processed caches, checkpoints, raw predictions, embeddings, images,
and logs are **not distributed here**. See [data/README.md](data/README.md) for
required file formats and [checkpoints/README.md](checkpoints/README.md) for
checkpoint paths. Exact numerical training reproduction requires the original
processed row order and pretraining cache, not an independently filtered
MoleculeNet download. Obtain those external artifacts from the authors.

Set your own paths. `RUN` should be an empty directory for a new reproduction.

```bash
export ZINC_CSV=/path/to/zinc15_250K.csv
export ZINC_CACHE=/path/to/zinc15_0.25_geo/preprocess
export DOWNSTREAM_DATA=/path/to/Downstream/Data
export RUN=/path/to/paper101_run
```

## Reproduction sequence

### 1. Prepare assignment arrays on local CPU

```bash
python scripts/prepare_matched_control_assignments.py \
  --zinc-csv "$ZINC_CSV" --downstream-data "$DOWNSTREAM_DATA" \
  --output-dir "$RUN/control_data" \
  --zinc-output "$RUN/control_data/descriptor_controls.npz"
```

Statistics and score thresholds are fitted on ZINC15, then reused downstream.
The canonical-SMILES random rule does not shuffle each downstream dataset.
`configs/reported_descriptor_statistics.json` records the parameters used in
the reported experiment for comparison. Do not refit boundaries on downstream
labels or validation performance. Extra legacy and sparse-bin arrays produced
by the original preparation code are not part of the final 16-model matrix.

### 2. Pretrain each configuration

The single source of model flags is `configs/final_model_matrix.tsv`.
This example prints the command without executing it.

```bash
python scripts/run_model.py --phase pretrain \
  --model p8_tpsa_stable_random_last --zinc-cache "$ZINC_CACHE" --run-root "$RUN"
```

Run the same command with `--execute` **inside your Slurm GPU job**. The wrapper
does not submit jobs or start a watcher. Request one GPU, activate the Conda
environment, and set your cluster's partition, QoS and time in the job script.
Pretraining uses 300 epochs, batch size 32 and seed 42. Existing completed
pretraining checkpoints can instead be supplied with `--pretrain-root`.
No model downloads occur in the training wrapper.

The original resume implementation is retained, but it does not save optimizer
state or all decoder states. It is not exact trajectory restoration. Do not
describe restarting from these component weights as an exact resumed run.

### 3. Finetune using complete validation splits

```bash
python scripts/run_model.py --phase finetune \
  --model p8_tpsa_stable_random_last --dataset bbbp \
  --downstream-data "$DOWNSTREAM_DATA" --run-root "$RUN"
```

Repeat for all 16 configurations and `bbbp,tox21,toxcast,sider,clintox,bace,hiv,muv`.
Each command runs five seeds, 42 to 46, when executed in a GPU job. The runner
uses the eight-layer backbone, complete deterministic validation, patience 20,
and no test evaluation. It saves validation predictions and the best checkpoint.
See [EXPERIMENT_PARAMETERS.md](EXPERIMENT_PARAMETERS.md) for all flags and
the distinction between pretraining and downstream losses.

### 4. Freeze the complete panel, then evaluate test once

```bash
python scripts/freeze_panel.py --run-root "$RUN"
python scripts/run_model.py --phase final_test \
  --model p8_tpsa_stable_random_last --dataset bbbp \
  --downstream-data "$DOWNSTREAM_DATA" --run-root "$RUN"
```

Freezing is a local CPU check of all 640 validation outputs and checkpoint
paths. It does not select models by test performance. Submit each final-test
command as a GPU job with `--execute`. Final test requires the run-specific
frozen definition. Existing test outputs are not overwritten by the wrapper.

### 5. CPU baselines and aggregation

```bash
python scripts/run_cpu_descriptor_ecfp_baselines.py \
  --data-root "$DOWNSTREAM_DATA" --output-root "$RUN/cpu_baselines"
python analysis/build_p0_corrected_panel.py \
  --lock "$RUN/model_definition_frozen.json" --result-root "$RUN/results" \
  --cpu-folds "$RUN/cpu_baselines/cpu_baseline_fold_auc.csv" \
  --output-dir "$RUN/summary"
```

CPU baselines use the same scaffold implementation. Their regularization is
selected using validation only. Aggregation reads completed predictions,
checks sample-ID coverage, and rebuilds the final statistics.

## Code checks and paper correspondence

```bash
python scripts/check_repository.py
python tests/test_descriptor_assignment.py
python tests/test_model_components.py
python tests/test_p8_control_invariants.py
python tests/test_final_architecture_loads.py
python tests/test_release_workflow.py
python tests/test_command_entrypoints.py
```

These are CPU tests, not full training reproductions. The method-to-function
map is in [docs/METHOD_CODE_MAP.md](docs/METHOD_CODE_MAP.md).
The release checks and their limits are recorded in [TEST_REPORT.md](TEST_REPORT.md).

`tables_v2/`, `configs/model_matrix.tsv`, the old manuscript snapshot and the
older plotting scripts are **legacy development material**, not the corrected
main benchmark. They remain available for historical SI context. Use
`results/corrected/` and `configs/final_model_matrix.tsv` for the current panel.
The code package contains no figure files. The methods revision is delivered
separately and does not silently replace the old manuscript snapshot.
