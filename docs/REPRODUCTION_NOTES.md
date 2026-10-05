# Reproduction notes

The [README](../README.md) contains the installation and execution sequence.
This page collects checks, historical settings and external-artifact limits.

## CPU code checks

From the repository root, with the full project environment active:

```bash
python scripts/check_repository.py
python tests/test_descriptor_assignment.py
python tests/test_model_components.py
python tests/test_p8_control_invariants.py
python tests/test_final_architecture_loads.py
python tests/test_release_workflow.py
python tests/test_command_entrypoints.py
```

These checks run on local CPU without Slurm. They exercise assignment,
ConLoss, molecule-level routing, checkpoint structure and command generation.
They do not reproduce pretraining or finetuning.

## Reported results and new runs

[The final matrix](../configs/final_model_matrix.tsv) contains 16 neural
configurations. Each runs on eight datasets with seeds 42 to 46. Fold labels
1 to 5 identify repeated scaffold splits, not disjoint cross-validation folds.
The `paper_label` column maps manuscript names to unchanged `model` and
`result_exp` identifiers.

The statistics-only command reads [results/corrected](../results/corrected/)
and checks the rebuilt summaries against the supplied tables. It cannot
independently verify sample coverage because per-molecule predictions are not
distributed. The accompanying coverage table records the original checks.

For a new run, preserve validation predictions, sample manifests and selected
checkpoints. `freeze_panel.py` checks all 640 validation outputs and checkpoint
paths before writing the run-specific frozen definition. It does not select
configurations by test performance. A single-model run cannot pass this full
panel check.

## Data and weights

Datasets, processed caches, checkpoints, raw predictions, embeddings, images
and logs are excluded from this repository. See [data formats](../data/README.md)
and [checkpoint locations](../checkpoints/README.md).

Exact numerical reproduction requires the original processed row order,
tokenizer, corruption masks and pretraining cache. Obtain these resources from
the authors under their applicable terms. A separately filtered MoleculeNet
download is not a replacement for the original processed files. This release
does not advertise a public download for those exact artifacts.

The 240 downstream checkpoints for concatenation, auxiliary prediction and
quantile controls were not retained in the research directory. Their recorded
predictions and lightweight results remain, but repeating model inference
requires new finetuning. No model weights are included in this repository.

For existing pretrained weights, `--pretrain-root` points to the directory
containing `<model>/compt/periodic_latest.pth`. Downstream transfer uses the
completed graph checkpoint at zero-based epoch 299. Old incomplete-batch
validation checkpoints cannot stand in for corrected final-test checkpoints.

## Pretraining preparation and restart

Fixed TPSA and MolLogP boundaries are predefined. Descriptor normalization,
quantile boundaries and stable-random score thresholds are fitted on ZINC15
and reused downstream. The stable-random rule uses canonical molecular keys
rather than independently shuffling each downstream dataset.
[Reported descriptor statistics](../configs/reported_descriptor_statistics.json)
are supplied for comparison. Do not refit thresholds using downstream labels
or validation performance.

The original preparation code also exports legacy and sparse-bin arrays.
These additional arrays are not part of the final 16-model matrix.

The legacy restart implementation restores component weights but not optimizer
state or all decoder states. It does not exactly restore the training trajectory.
The training wrapper does not download weights or manage a background watcher.

## Historical SI screens

[Historical results](../results/historical/README.md) contain layer-placement
and single-branch KAN screens. Their configuration switches are documented in
[Experiment parameters](../EXPERIMENT_PARAMETERS.md), and their identifiers
are in [the historical matrix](../configs/model_matrix.tsv).

Keep these results separate from the corrected panel. Their downstream depth
and incomplete-batch evaluation differ from the final protocol. Evaluating
those architectures with the current runner is a new evaluation, not a
reproduction of the historical numerical tables. The final controls use
cross-stage stable-random assignment and molecule-level routing, not the old
stage-specific random assignment or token-level router.
