# External Checkpoints

Pretraining and downstream checkpoints are excluded because of their size.
The expected pretraining layout is:

```text
MoleSG/pretrain/Model/<experiment_name>/compt/periodic_latest.pth
MoleSG/pretrain/Model/<experiment_name>/smiles_encoder/periodic_latest.pth
MoleSG/pretrain/Model/<experiment_name>/total_encoder/periodic_latest.pth
```

Experiment names and changed flags are listed in
`EXPERIMENT_PARAMETERS.md`.

Corrected downstream checkpoints are written outside version control, for
example:

```text
MoleSG/Downstream/Model_evalfix/<dataset>/<experiment>/fold_<fold>/best_model.pth
```

Do not commit checkpoint files. The model matrix records the expected
pretraining checkpoint name for every manuscript configuration.
