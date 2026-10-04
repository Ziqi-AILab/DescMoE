# External Raw Results

Raw prediction CSV files, node-embedding pickles, training histories, and Slurm
logs are excluded. Lightweight fold results and summaries are included in
`results/corrected/`. Historical screening summaries are in `results/historical/`.

The original downstream layout is:

```text
MoleSG/Downstream/Result/<dataset>/
MoleSG/Downstream/Model/<dataset>/<experiment>/
```

Corrected outputs use a separate namespace:

```text
MoleSG/Downstream/Result_evalfix/<dataset>/
MoleSG/Downstream/Model_evalfix/<dataset>/<experiment>/
```

For a new reproduction, `scripts/run_model.py` writes under the supplied
`--run-root` instead of these research-directory paths. Retain predictions,
sample manifests, and validation-selected checkpoints there. The included
statistics script can rebuild reported summaries without those large files.
