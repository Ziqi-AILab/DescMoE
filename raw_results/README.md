# External Raw Results

Raw prediction CSV files, node-embedding pickles, training histories, and Slurm
logs are excluded. Canonical derived tables required by the plotting scripts
are included in `tables_v2/`.

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

The canonical CSV files under `tables_v2/` are small derived artifacts used to
regenerate the manuscript figures. They are intentionally included. Raw model
predictions and embeddings are intentionally excluded.
