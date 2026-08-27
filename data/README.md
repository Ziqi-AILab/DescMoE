# External data

The repository does not redistribute ZINC15 or MoleculeNet data. Obtain source
files under their original licenses and terms. The project expects the
following local layout.

Expected ZINC files:

```text
MoleSG/Data/zinc15/zinc15_250K.csv
MoleSG/Data/zinc15/expert_ids.npz
MoleSG/Data/zinc15/zinc15_0.25_geo/preprocess/
```

Downstream datasets follow the paths expected by
`MoleSG/Downstream/dataset_graph.py`:

```text
MoleSG/Downstream/Data/bbbp/preprocess/bbbp.pickle
MoleSG/Downstream/Data/tox21/preprocess/tox21.pickle
MoleSG/Downstream/Data/toxcast/preprocess/toxcast.pickle
MoleSG/Downstream/Data/sider/preprocess/sider.pickle
MoleSG/Downstream/Data/clintox/preprocess/clintox.pickle
MoleSG/Downstream/Data/bace/preprocess/bace.pickle
MoleSG/Downstream/Data/hiv/preprocess/hiv.pickle
MoleSG/Downstream/Data/muv/preprocess/muv.pickle
```

The processed objects must preserve source-row order so the corrected runner
can create stable `<dataset>:<source_row_index>` sample identifiers. Dataset
contents are ignored by Git through the repository `.gitignore`.
