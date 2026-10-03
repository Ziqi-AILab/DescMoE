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

The release wrapper accepts these locations outside the repository with
`--zinc-cache`, `--downstream-data` and `--control-dir`.
Each downstream pickle stores `(RDKit molecule sequence, label array)` with
the original processed row order. Missing labels are -1. Stable sample IDs
use that processed row index, not a newly deduplicated SMILES list.

Each ZINC cache file is named `processed_geometric_mol<row_index>.npz`.
Its keys are `node_features`, `bond_features`, `adjacency_matrix`,
`mask_node_labels`, `masked_atom_indices`, `token_ids`, `labels`, `edge_attr`,
`edge_index`, `num_atoms`, and `x`. The ZINC CSV `smiles` column must have the
same source-row order. The assignment preparation script creates the required
`descriptor_controls.npz` and frozen metadata from these external inputs.

Obtain the exact processed MoleSG resources from the authors for numerical
reproduction. A generic ZINC or MoleculeNet download is not a substitute for
the supplied filtering, ordering, tokenizer and stored corruption masks.
The original preprocessing functions are included for inspection. Regenerating
stochastic masks does not reproduce an already saved cache bit for bit.
No download service for the exact processed artifacts is advertised here.
