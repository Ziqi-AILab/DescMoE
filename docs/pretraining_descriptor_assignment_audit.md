# Pretraining Descriptor Assignment Audit

The audit recomputed TPSA and Wildman-Crippen MolLogP for all 250,000
ZINC rows using RDKit 2022.09.5. The fixed project boundaries were
then compared row by row with the arrays loaded by pretraining.

| Check | Result |
| --- | ---: |
| Valid molecules | 250,000/250,000 |
| TPSA assignment mismatches | 0 |
| MolLogP assignment mismatches | 0 |
| Indexed preprocessed files | 250,000/250,000 |
| Missing or unexpected preprocessing indices | 0 |

The arrays used by pretraining therefore implement the documented fixed
boundaries exactly. The separate quantile-bucketing code in
`descriptor_analysis.py` belongs to the earlier descriptor-screening analysis
and does not generate `expert_ids.npz`.

Ibuprofen is not present in the
ZINC 250K source rows. Figure 1 consequently labels it as an illustrative
molecule rather than a sampled pretraining record.
