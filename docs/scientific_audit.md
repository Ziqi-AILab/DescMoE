# Scientific Audit

## Audit Outcome

The manuscript contains a coherent chemical-informatics question, a substantial
experimental sequence, and a well-defined implementation. It is not yet ready
to make a final benchmark claim because the legacy downstream protocol does not
evaluate complete validation and test splits. The revised draft therefore uses
the current numbers as a qualified development snapshot.

## Scientific Core

The method maps a scalar descriptor to one of eight fixed intervals. The region
index selects one branch in a conditioned graph Transformer feed-forward layer.
MolLogP and TPSA provide two chemically recognizable projections of molecular
space. Shared graph and SMILES pretraining retains structural learning, while
the descriptor determines a high-level nonlinear transformation.

The pretraining assignment file was checked independently against the source
corpus. RDKit 2022.09.5 was used to recompute TPSA and Wildman-Crippen MolLogP
for all 250,000 ZINC rows. The stored TPSA and MolLogP arrays each had zero
mismatches under the documented fixed boundaries. The preprocessing directory
also contains one indexed molecule file for every source-row index. The
equal-frequency code in `descriptor_analysis.py` belongs to descriptor
screening and did not generate the assignment arrays used during pretraining.

Figure 1 uses neutral ibuprofen as an illustrative molecule. It is absent from
the ZINC 250K source rows and is not described as a training example. Its
verified values are TPSA 37.30 square angstroms and MolLogP 3.0732. These map
to zero-based TPSA region 1 and MolLogP region 5. Under the locked branch
allocation, those paths use KAN and MLP, respectively.

## Evidence Status

| Evidence | Status | Suitable manuscript use |
| --- | --- | --- |
| ZINC15 descriptor occupancy | Existing deterministic analysis | Main text and SI |
| Last, odd, even placement | Legacy exploratory downstream screen | Design rationale only |
| TPSA and MolLogP k0 to k7 | Legacy exploratory downstream screen | Branch-function analysis only |
| N8-aligned 440 fold CSVs | File-complete but sample-truncated legacy evaluation | Development snapshot, not final corrected benchmark |
| N4 40 fold CSVs | Legacy auxiliary depth reference | SI only |
| ConLoss embedding analysis | Existing legacy embeddings | Objective-consistency analysis |
| Runtime and FLOPs | Existing logs and analytical estimates | Qualified computational analysis |
| Corrected full-validation and full-test results | Not available | Must remain absent |

## P0 Findings

1. Legacy validation used `shuffle=True` with `drop_last=True`.
2. Legacy test used `shuffle=False` with `drop_last=True`.
3. Only the old-protocol best downstream checkpoint is retained for each run.
4. Exact corrected checkpoint selection cannot be reconstructed without
   rerunning downstream finetuning.
5. Historical placement and branch screens appear to use test-set summaries.
6. The current complete models do not outperform the matched Base on macro
   ROC-AUC.

## Claim Boundaries

### Supported

- Fixed TPSA and MolLogP intervals create measurable, chemically distinct, and
  strongly imbalanced molecular populations.
- Descriptor values deterministically select the active computation branch.
- The two descriptor axes yield different branch-wise MLP or KAN choices in
  the historical screen.
- ConLoss produces much stronger descriptor-region separation in the examined
  embedding spaces.
- Fixed assignment removes a learned router but does not reduce observed
  runtime in the current implementation.

### Not Supported

- General superiority over vanilla pretraining.
- A claim that TPSA or MolLogP is the best descriptor for conditional
  computation.
- A claim that KAN is generally better than MLP.
- A claim that each branch corresponds to a chemically discrete class.
- A claim that t-SNE separation proves improved molecular property prediction.
- A final corrected benchmark claim from the legacy 440 CSVs.

## Required Development-Draft Treatment

- Replace `strict N8` with `legacy N8-aligned result snapshot`.
- State that 440 fold files exist while complete sample coverage is not
  established.
- Put an author-facing development notice before the main text.
- Describe placement and KAN screening as exploratory model-development
  evidence.
- Keep all current numbers unchanged and visually mark their evidence status.
- Reserve final benchmark wording for a future corrected evaluation.

## Recommended Submission Gate

The paper should not be labeled submission-ready until the corrected protocol
uses deterministic validation and test loaders with `drop_last=False`, freezes
model choices using validation evidence, and evaluates the test split only
after model definition is fixed.
