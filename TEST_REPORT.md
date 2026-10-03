# Release test report

Checked on 2026-10-03 in the project MoleSG environment. All checks below ran
on local CPU. No training, dataset inference, Slurm submission, commit, or push
was performed as part of this release preparation.

| Check | Outcome |
| --- | --- |
| Reported statistics | 640 neural folds and 200 CPU folds reproduced the supplied dataset summaries, macro summaries, and paired contrasts, including 10,000-draw bootstrap intervals, within numerical tolerance of 1e-12 |
| Configuration commands | All 16 configurations generated pretraining, finetuning, and final-test commands with their declared options |
| Descriptor assignment | Fixed-boundary behavior and the ibuprofen TPSA/MolLogP example passed |
| Stable random assignment | Canonical-molecule keys retained their assignments across stage input orderings |
| Molecule routing | One branch per molecule, padding invariance, and nonzero primary-loss router gradients passed |
| ConLoss | A small-batch formula comparison, finite gradients, and the no-positive case passed |
| N8 loading | Synthetic pretraining state dictionaries loaded strictly into five architecture families covering the 16 configurations, with final-layer branch placement checked |
| Validation completion | A temporary 640-fold validation fixture released the panel. Missing validation or a mismatched sample manifest did not |
| Command portability | Nine public command entrypoints imported and displayed help from outside the checkout with CUDA disabled |
| Repository contents | Required files and bibliography keys passed the repository check. No weights, molecular datasets, figures, credentials, or project-server paths are included |

Run the commands in the README to repeat these checks. The tests use temporary
synthetic tensors or tiny prediction fixtures where model components are
involved. They do not perform inference on the experimental datasets.

## What was not repeated

- A new Conda environment was not installed from scratch. Dependency commands
  were checked in the existing project environment.
- Pretraining, full downstream finetuning, and final-test inference were not
  rerun. Statistical agreement reconstructs tables from reported fold metrics,
  not predictions from weights.
- Original large checkpoint files and processed datasets are external. Exact
  row order and the stored molecular corruption cache are required for the
  original protocol. See `data/README.md`.
- The retained pretraining resume path does not save optimizer state and every
  reconstruction-head state. It must not be described as exact trajectory
  restoration.

The main release changes expose portable paths and a compact command wrapper.
The final P6/P8 model computations, assignment rules, corrected evaluation,
and statistical implementations were copied from the experiment source.
