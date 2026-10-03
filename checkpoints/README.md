# External checkpoints

Weights are not included. Provide completed weights from the authors or train
using the final model matrix. With `RUN` as your reproduction directory:

```text
RUN/pretrain/<model>/compt/periodic_latest.pth
RUN/pretrain/<model>/smiles_encoder/periodic_latest.pth
RUN/pretrain/<model>/total_encoder/periodic_latest.pth
RUN/finetuned/<dataset>/best_model_<result_exp>_<dataset>_fold_<1-5>.pt
```

The graph checkpoint at zero-based epoch 299 is used for downstream transfer.
Only `compt/periodic_latest.pth` is needed for finetuning. The other component
weights are pretraining outputs, not downstream inputs. The legacy resume path
does not restore optimizer or decoder states and is not exact resume.

For existing weights elsewhere, supply `--pretrain-root /path/to/Model` to
`scripts/run_model.py`. Corrected finetuned checkpoints must be selected using
complete validation splits. Old incomplete-batch best checkpoints are not
accepted as corrected final-test checkpoints.
