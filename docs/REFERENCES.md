# Implementation references

The primary source for this package is the code developed in this conversation.
The packaged code is not a reproduction of an external separator implementation.

Official documentation consulted for the surrounding implementation:

- NumPy loading / NpzFile lifecycle: https://numpy.org/doc/stable/reference/generated/numpy.load.html
- PyTorch checkpoint-loading contract: https://docs.pytorch.org/docs/stable/generated/torch.load.html
- PyTorch Lightning optimizer/module interface: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
- Original score-SDE predictor/corrector implementation, for terminology and reference:
  https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py

No external model weights, datasets, or vendor code are redistributed. The
mathematical conventions specific to this complex convolutive model are written
explicitly in MATHEMATICS.md and tested locally; they are not inferred simply by
copying scalar real-valued score-SDE code.
