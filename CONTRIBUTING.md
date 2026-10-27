# Contributing

Use Python 3.11 and the environment in `environment_scientific.yml`.

Before submitting a change, run:

```bash
python -m pytest -q
python -m py_compile scientific/*.py model/**/*.py training/*.py
```

Keep scientific identities explicit: do not alter the frozen protocol, data split, seeds, or evaluation task without updating the protocol and its documentation. Never commit videos, model weights, checkpoints, WandB credentials, or campaign outputs. Use a new campaign directory for experiments and publish its manifests and metrics separately.
