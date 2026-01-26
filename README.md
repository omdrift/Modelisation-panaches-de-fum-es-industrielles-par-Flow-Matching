# Modelisation-panaches-de-fum-es-industrielles-par-Flow-MatchingSetup Environment

Prerequisites: Install conda (or mamba) and ensure NVIDIA drivers are installed (nvidia-smi works).

Env file: use the environment file env.yml.

Create the env (recommended):

```
# with conda
conda env create -f env.yml -n smoke

# or faster with mamba
mamba env create -f env.yml -n smoke

conda activate smoke
```