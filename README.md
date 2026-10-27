# Smoke Flow Matching

Reproducible visual forecasting of industrial smoke with CLIPSeg, VQGAN and sparsely conditioned Flow Matching inspired by [RIVER](https://github.com/araachie/river).

The pipeline preserves **36 consecutive frames per video**. Smoke segmentation runs at the source resolution, followed by spatial guided filtering and downsampling to **64×64 RGB**. Choose the observed context `C` and forecast horizon `H` explicitly, with `C ≥ 2` and `C + H ≤ 36`.

The comparison includes persistence, optical-flow extrapolation, a compact SimVP adaptation, uniform Flow Matching, and alpha-weighted Flow Matching. The repository provides the implementation and experimental protocol. Predictive performance must be established by running the study.

## Installation

Use Python 3.11 and a CUDA-capable NVIDIA GPU for training and evaluation. The pinned environment uses PyTorch 2.7.1 and CUDA 12.8.

```bash
conda env create -f environment_scientific.yml
conda activate smoke-science
python -m scientific.cli doctor
python -m pytest -q
```

Run commands from the repository root. `doctor` checks the environment without downloading model weights or starting a job.

## Prepare data

The public RISE metadata snapshot is included in `prepare_data/metadata_02242020.json`. Video files are downloaded separately.

```bash
python -m scientific.cli inventory --output scientific_data/inventory.json
python -m scientific.cli download --videos smoke_videos
python -m scientific.cli prepare \
  --videos smoke_videos --output scientific_data/smoke \
  --cache-dir scientific_models/huggingface \
  --device cuda --batch-frames 1
python -m scientific.cli check-data --data scientific_data/smoke
```

CLIPSeg weights are downloaded to the selected cache on first use. Add `--local-files-only` to reuse a populated cache without network access. `--device cpu` supports CPU preparation. The output directory must be new.

Each NPZ contains the 36 resized source RGB frames, segmented RGB frames, and soft masks. Metadata, partitions, acquisition indices and checksums are retained. The masks are semantic estimates, not independent annotations or physical opacity measurements.

## Train

Choose and freeze a forecast task before training. These values are an example:

```bash
export SMOKE_CONTEXT_FRAMES=5
export SMOKE_FORECAST_HORIZON=15

python -m scientific.cli \
  --context-frames "$SMOKE_CONTEXT_FRAMES" --horizon "$SMOKE_FORECAST_HORIZON" \
  pilot --campaign runs_scientific/smoke

python -m scientific.cli \
  --context-frames "$SMOKE_CONTEXT_FRAMES" --horizon "$SMOKE_FORECAST_HORIZON" \
  autoencoder --campaign runs_scientific/smoke --data scientific_data/smoke
```

The VQGAN checkpoint selected on validation is `runs_scientific/smoke/autoencoder/best.pth`. Then train predictors with seeds **17, 42 and 2026**. See the [runbook](doc/SCIENTIFIC_RUNBOOK.md) for all commands.

Add `--wandb --wandb-project smoke-science` to `autoencoder` or `train` to enable online monitoring after `python -m wandb login`. `--wandb-mode offline` records events locally. Scalar metrics are always written to `metrics.jsonl`. Use `--resume latest` to continue a compatible checkpoint with the same data, configuration and source code.

## Documentation

- [Scientific protocol, in French](doc/SCIENTIFIC_PROTOCOL.md): data splits, methodology, experiments, evaluation and limitations.
- [Runbook](doc/SCIENTIFIC_RUNBOOK.md): preparation, training, monitoring, evaluation and export commands.
- [Experiment matrix](doc/EXPERIMENT_MATRIX.csv): planned inputs, commands and outputs.
- [References](doc/SCIENTIFIC_LITERATURE.md) and [BibTeX](doc/references.bib).

## Repository layout

```text
configs/       Scientific configuration
scientific/    Preparation, training, monitoring, evaluation and analysis
model/         Flow Matching predictor and VQGAN components
training/      Checkpoint and random-state serialization
lutils/        Tensor and dictionary helpers
prepare_data/  Public RISE metadata snapshot
tools/         Reproducible inference benchmark
tests/         Automated correctness and recovery tests
```

The command-line benchmark is `tools/benchmark_clipseg.py`. Tests live in `tests/`. Data, downloaded weights, checkpoints, credentials and experiment outputs are excluded from source control and source releases.

## Scope

The study concerns two-dimensional visual prediction. CLIPSeg can confuse smoke, steam, clouds and background structures. RGB multiplied by a soft mask can retain background visible through transparent smoke. This pipeline does not estimate concentrations, emissions or validated physical plume dynamics. LPIPS is the primary forecasting metric; prediction errors and uncertainty must be reported on the fixed test panels.

## License and attribution

Code is distributed under [GPL-3.0](LICENSE), consistent with the RIVER-derived components. Required component notices and data attribution are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). External datasets and pretrained model weights retain their own terms and are obtained separately.

For contributions, see [CONTRIBUTING.md](CONTRIBUTING.md).
