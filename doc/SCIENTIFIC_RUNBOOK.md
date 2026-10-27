# Scientific workflow

Run commands from the repository root in the `smoke-science` environment. The configuration is `configs/scientific_protocol.yaml`; the full experimental design is in [SCIENTIFIC_PROTOCOL.md](SCIENTIFIC_PROTOCOL.md).

## Environment

```bash
conda env create -f environment_scientific.yml
conda activate smoke-science
python -m scientific.cli doctor
python -m pytest -q
```

For an existing environment, use `conda env update -n smoke-science -f environment_scientific.yml`. Check `python -m pip --version` if the interpreter appears incorrect. CUDA training requires a working NVIDIA driver. LPIPS downloads official AlexNet and calibration weights on first use; preload them when working offline.

## Data preparation

```bash
python -m scientific.cli inventory --output scientific_data/inventory.json
python -m scientific.cli download --videos smoke_videos
python -m scientific.cli prepare \
  --videos smoke_videos --output scientific_data/smoke \
  --cache-dir scientific_models/huggingface --device cuda --batch-frames 1
python -m scientific.cli check-data --data scientific_data/smoke
```

Skip `download` when the expected MP4 files already exist. Preparation loads one pinned CLIPSeg model and checks its safetensors fingerprint before creating the dataset. The initial weight download is approximately 603 MB. Add `--local-files-only` for a populated cache, or `--device cpu` for CPU preparation.

Text conditions are cached and each visual encoding is shared between prompts. `--batch-frames` controls independent frames per inference batch; its default is one. Every source acquisition is retained. The progress bar counts eligible videos present on disk; excluded metadata and missing files are still recorded.

Outputs:

- `clips/*.npz`: source RGB `raw_frames`, segmented RGB `frames`, and soft masks `alpha`, each with 36 time steps at 64×64.
- `all_windows.jsonl`, `train.jsonl`, `val.jsonl`, `test_dates.jsonl`, `test_views.jsonl`: identifiers, partitions, indices, timestamps and checksums.
- `preparation_events.jsonl`: source properties, missing videos and rejection reasons.
- `dataset_manifest.json`: configuration, selections, checksums, CLIPSeg revision and execution details.

Preparation refuses an existing output directory. An interrupted preparation is incomplete; use a new directory and point subsequent commands to that completed dataset. `check-data` reads all selected clips and checks partitions, temporal continuity, array shapes, types and mask bounds.

## Forecast task

Choose `C` and `H` before training, with `C ≥ 2`, `H ≥ 1`, `C+H ≤ 36`. The following example uses five observations and fifteen forecasts:

```bash
export SMOKE_CONTEXT_FRAMES=5
export SMOKE_FORECAST_HORIZON=15
```

Use these same values throughout the campaign. Stored clips remain 36 frames long. RIVER training samples a target, its preceding reference and an earlier condition; evaluation supplies only the chosen observations.

## Resource probe and VQGAN

```bash
python -m scientific.cli \
  --context-frames "$SMOKE_CONTEXT_FRAMES" --horizon "$SMOKE_FORECAST_HORIZON" \
  pilot --campaign runs_scientific/smoke

python -m scientific.cli \
  --context-frames "$SMOKE_CONTEXT_FRAMES" --horizon "$SMOKE_FORECAST_HORIZON" \
  autoencoder --campaign runs_scientific/smoke --data scientific_data/smoke
```

The pilot uses synthetic tensors to measure memory and choose micro-batches; it does not estimate predictive quality. `training.max_updates` fixes the number of optimizer updates. Effective batch size is sixteen, implemented through accumulation. VQGAN uses reconstruction, calibrated LPIPS, VQ and adversarial losses, and selects `best.pth` using validation LPIPS.

## Predictor training

```bash
for method in simvp flow_uniform flow_alpha; do
  for seed in 17 42 2026; do
    python -m scientific.cli \
      --context-frames "$SMOKE_CONTEXT_FRAMES" --horizon "$SMOKE_FORECAST_HORIZON" \
      train --campaign runs_scientific/smoke --data scientific_data/smoke \
      --method "$method" --seed "$seed" \
      --autoencoder runs_scientific/smoke/autoencoder/best.pth \
      --pilot runs_scientific/smoke/pilot.json || break 2
  done
done
```

SimVP uses RGB directly; Flow Matching uses the frozen common VQGAN. Checkpoints contain parameters, optimizers, scheduler, random states and provenance. An artifact becomes a completed selected model only after all planned updates and the final validation finish. Commands sharing a campaign directory are protected against concurrent writes.

## Monitoring and recovery

```bash
python -m wandb login
```

Add `--wandb --wandb-project smoke-science` to either training command. The run URL appears in the terminal. `--wandb-entity` and `--wandb-name` customize its destination and display name. `--wandb-mode offline` saves local sessions without cloud synchronization.

Metrics include generator/discriminator losses, reconstruction, perceptual and VQ terms, validation LPIPS, learning rates, progress, update time and CUDA memory. Validation media compare real RGB on the left with reconstruction or prediction on the right. Defaults are `--wandb-log-every 25` and `--wandb-media-every 250`; images are produced at eligible validation steps and final validation. Before the discriminator activates, its loss is zero and its active flag is false.

Scalar output is always kept in `<run>/metrics.jsonl`. WandB identity is stored in `<run>/wandb.json`, and session data in `<run>/tracking/`. A compatible online restart reuses its run ID; offline restarts create separate local sessions. Charts use optimizer `global_step` as their axis.

After a clean interruption, repeat the training command with `--resume latest`, or give `--resume <checkpoint_path>`. The same sources, configuration, data, seed and referenced artifacts are required. A process must be restarted to load source changes. Monitoring preserves the training RNG state; no checkpoint restores metrics that were never recorded.

## Evaluation

```bash
for method in persistence optical_flow simvp flow_uniform flow_alpha; do
  for seed in 17 42 2026; do
    for split in test_dates test_views; do
      extra=()
      if [ "$method" != persistence ] && [ "$method" != optical_flow ]; then
        extra=(--checkpoint "runs_scientific/smoke/${method}_${seed}/best.pth" \
               --autoencoder runs_scientific/smoke/autoencoder/best.pth)
      fi
      python -m scientific.cli \
        --context-frames "$SMOKE_CONTEXT_FRAMES" --horizon "$SMOKE_FORECAST_HORIZON" \
        evaluate --campaign runs_scientific/smoke --data scientific_data/smoke \
        --method "$method" --seed "$seed" --split "$split" "${extra[@]}" || break 3
    done
  done
done
```

This example uses Bash. Deterministic methods are recorded under the three seed identifiers for paired comparisons; this does not create independent realizations. Flow Matching produces four realizations per clip, all included in the reported mean.

Each evaluation exports `metrics.csv`, `runtime.csv`, `evaluation.json`, NPZ predictions with checksums and fixed qualitative panels. Evaluation refuses existing output. To retry an incomplete evaluation, repeat its command with `--retry-incomplete`; the failed attempt is kept separately and the complete fixed panel is regenerated.

## Analysis

```bash
python -m scientific.cli \
  --context-frames "$SMOKE_CONTEXT_FRAMES" --horizon "$SMOKE_FORECAST_HORIZON" \
  summarize --inputs runs_scientific/smoke/evaluation/* \
  --output runs_scientific/smoke/analysis
```

Outputs include `summary.json`, `horizons.csv`, `quality_cost.csv`, horizon curves, quality/cost plots and `failure_review.csv`. Aggregation checks identity and complete window/horizon/realization coverage. It rejects injected test metrics, non-finite values and duplicated rows. Partial campaigns are explicitly labeled.

Preserve the configuration, source revision, environment, manifests, selected weights and raw metrics with a published result. Choose displayed horizons and qualitative samples before examining test performance.

## Segmentation benchmark

```bash
python tools/benchmark_clipseg.py \
  --input-npz scientific_data/smoke/clips/VIDEO_ID_0000.npz \
  --output runs_scientific/clipseg_benchmark.json --device cuda
```

Replace the input with an existing clip. The tool compares the standard HF forward pass with shared visual encodings and frame batches 1/4/8. It synchronizes CUDA, excludes warm-up, repeats measurements and checks score/mask differences. Input shape is recorded; comparisons should use the same shape, device and numerical settings. Timing excludes model loading, video decoding, spatial filtering and file compression.
