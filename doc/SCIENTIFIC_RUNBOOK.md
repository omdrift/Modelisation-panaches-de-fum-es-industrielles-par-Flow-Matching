# Industrial Plume Forecasting — execution guide

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

For the complete study, use the single orchestrator. It launches preparation, automatic checks, the pilot, VQGAN, all predictor seeds, both test splits and the summary sequentially:

```bash
python -m scientific.cli --context-frames 5 --horizon 15 pipeline \
  --videos smoke_videos --data scientific_data/smoke \
  --campaign runs_scientific/smoke \
  --cache-dir scientific_models/huggingface \
  --wandb --wandb-project smoke-science
```

Add `--dry-run` to print the ordered matrix (45 stages) without starting work. Add `--resume` after an interruption. If the reference configuration has changed, also use `--protocol runs_scientific/smoke/protocol.yaml` to select the original frozen settings. The pipeline verifies hashes before reusing any completed stage and writes `pipeline.json`, a frozen `protocol.yaml`, and one log per stage.

### Starting from prepared NPZ files

A completed dataset can be reused after removal of the source MP4 directory. Keep the complete prepared directory: NPZ files, `dataset_manifest.json`, and the partition manifests. No CLIPSeg cache is needed to reuse these outputs:

```bash
python -m scientific.cli --context-frames 5 --horizon 15 pipeline \
  --data scientific_data/smoke --campaign runs_scientific/smoke \
  --wandb --wandb-project smoke-science
```

The preparation stage is reused automatically. Its specification must match the requested protocol; `check-data` still verifies hashes, shapes, masks, temporal continuity and disjoint partitions before the pilot or training. Incomplete or incompatible prepared data is rejected. A completed preparation also permits campaign resume without the original `--videos` argument; its recorded origin is retained. Source MP4s and their original path are needed to resume an interrupted preparation; `--videos` defaults to `smoke_videos`. Initial preparation or changing segmentation also requires MP4s. Reusing data does not reuse model weights from another campaign.

## Hyperparameter calibration

The supplied values are exploratory. The pipeline does not automatically select a learning rate. The [training design](TRAINING_DESIGN.md) documents their origins, the controlled calibration procedure and chart interpretation.

`python tools/plan_calibration.py --data scientific_data/smoke --output runs_scientific/calibration_vqgan_lr --learning-rates 0.00005 0.0001 0.0002 --updates 15000` registers three equal-update VQGAN candidates without starting them. Review `study.json` and `candidate_*.yaml`; `bash runs_scientific/calibration_vqgan_lr/commands.sh` executes them sequentially when intended. Every candidate starts in a separate directory with seed 42, fixed data and the same image sampling schedule. Test scores are excluded from selection. The declared duration limits the comparison; it does not establish eventual convergence.

VQGAN settings live in `training.autoencoder`; common LR, weight decay, warmup and validation interval apply to predictors. Each run exports `effective_training.json` and adds that description to its W&B configuration. Changing settings requires a separate experiment, not a checkpoint resume.

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

Each validation also computes diagnostic metrics on a fixed training panel in inference mode. VQGAN uses up to 60 training clips, one image per clip; predictors use up to 8 clips to limit additional autoregressive generation. Training clips are ranked by seeded identifier hashes. The validation panel and the primary checkpoint-selection LPIPS remain unchanged.

### Compact workspace

An open interpretation card explains the axes, units, useful signals and limitations. The main VQGAN section contains six line charts: train/validation/best LPIPS, train/validation smoke MSE, train/validation background MSE, LPIPS gap, codebook perplexity, and fraction of codes used. The next section contains fixed reconstruction/error/mask examples. Date/view analysis, optimization/resources, and selection/baseline diagnostics are collapsed. Date and view metrics share two charts instead of producing one panel per group. Predictor layouts omit VQGAN codebook charts.

All charts use optimizer `global_step` without smoothing. Counters such as valid frames, assignments and window counts remain available in local JSON and recorded W&B data, without automatic panels. Logging hides metrics from automatic plots by default and explicitly exposes a small set of quality signals. The named workspace disables automatic panel creation. Changing logging metadata alone does not reorganize an already saved workspace.

The optional editor uses a separate environment so its SDK does not change the frozen training runtime:

```bash
python -m venv .venv-dashboard
.venv-dashboard/bin/python -m pip install -r tools/requirements_dashboard.txt
.venv-dashboard/bin/python tools/configure_wandb.py \
  --run-directory runs_scientific/smoke/autoencoder --apply
```

Omit `--apply` to inspect the plan. The editor requires the existing W&B login and a run directory containing `wandb.json` and `run.json`. It creates or updates the current user's named view `Suivi scientifique - VQGAN` (or the selected predictor method), filtered to that method. The returned URL opens this view directly. It does not create a separate experiment run. Named views keep the curated layout separate from automatic or personal views that another browser tab may save concurrently.

Before a save, `<campaign>/dashboard/workspace_before_*.json` records the previous view, or `null` for a new view. Updates submit the existing view identity and timestamp. An independent API read verifies persistence; `<campaign>/dashboard/workspace_<method>.json` records the result and URL. Existing named-view filters and run selection are preserved during layout updates. Histories, checkpoints and training processes are preserved. The layout can be changed while training is active.

The design uses W&B's documented [metric visibility and custom axes](https://docs.wandb.ai/ref/python/experiments/run/), [multi-series line plots](https://docs.wandb.ai/guides/app/features/panels/line-plot/reference), and the official [Workspaces SDK](https://github.com/wandb/wandb-workspaces). The pinned SDK creates named views and serializes panels; its GraphQL transport updates an existing owned view after a local backup. The pinned API was verified against the configured service.

### Recorded diagnostics

| WandB series | Meaning |
|---|---|
| `train_eval/lpips`, `validation/lpips` | Same perceptual metric, fixed panels, no data augmentation or dropout |
| `generalization/lpips_gap` | Validation minus training-panel LPIPS; descriptive, not a statistical test |
| `validation/smoke_mse`, `validation/background_mse` | Errors on estimated smoke and background regions; absent regions remain missing |
| `validation/baseline_lpips` | Black-image reconstruction baseline for VQGAN; persistence baseline for predictors |
| `validation/by_date/*`, `validation/by_view_id/*` | Performance by acquisition date and camera/view |
| `validation/codebook/*`, `validation/smoke_codebook/*` | VQ perplexity, active codes, used fraction and number of assignments |
| `train/gradient_norm_before_clip` | Gradient norm before the existing clipping operation |
| `selection/best_step`, `selection/updates_since_best` | Selected validation checkpoint and time since its last improvement |

`generalization/possible_overfit` is available after five diagnostic validations. It is an advisory flag when the fitted training LPIPS slope is negative, the validation slope is positive, and current validation LPIPS exceeds the best value by more than 10%. This heuristic does not demonstrate overfitting, stop training, reject a checkpoint or change model selection. Train/validation distribution differences and noisy validation scores can produce similar patterns.

Per-image measurements, panel identifiers, mask-region coverage and codebook assignment counts are summarized in `<run>/diagnostics/step_<update>.json`. These files contain no model weights. The selected checkpoint carries `best_diagnostics` when diagnostics were available at selection. A local `diagnostics/summary.png` is generated at training completion. Four-column validation media display true RGB, prediction, absolute error and estimated mask.

A saved VQGAN checkpoint can also be inspected without resuming training:

```bash
python -m scientific.cli --protocol runs_scientific/smoke/protocol.yaml \
  inspect-autoencoder --campaign runs_scientific/smoke \
  --data scientific_data/smoke \
  --checkpoint runs_scientific/smoke/autoencoder/best.pth
```

Inspection reads only the training and validation partitions. It writes `inspection.json`, reconstruction/error panels and a diagnostic plot under `runs_scientific/smoke/inspection/vqgan_step_<update>/`. It does not modify the checkpoint or evaluate forecasting. Existing inspection outputs require a new `--output` directory.

Scalar output is always kept in `<run>/metrics.jsonl`. WandB identity is stored in `<run>/wandb.json`, and session data in `<run>/tracking/`. A compatible online restart reuses its run ID; offline restarts create separate local sessions. Charts use optimizer `global_step` as their axis.

Recovery retains the two newest numbered checkpoints and the validation-selected `best.pth`. A checkpoint is committed by atomic replacement; partial files from failed writes are removed. Rotation occurs only after a new checkpoint has been written successfully. Additional temporary space for one complete checkpoint is required during serialization and best-checkpoint updates.

Resume compatibility checks the actual Python sources, dataset, protocol, seed, model artifacts and runtime. Git commit metadata and documentation diffs are retained as provenance but do not block an otherwise compatible resume. A training-preserving I/O or observational-monitoring maintenance patch for an existing campaign requires an explicit local `maintenance.json` binding its reviewed source fingerprints to the frozen protocol; unrelated source or configuration changes remain rejected.

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
