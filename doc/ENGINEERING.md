# Industrial Plume Forecasting — engineering contracts

## Responsibilities

The CLI exposes preparation, fixed-configuration campaigns and controlled studies. Data preparation owns RGB/mask transformation and immutable manifests. Training owns parameter updates and recoverable state. Diagnostics measure fixed panels without advancing training randomness. Study orchestration ranks candidates on validation and freezes selected artifacts before held-out evaluation.

| Concern | Implementation | Contract |
|---|---|---|
| Public entry point | `scientific/cli.py` | Existing imports and commands remain compatible; `study` adds controlled comparison. |
| Study orchestration | `scientific/study.py` | Sequential execution, explicit phase dependencies, provenance, persistent trials and final selection. |
| Data and partitions | `scientific/data.py`, `scientific/protocol.py` | 36 frames, 64×64 RGB, finite estimated alpha, temporal continuity and disjoint identifiers. |
| Training | `scientific/runner.py`, `model/` | Fixed effective batch, declared objectives, compatible latent dimensions and reproducible generation. |
| Validation plateau | `scientific/early_stopping.py` | Raw-minimum selection is separate from significant-improvement patience; counters persist. |
| Checkpoints | `training/checkpoints.py` | Atomic writes, two recovery states, selected model, RNG and optional scaler restoration. |
| Diagnostics | `scientific/diagnostics.py`, `scientific/monitoring.py` | Fixed panels and draws, explicit units, image/animation exports, local and optional W&B outputs. |
| Presentation | `tools/configure_wandb.py`, `tools/monitor_campaign.py` | Method-specific views, escaped status HTML, isolated observation and offline-mode preservation. |

## Study contract

`configs/experiment_study.yaml` declares the experiment before execution. The default comparison crosses three learning rates and two schedulers for each model. Exploration and transfer probes are capped at 15,000 updates; finalist confirmation and weighted Flow at 50,000. The complete staged plan contains at most 29 training runs, including three-seed confirmations.

`study --dry-run` creates no files, starts no GPU work and does not require importing Optuna. An executed study stores trials in `optuna.sqlite3`, progress in `study.json`, tabular evidence in `experiments.csv`, and automatic reports in JSON/Markdown. Each run has a resolved protocol. Resume validates identities and hashes before reusing completed results.

VQGAN is selected by reconstruction followed by controlled forecast probes. Flow is selected by mean validation LPIPS across confirmation seeds; weighted Flow reuses the selected settings. Intermediate and final test results cannot promote candidates. `final_selection.json` records the frozen choices and artifact hashes before test decoding and evaluation.

A test set may be identified by manifest during search, but its pixels and scores remain deferred. Search-time audits inspect train/validation integrity and exact raw-RGB duplicates. Similarity flags are evidence, not instructions to silently remove or move data. Correcting an invalid partition requires a distinct dataset version.

## Training and recovery invariants

- Context, future horizon and stored clip length are distinct: 5, 15 and 36 in the reference study.
- Flow training uses a target/reference/earlier-condition triplet; generation receives observations only and then its own predictions.
- Diagnostics preserve Python, NumPy, torch and CUDA random states and module modes. Fixed Flow objective draws do not consume training randomness.
- The minimum raw validation LPIPS selects `best.pth`. `min_delta` controls the patience reference, not whether a lower raw score is eligible.
- VQGAN validation/patience are 250 updates/10 checks; Flow uses 2,500/four in the study. Both begin counting after 5,000 updates and use `min_delta=1e-4`.
- A valid early stop completes the run with an explicit reason. An interrupted validation is replayed before another update. Terminal outputs remain immutable on resume.
- Checkpoints include optimizer, scheduler, RNG, patience and scaler state when AMP is enabled. VQGAN includes discriminator state. Rotation follows a successful atomic write.
- Exact resume preserves the recipe. Fine-tuning is a new run using `training.init_from`, a parent fingerprint and explicit resets of optimizer, scheduler, RNG and early stopping. A missing parent discriminator requires an explicit discriminator reset.

The import namespace `scientific`, model classes and existing public signatures remain stable. Optional study, scheduler, initialization and precision settings extend the implementation without silently changing a fixed campaign's frozen configuration.

## Observability and failures

Quality curves use comparable fixed train/validation panels. VQGAN adds codebook usage and full-clip reconstructions; Flow adds fixed-draw latent objectives, decoded-image quality and full-horizon previews. PNG/GIF exports use fixed identifiers. LPIPS controls selection; an objective decrease, a trend flag or a patience limit is not proof of generalization.

W&B is optional. Local logs, metrics and reports remain usable without a network connection. W&B views group meaningful comparisons and leave detailed slices collapsed. The artifact observer has its own lock and `status.json`, separate from training `run.json` records and provenance transitions.

Observer initialization is followed immediately by protected cleanup. Disk, logging and final-status errors still attempt SDK finalization. A secondary cleanup error warns without replacing the triggering exception. Resume checks project/entity destination; offline runs do not implicitly initialize online tracking. All dynamic HTML fields are escaped and external links are restricted to supported HTTPS tracking hosts. The observed protocol must match the frozen pipeline identity.

Failed, interrupted, OOM, non-finite and pruned runs remain recorded. An incomplete output is not a completed experiment. Evaluation summaries reject incomplete horizon/realization grids, duplicate keys, non-finite values and mismatched panels.

## Verification

The automated suite covers interface contracts and behavior, including:

- causal preprocessing, clip geometry, temporal continuity, mask bounds and split integrity;
- fixed search spaces, stage counts, candidate ranking, same-stage pruning and test deferral;
- scalar metrics, realization means, codebook statistics and fixed visual identifiers;
- training-RNG preservation, generation without future inputs and meaningful train/validation objective separation;
- constant/cosine scheduling, warmup, early-stop counters and strict checkpoint compatibility;
- checkpoint rotation, parent initialization, interruption recovery and optional scaler restoration;
- W&B failures, offline mode, disk failures, HTML escaping and independent observer cleanup;
- local plots, images, animations, structured reports and a short synthetic CPU study.

```bash
python -m pytest -q
python -m compileall -q scientific model training lutils tools tests
git diff --check
```

Tests establish software behavior on fixtures. They do not establish CUDA performance, segmentation accuracy, forecasting superiority or the selected hyperparameters' scientific validity. Such claims require the corresponding completed experiments and identified artifacts. The CPU GitHub Actions workflow is a software check; its configured existence alone is not evidence that a particular remote run succeeded.

## Publication boundary

Publish implementation, configurations, protocols, references, tests and component notices. Keep credentials, source videos, prepared tensors, pretrained weights, checkpoints and tracking sessions outside Git. Forecast claims must refer to complete held-out evaluations; reconstruction scores alone do not establish future-frame accuracy.
