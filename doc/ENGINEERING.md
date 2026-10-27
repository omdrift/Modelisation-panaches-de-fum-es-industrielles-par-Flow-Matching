# Industrial Plume Forecasting — engineering design and verification

## Responsibilities and boundaries

The command-line entry point coordinates the workflow. Preparation owns RGB/mask transformation and immutable dataset manifests. Training owns model updates and recoverable optimizer/RNG state. Diagnostics measure fixed panels without advancing training randomness. Evaluation receives only observed frames during generation; aggregation verifies complete metric coverage before producing reports.

| Concern | Implementation | Contract |
| --- | --- | --- |
| CLI definition and dispatch | `scientific/cli.py` | Existing subcommands, arguments, help text and exit behavior |
| RISE clips and partitions | `scientific/data.py`, `scientific/protocol.py` | 36 frames, estimated alpha, temporal continuity, disjoint identifiers |
| Execution provenance | `scientific/protocol.py` | One source-manifest implementation shared by runs, orchestration and calibration registration |
| Model updates | `scientific/runner.py`, `model/` | Existing sampling, objective, optimizer order and generation behavior |
| Validation plateau | `scientific/early_stopping.py` | Shared patience policy, committed validation decisions, raw minimum selection, persisted counters |
| Checkpoint storage | `training/checkpoints.py` | Atomic replacement, two recovery checkpoints, separate validation-selected artifact |
| Measurements and presentation | `scientific/diagnostics.py`, `scientific/monitoring.py` | Fixed examples, regional errors, explicit units, RNG preservation |
| Evaluation summary | Private helpers in `scientific/cli.py` | Validation, reduction, comparisons and plotting have separate responsibilities |

The import namespace, public function signatures, classes and canonical filenames are stable. The project name is Industrial Plume Forecasting. Compatibility paths are retained instead of introducing a second command-line implementation.

## Structural changes

- Separate parser construction, locked GPU dispatch and training dispatch. The same argument definitions and checks execute in the same order.
- Separate evaluation loading/validation, horizon reductions, paired comparisons, quality/cost reductions and plotting. Numerical reduction order and output schemas are retained.
- Consolidate duplicated source-tree walks into one private manifest helper. The four source roots, relative keys and hashing algorithm are unchanged.
- Keep training kernels and generation algorithms intact. Extracting them solely to reduce line count would introduce extra state boundaries around optimizer updates, tensor lifetimes and interrupted checkpoints.
- Retain independent automatic data checks when source MP4s are absent. A finalized prepared dataset is reusable; an incomplete preparation still needs its original video path.

## Verification contracts

`tests/contracts/public_api.json` records readable argument and return annotations for the declared public functions, classes and methods. `tests/contracts/cli_help.json` records the root help and twelve subcommand help screens. These are interface contracts, not experiment results. Deliberate API changes require explicit review of these files.

The fixed full-summary fixture exercises all five methods, the three predictor seeds, both test splits, and every horizon/realization cell. Aggregation rejects missing cells, non-finite metrics and mismatched test panels. The refactoring was also checked against the preceding implementation: the JSON, three CSV files and three PNG figures were byte-identical on that fixture.

The broader suite covers causal extraction, clip geometry and mask validity, target sampling, deterministic generation, gradient accumulation, codebook statistics, checkpoint rotation and interrupted recovery, optional W&B logging, and pipeline reuse after MP4 removal.

Early-stopping tests exercise all four learned trainers with CPU fixtures: significant versus raw improvements, initial phases, incomplete validation replay, finalization failures, terminal artifact hashes, and old-campaign opt-in through evaluation and inspection. A short completed run requires a decision bound to the frozen policy and maximum updates. The explicit upgrade only enables that policy, verifies old source fingerprints, refuses active or tested campaigns, and leaves checkpoint tensors unchanged. Fixed-exposure calibration disables the policy.

The GitHub Actions workflow runs the suite on Python 3.11 with CPU PyTorch. Other dependency pins are read from the canonical environment file. This job checks software contracts without starting a training campaign; it does not validate CUDA throughput or predictive quality.

```bash
python -m pytest -q
python -m compileall -q scientific model training lutils tools tests
git diff --check
```

No GPU campaign is needed for these software checks. They establish implementation contracts on fixtures, not forecasting quality on RISE. GPU numerical equivalence, completed multi-seed comparisons, independent segmentation accuracy and final held-out forecasting claims require their corresponding experimental evidence.

## Publication scope

The repository publishes implementation, protocol, references, tests and component licenses. Source videos, NPZ datasets, model caches, checkpoint tensors, credentials and tracking sessions remain outside Git. Forecast quality claims must link to complete, identified evaluations; reconstruction scores alone do not establish future-frame accuracy.
