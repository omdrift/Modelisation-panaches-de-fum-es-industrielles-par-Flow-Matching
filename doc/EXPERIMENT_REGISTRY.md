# Experiment registry

This registry records what can be tied to an artifact with evidence. A filename or README example alone does not prove which configuration produced a historical result.

| Experiment | Config | Checkpoint | Resolution | Latent | Dataset | Result |
|---|---|---|---:|---|---|---|
| Flow Matching media, step 109999 | Unknown; the GIFs do not identify a config snapshot | Not found in this checkout | Unknown | Unknown | Historical version unknown | `media/generated_videos_109999_f96b7834ed7ac1939875.gif`; `media/real_videos_109999_c99bc7aa458f1f121210.gif` |
| VQGAN epoch 56 reconstruction | Unknown | Not found in this checkout | Unknown | Unknown | Historical version unknown | `media/custom_recon_epoch_56.png` |
| VQGAN comparison | Unknown | Not found in this checkout | Unknown | Unknown | Historical version unknown | `media/vqgan_reconstruction_comparison.png` |
| Flow Matching 64 primary setup | `configs/flow_matching_64.yaml` | Expected at `runs_vqgan/vqgan_64_baseline/checkpoints/vqgan_epoch_50.ckpt`; not present | 64×64 | 8×8×256 | Local `final_dataset` | Config prepared; no run recorded |
| Flow Matching 128 historical setup | `configs/flow_matching_128.yaml` | Expected at `runs_vqgan/vqgan_128_historical/checkpoints/vqgan_epoch_50.ckpt`; not present | 128×128 | 16×16×256 | Local `final_dataset` | Historical/secondary; compatibility unverified |
| VQGAN 64 baseline setup | `configs/vqgan_64.yaml` | To be produced by the training command | 64×64 | 8×8×256 | Local `final_dataset` | Config prepared; no run recorded |

## Baseline captured before reorganization

- The backup branch `backup/before-reorganization` and tag `archive/pre-cleanup` point to commit `3d9814d`.
- The tracked visual results are under `media/`. No `*.ckpt`, `*.pth`, or `*.pt` file was found under this checkout during the initial inventory; the local `runs/` and `runs_vqgan/` directories were empty.
- Dataset split lists were present locally. `tools/validate_dataset.py` later counted 128,680 train frames, 15,818 validation frames, and 16,160 test frames grouped into 3,989, 498, and 497 videos respectively.
- The dataset scan found 263 train, 38 validation, and 29 test videos shorter than 16 frames. Those clips need an explicit policy for Flow Matching.

## New run record

For every new experiment, preserve a config snapshot and record the source commit, config and checkpoint checksums, dataset version, seed, GPU, Python/PyTorch/CUDA versions, exact command, execution date, and metrics. Link results here only after the checkpoint and its producing config have been checked together.
