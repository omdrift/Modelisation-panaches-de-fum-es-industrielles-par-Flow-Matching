# Smoke Flow Matching Training Pipeline

## Overview
This document explains how to train flow matching for smoke prediction using the RIVER algorithm approach.

## Pipeline Architecture

```
Raw Videos → Smoke Extraction → VQVAE Encoding → Flow Matching Training → Smoke Prediction
```

### Stage 1: Data Preparation
Your smoke frames are already extracted and organized in `final_dataset/`:
- **Train**: 128,680 frames
- **Val**: 15,818 frames  
- **Test**: 16,160 frames

Each frame is an extracted smoke region (64x64 RGB) on black background.

### Stage 2: VQVAE/VQGAN Training 
You have trained VQVAE checkpoints:
- Checkpoint: `runs/vqvae_vqvae_masked_finetune/vqvae_epoch_25.ckpt`
- Encoding: 64×64 RGB image → 8×8×256 latent representation
- This compresses each frame to a compact latent code `z`

**Key Point**: VQVAE/VQGAN is frozen during flow matching training. It only serves as an encoder/decoder.

### Stage 3: Flow Matching Training (RIVER Algorithm)

#### Algorithm (from RIVER paper)
```
For each training iteration:
  1. Load video sequence x = {x₁, ..., x₁₀}  (10 smoke frames)
  2. Encode with VQVAE: z = {z₁, ..., z₁₀}
  3. Sample target frame index: τ ~ {3, ..., 10}
  4. Sample diffusion timestep: t ~ U[0,1]
  5. Sample noisy latent: zₜ ~ p(z|z^τ)
  6. Calculate target vector: u_t(z|z^τ)
  7. Sample context frame: c ~ {1, ..., τ-2}
  8. Predict with model: v_t(zₜ | z^{τ-1}, z^c, τ-c)
  9. Loss: ||v_t - u_t||²
```

#### What Each Component Does

**Reference Frame (z^{τ-1})**:
- The frame immediately before the target
- Provides local motion information
- Critical for learning smoke flow dynamics

**Context Frame (z^c)**:
- A random earlier frame (from start to τ-2)
- Provides long-term temporal context
- Helps model understand smoke evolution patterns

**Index Distance (τ - c)**:
- How far apart context and target are
- Encoded as input to the model
- Helps model understand temporal scale

**Flow Matching**:
- Models the "vector field" from noise to real smoke
- Learns smooth interpolation paths in latent space
- MSE loss between predicted and target vectors

#### Current Configuration

```yaml
# configs/smoke_dataset.yaml
training:
  num_observations: 10        # Use 10 frames per sequence
  condition_frames: 1         # Number of explicit conditioning frames
  frames_to_generate: 9       # Predict 9 future frames
  
model:
  sigma: 0.0000001           # Small noise for stability
  vector_field_regressor:
    state_size: 256          # Matches VQVAE latent channels
    state_res: [8, 8]        # Latent spatial resolution
    depth: 4                 # Temporal transformer layers
```

#### Training Command

```bash
python train.py \
    --run-name smoke_flow_v1 \
    --config configs/smoke_dataset.yaml \
    --num-gpus 1 \
    --random-seed 42 \
    --wandb
```

### Stage 4: Inference (Generating Smoke Predictions)

#### Generation Process
```
Given: First 10 observed smoke frames {x₁, ..., x₁₀}

For each future frame T:
  1. Encode observations with VQVAE: {z₁, ..., z₁₀}
  2. Sample initial noise: z^T₀ ~ N(0, 1)
  3. Run ODE integration (Euler or better solver):
     - For i = 0 to N-1:
       - Sample context: c ~ {1, ..., T-2}
       - Query model: v_t(z^T_i | z^{T-1}, z^c, T-c)
       - Update: z^T_{i+1} = z^T_i + (1/N) * v_t
  4. Decode: x^T = VQVAE_decoder(z^T_1)
  5. Use x^T as new reference for T+1
```

#### Warm-Start Sampling (RIVER Trick)
Instead of starting from pure noise z₀ ~ N(0,1), start from noisy previous frame:
```
z′_s ~ p_s(z | z^{T-1})  where s ∈ [0, 1]
```
- **s=0**: Start from noise (slower, more diverse)
- **s=0.1-0.3**: Good balance (faster, stable)
- **s→1**: Start near previous frame (fastest, less variation)

This is particularly useful for smoke because consecutive frames are similar!

### Stage 5: Evaluation

#### Metrics
- **FVD (Fréchet Video Distance)**: Video quality
- **MSE/PSNR**: Pixel-level accuracy
- **SSIM**: Structural similarity
- **Visual inspection**: Most important for smoke!

#### Test Command
```bash
python test_model.py
```

## Key Insights for Smoke Prediction

### Why This Works Well for Smoke

1. **Temporal Smoothness**: Smoke evolves gradually
   - Warm-start sampling is very effective
   - Small timestep changes capture smoke motion

2. **Latent Space Benefits**: 
   - 8×8×256 latent is compact but expressive
   - Smooth interpolation in latent space
   - Removes pixel-level noise

3. **Distributed Conditioning**:
   - Reference frame (z^{τ-1}) captures immediate motion
   - Context frame (z^c) captures long-term patterns
   - Model learns both local and global smoke dynamics

4. **No Filtering Needed**:
   - Flow matching works on any data distribution
   - Learns to predict smoke regardless of density
   - Model naturally handles sparse and dense smoke

### Training Tips

1. **VQVAE Quality Matters**:
   - Ensure VQVAE reconstructs smoke well
   - Test with: `python visualize_vqvae_reconstruction.py`
   - If reconstructions are poor, flow matching will fail

2. **Check Latent Statistics**:
   ```python
   # Verify latent space properties
   with torch.no_grad():
       latents = vqvae.encode(smoke_batch)
       print(f"Mean: {latents.mean():.3f}, Std: {latents.std():.3f}")
       # Should be roughly N(0, 1) distributed
   ```

3. **Monitor Loss**:
   - Flow matching loss should steadily decrease
   - Typical range: 0.01 - 0.1 after training
   - If stuck high (>1.0), check VQVAE or learning rate

4. **Inference Tuning**:
   - Start with N=50 integration steps
   - Try warm-start s=0.1 for 2x speedup
   - Generate short sequences first (3-5 frames)

## File Structure

```
├── train.py                    # Main training entry
├── training/
│   ├── trainer.py             # Flow matching training loop
│   └── training_loop.py       # Setup and orchestration
├── model/
│   ├── model.py               # RIVER model (encode + vector field)
│   └── vector_field_regressor.py  # Transformer for v_t
├── dataset/
│   └── text_based_video_dataset.py  # Loads smoke sequences
├── configs/
│   └── smoke_dataset.yaml     # All hyperparameters
└── final_dataset/             # Your smoke frames
    ├── train/
    ├── val/
    └── test/
```

## Troubleshooting

### Issue: "Reconstructions look blurry"
**Solution**: VQVAE needs more training or higher capacity

### Issue: "Generated smoke doesn't move"
**Solution**: 
- Increase `depth` in vector_field_regressor
- Check that sequences have temporal variation
- Lower learning rate

### Issue: "Training unstable / NaN losses"
**Solution**:
- Increase `sigma` from 1e-7 to 1e-6
- Reduce learning rate
- Check VQVAE outputs aren't NaN

### Issue: "Slow generation"
**Solution**:
- Use warm-start sampling (s=0.2)
- Reduce integration steps (N=25)
- Pre-cache latents (use LatentWrapperDataset)

## Next Steps

1. **Train Flow Matching**: Run `python train.py ...` with current config
2. **Monitor**: Watch WandB for loss curves and generated samples
3. **Evaluate**: After training, run `python test_model.py`
4. **Iterate**:
   - Adjust `num_observations` if sequences too short/long
   - Tune `depth` for model capacity
   - Try warm-start sampling at inference

## Expected Results

After successful training:
- **Training loss**: Should converge to ~0.01-0.05
- **Generated smoke**: Should follow plausible physics
- **Temporal consistency**: Smooth frame-to-frame transitions
- **Long-term**: May drift from reality but stay realistic

Remember: The goal is to predict **plausible smoke motion**, not exact ground truth!
