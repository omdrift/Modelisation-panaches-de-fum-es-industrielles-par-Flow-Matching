# Tailoring Flow Matching for Smoke Prediction

## Problem Characteristics

**Data**: Smoke foreground on black background
- Sparse signal: most pixels are black (background)
- Dynamic content: smoke is transparent, wispy, and evolves over time
- Low contrast: smoke often blends with background
- Temporal coherence: smoke motion is smooth but complex

## Key Adaptations from RIVER Algorithm

### 1.  Smoke Mask Weighting (IMPLEMENTED)

**Location**: [model/model.py](model/model.py#L108-L111) and [training/trainer.py](training/trainer.py#L203-L215)

```python
# In model.py
smoke_threshold = self.config.get("smoke_threshold", 0.1)
smoke_mask = (target_latents.norm(dim=1, keepdim=True) > smoke_threshold).float()

# In trainer.py - calculate_loss()
smoke_weight = self.config.get("smoke_weight", 5.0)
bg_weight = self.config.get("background_weight", 1.0)
weight_map = results.smoke_mask * smoke_weight + (1 - results.smoke_mask) * bg_weight
weighted_mse = mse_per_pixel * weight_map
flow_matching_loss = weighted_mse.mean()
```

**Configuration**: [configs/smoke_dataset_vqgan.yaml](configs/smoke_dataset_vqgan.yaml#L52-L54)
```yaml
smoke_weight: 5.0       # 5x more important than background
background_weight: 1.0  # Base weight
smoke_threshold: 0.1    # Latent norm threshold to detect smoke
```

**Why this matters**: Without weighting, 80-90% of the loss comes from predicting black pixels correctly, which teaches the model to output black everywhere. With weighting, we force it to learn smoke .

### 2.  Calibrate Smoke Threshold

The threshold `0.1` may not be optimal for your VQGAN encoding. To find the best threshold:

```bash
# Analyze your data to find optimal threshold
python visualize_smoke_threshold.py --data-root final_dataset --split train --num-samples 500
```

This will show you:
- Distribution of latent norms across your dataset
- Samples at different thresholds (0.05, 0.1, 0.15, 0.2)
- Recommended threshold based on smoke presence

**Adjust in config based on results**:
```yaml
smoke_threshold: 0.08  # If smoke is subtle
smoke_threshold: 0.15  # If only strong smoke matters
```

### 3. 🎯 Temporal Conditioning Strategy

**RIVER uses**: Reference frame (τ-1) + Random context frame (c < τ-2)

**For smoke**, this is optimal because:
- **Reference frame (τ-1)**: Captures immediate smoke position/shape
- **Context frame (c)**: Captures smoke source location and wind direction
- **Gap (τ-c)**: Network learns to infer motion velocity

**Your config** ([smoke_dataset_vqgan.yaml](configs/smoke_dataset_vqgan.yaml#L47-L48)):
```yaml
num_observations: 10    # See 10 frames of smoke evolution
condition_frames: 1     # Use reference + 1 random context
```

**Recommendation**: Keep this as-is. RIVER's distributed conditioning is specifically designed for motion prediction.

### 4. 🚀 Warm-Start Sampling (Optional Speedup)

From RIVER paper (Figure 3), you can start sampling from t=0.1 instead of t=0:

**Add to model.py** in `generate_frames()`:
```python
def generate_frames(self, observations, num_frames, steps=100, warm_start=0.0):
    # ...
    for frame_index in range(num_frames):
        if warm_start > 0:
            # Start from t=warm_start instead of t=0
            zt = torch.randn_like(reference_latents) * (1 - warm_start)
            t_values = torch.linspace(warm_start, 1.0, steps)
        else:
            zt = torch.randn_like(reference_latents)
            t_values = torch.linspace(0, 1.0, steps)
```

**Usage**:
```python
# During inference
generated = model.generate_frames(
    observations=obs,
    num_frames=6,
    steps=50,        # Reduced from 100
    warm_start=0.1   # Skip noisy initial steps
)
```

**Trade-off**: `warm_start=0.1` speeds up by 10% with minimal quality loss (may even improve FVD per RIVER paper).

### 5. 📊 Smoke-Specific Metrics

**Add to trainer.py** logging:

```python
def log_scalars(self, auxiliary_output, additional_info, logger):
    # ... existing code ...
    
    # Log smoke-specific metrics
    if 'smoke_ratio' in auxiliary_output:
        logger.log_scalar('train/smoke_ratio', auxiliary_output['smoke_ratio'])
    
    if 'smoke_loss' in auxiliary_output and 'bg_loss' in auxiliary_output:
        logger.log_scalar('train/smoke_loss', auxiliary_output['smoke_loss'])
        logger.log_scalar('train/bg_loss', auxiliary_output['bg_loss'])
        logger.log_scalar('train/loss_ratio', 
                         auxiliary_output['smoke_loss'] / (auxiliary_output['bg_loss'] + 1e-8))
```

**Update calculate_loss()** to return these:
```python
def calculate_loss(self, results):
    # ... existing code ...
    
    # Separate smoke and background losses
    smoke_loss = (mse_per_pixel * results.smoke_mask).sum() / (results.smoke_mask.sum() + 1e-8)
    bg_loss = (mse_per_pixel * (1 - results.smoke_mask)).sum() / ((1 - results.smoke_mask).sum() + 1e-8)
    
    auxiliary_output = DictWrapper(
        loss=loss.item(),
        flow_matching_loss=flow_matching_loss.item(),
        smoke_ratio=smoke_ratio.item(),
        smoke_loss=smoke_loss.item(),
        bg_loss=bg_loss.item()
    )
    
    return loss, auxiliary_output
```

### 6. 🎨 Data Augmentation for Smoke

**Current augmentation** ([configs/smoke_dataset_vqgan.yaml](configs/smoke_dataset_vqgan.yaml#L8)):
```yaml
random_horizontal_flip: True
```

**Additional smoke-specific augmentations** (add to dataset class):

```python
class SmokeAugmentation:
    def __init__(self, brightness_range=0.2, contrast_range=0.2):
        self.brightness = brightness_range
        self.contrast = contrast_range
    
    def __call__(self, frames):
        # Brightness: smoke can be lighter/darker
        if random.random() > 0.5:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            frames = frames * factor
        
        # Contrast: enhance/reduce smoke visibility
        if random.random() > 0.5:
            mean = frames.mean()
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            frames = (frames - mean) * factor + mean
        
        # NO ROTATION: smoke source position matters!
        # NO VERTICAL FLIP: smoke rises upward
        
        return frames.clamp(0, 1)
```

**Why these matter**:
- ✅ Horizontal flip: OK, smoke source can be on left or right
- ✅ Brightness/contrast: Smoke opacity varies with lighting
- ❌ Rotation: NO - smoke source position is fixed (e.g., chimney location)
- ❌ Vertical flip: NO - smoke always rises upward (physics!)

### 7. 🔍 Monitor Training Health

**Key metrics to watch**:

1. **Smoke Ratio**: Should be 10-30% on average
   - If < 5%: Your threshold is too high
   - If > 50%: Your threshold is too low or images aren't properly extracted

2. **Loss Ratio (smoke_loss / bg_loss)**:
   - Should be > 1.0 initially (smoke is harder to predict)
   - Should converge to ~1.0-2.0 (balanced learning)
   - If stays > 5.0: Increase `smoke_weight` or lower threshold

3. **Generated samples**:
   - Check if smoke appears in generated frames
   - Check if smoke motion is smooth
   - Check if background stays black

### 8. 🎯 Hyperparameter Recommendations

Based on your 64x64 VQGAN and smoke data:

```yaml
# RECOMMENDED CONFIG
training:
  batching:
    batch_size: 24              # Good for 64x64
  
  # Smoke weighting
  smoke_weight: 5.0             # Start here
  background_weight: 1.0        
  smoke_threshold: 0.1          # Calibrate with visualize_smoke_threshold.py
  
  # Temporal context
  num_observations: 10          # ✅ Good: captures smoke evolution
  condition_frames: 1           # ✅ Good: reference + context (RIVER default)
  frames_to_generate: 6         # ✅ Good: predict 6 frames ahead
  
model:
  sigma: 0.001                  # ✅ Good: stable for 64x64 latents
  vector_field_regressor:
    state_size: 256             # ✅ Matches VQGAN embedding
    state_res: [8, 8]           # ✅ Matches 64→8x8 compression
    inner_dim: 512              # ✅ Good capacity for smoke dynamics
    depth: 6                    # ✅ Good temporal modeling

evaluation:
  steps: 100                    # Can try 50 with warm_start=0.1
```

### 9. 🐛 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| All black outputs | Threshold too high or weight too low | Lower `smoke_threshold` to 0.05, increase `smoke_weight` to 10.0 |
| Smoke everywhere | Threshold too low | Increase `smoke_threshold` to 0.15-0.2 |
| Jittery motion | Not enough temporal context | Increase `num_observations` to 16 |
| Blurry smoke | Not enough denoising steps | Increase `evaluation.steps` to 150 |
| Mode collapse | Learning rate too high | Reduce `learning_rate` to 5e-5 |

### 10. 📝 Training Command

```bash
# Full training with smoke-specific config
python train.py \
    --run-name flow_smoke_weighted_v1 \
    --config configs/smoke_dataset_vqgan.yaml \
    --num-gpus 1 \
    --random-seed 42 \
    --wandb
```

**Monitor in W&B**:
- `train/flow_matching_loss`: Should decrease steadily
- `train/smoke_ratio`: Should be stable (10-30%)
- `train/smoke_loss` vs `train/bg_loss`: Should converge
- Generated samples: Smoke should appear and move smoothly

## Summary

Your implementation already has the **most critical adaptation** (smoke mask weighting). Additional improvements:

1. ✅ **Already implemented**: Weighted loss, temporal conditioning, VQGAN encoding
2. 🔧 **Calibrate**: Run `visualize_smoke_threshold.py` to optimize threshold
3. 📊 **Monitor**: Add smoke-specific logging (smoke_loss, bg_loss, ratio)
4. 🚀 **Optimize**: Try `warm_start=0.1` for faster inference
5. 🎨 **Augment**: Add brightness/contrast (but NOT rotation/vertical flip)

The RIVER algorithm is well-suited for smoke because:
- Distributed conditioning captures wind/source information
- Flow matching handles sparse data better than diffusion
- Per-frame VQGAN avoids temporal artifacts in transparent smoke

Your main risk is **threshold calibration** - if it's wrong, the model either ignores smoke or hallucinates it everywhere. Use the visualization tool to verify!
