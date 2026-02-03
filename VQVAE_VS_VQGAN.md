# VQVAE vs VQGAN Implementation

## Overview

This project now includes **both VQVAE and VQGAN** implementations for smoke plume modeling. Here's what you need to know about each:

---

## 🔷 VQVAE (Vector Quantized VAE)

### What it is:
A simpler autoencoder with vector quantization. Uses only reconstruction loss and quantization loss.

### Key Components:
- **Encoder**: Compresses images to continuous latents
- **Vector Quantizer**: Discretizes latents using a learned codebook
- **Decoder**: Reconstructs images from quantized latents

### Loss Function:
```
Total Loss = Reconstruction Loss + VQ Loss
```

- **Reconstruction Loss**: MSE between original and reconstructed images (with masking for smoke)
- **VQ Loss**: Commitment loss + codebook loss

### Training Script:
```bash
python train_vqvae.py --config configs/config_vqvae.yaml --run-name my_vqvae --batch-size 64 --epochs 100
```

### Advantages:
- ✅ Simpler to train (no adversarial dynamics)
- ✅ More stable convergence
- ✅ Faster training (single optimizer)
- ✅ No mode collapse issues
- ✅ Good for downstream tasks (Flow Matching)

### Disadvantages:
- ❌ May produce blurry reconstructions
- ❌ Less perceptually accurate
- ❌ Can miss fine texture details

---

##  VQGAN (Vector Quantized GAN)

### What it is:
An enhanced version of VQVAE that adds adversarial training and perceptual loss for sharper, more realistic reconstructions.

### Key Components:
- **Encoder** + **Vector Quantizer** + **Decoder** (same as VQVAE)
- **Discriminator**: PatchGAN that distinguishes real from reconstructed images
- **Perceptual Loss**: Uses pretrained VGG16 to compare feature representations

### Loss Function:
```
Generator Loss = Reconstruction Loss + Perceptual Loss + VQ Loss + Adversarial Loss
Discriminator Loss = Hinge Loss (real vs fake)
```

- **Reconstruction Loss**: MSE with smoke masking (same as VQVAE)
- **Perceptual Loss**: LPIPS-style loss using VGG16 features
- **VQ Loss**: Quantization losses (same as VQVAE)
- **Adversarial Loss**: Generator tries to fool discriminator
- **Discriminator Loss**: Hinge loss to distinguish real/fake

### Training Script:
```bash
python train_vqgan.py --config configs/config_vqgan.yaml --run-name my_vqgan \
    --batch-size 32 --epochs 100 --lr 1e-4 --lr-disc 1e-4 \
    --perceptual-weight 1.0 --adversarial-weight 0.5 --disc-start 10000
```

### Key Arguments:
- `--lr`: Generator learning rate
- `--lr-disc`: Discriminator learning rate
- `--perceptual-weight`: Weight for perceptual loss (default: 1.0)
- `--adversarial-weight`: Weight for adversarial loss (default: 0.5)
- `--disc-start`: Steps before starting discriminator (default: 10000)
- `--disc-weight-max`: Maximum discriminator loss weight (default: 0.75)

### Advantages:
- ✅ Sharper reconstructions
- ✅ Better perceptual quality
- ✅ Preserves fine texture details (important for smoke)
- ✅ More realistic outputs
- ✅ Better for visual quality metrics

### Disadvantages:
- ❌ More complex training (two optimizers)
- ❌ Requires careful hyperparameter tuning
- ❌ Risk of mode collapse if not balanced
- ❌ Slower training (discriminator overhead)
- ❌ Requires more GPU memory

---

## 📊 Comparison Table

| Feature | VQVAE | VQGAN |
|---------|-------|-------|
| **Loss Components** | 2 (Recon + VQ) | 4 (Recon + Perceptual + VQ + Adversarial) |
| **Optimizers** | 1 (Generator) | 2 (Generator + Discriminator) |
| **Training Stability** | High | Medium |
| **Reconstruction Quality** | Good | Excellent |
| **Perceptual Quality** | Good | Excellent |
| **Training Speed** | Fast | Slower |
| **GPU Memory** | Lower | Higher |
| **Best For** | Latent representations, downstream tasks | High-quality reconstructions, visual fidelity |

---

## 🎯 Which One Should You Use?

### Use **VQVAE** if:
- You primarily need latent representations for Flow Matching
- You want stable, fast training
- You have limited GPU resources
- You're doing initial experiments
- Reconstruction quality is "good enough"

### Use **VQGAN** if:
- You need high-quality, sharp reconstructions
- Visual fidelity is critical
- You want to preserve fine smoke textures
- You have sufficient GPU memory and time
- You're willing to tune hyperparameters carefully

---

## 🔧 Implementation Details

### File Structure:
```
model/vqgan/
├── vqvae.py              # Generator (encoder + decoder + VQ)
├── encoder.py            # Encoder architecture
├── decoder.py            # Decoder architecture
├── vector_quantizer.py   # Vector quantization layer
├── discriminator.py      # PatchGAN discriminator (VQGAN only)
├── losses.py             # Perceptual loss (VQGAN only)
└── utils.py              # Utility functions

train_vqvae.py            # VQVAE training script
train_vqgan.py            # VQGAN training script

configs/
├── config_vqvae.yaml     # VQVAE configuration
└── config_vqgan.yaml     # VQGAN configuration
```

### Architecture Specifications:

**Encoder:**
- Input: 64×64 RGB images
- Residual blocks with downsampling (64 → 32 → 16 → 8)
- Channel progression: 128 → 256 → 512 → 1024
- Self-attention at 8×8 resolution
- Output: 256-channel latents at 8×8

**Decoder:**
- Input: 256-channel quantized latents at 8×8
- Self-attention at 8×8 resolution
- Residual blocks with upsampling (8 → 16 → 32 → 64)
- Channel progression: 128 → 64 → 32 → 16
- Output: 64×64 RGB images (tanh activation)

**Vector Quantizer:**
- Codebook size: 1024 embeddings
- Embedding dimension: 256
- Commitment cost: 0.1

**Discriminator (VQGAN only):**
- PatchGAN architecture (3 layers)
- Base filters: 64
- Receptive field: ~70×70 patches
- Output: Single-channel prediction map

**Perceptual Loss (VQGAN only):**
- VGG16 features from layers: relu1_2, relu2_2, relu3_3, relu4_3
- Equal weighting across layers
- Frozen VGG weights

---

## 🚀 Training Tips

### For VQVAE:
1. Start with batch size 64 for good codebook utilization
2. Use masked loss to focus on smoke regions
3. Monitor codebook usage (should use most codes)
4. Train for 50-100 epochs

### For VQGAN:
1. Start discriminator after ~10k steps (let generator learn first)
2. Use smaller batch size (32) due to memory overhead
3. Balance perceptual and adversarial weights carefully
4. Monitor discriminator loss - shouldn't dominate generator
5. Check for mode collapse (all outputs look similar)
6. Train for 50-100 epochs with careful monitoring

### General:
- Use WandB logging to track all losses
- Save checkpoints every 5 epochs
- Evaluate on validation set every epoch
- Use a fixed test image to track qualitative progress
- Monitor VQ loss - should decrease and stabilize

---

## 📈 Expected Results

### VQVAE:
- Reconstruction loss: ~0.001-0.005
- VQ loss: ~0.1-0.3
- Training time: ~6-8 hours (100 epochs, batch_size=64, single GPU)

### VQGAN:
- Reconstruction loss: ~0.001-0.003
- Perceptual loss: ~0.5-1.5
- VQ loss: ~0.1-0.3
- Generator loss: ~0.002-0.006
- Discriminator loss: ~0.5-2.0
- Training time: ~10-15 hours (100 epochs, batch_size=32, single GPU)

---

## 🔗 Integration with Flow Matching

Both VQVAE and VQGAN produce the same latent space format (8×8 with 256 channels). However, your Flow Matching model expects 4 channels. You need to either:

1. **Option A**: Add a projection layer in the encoder output (256 → 4 channels)
2. **Option B**: Update Flow Matching to accept 256 channels
3. **Option C**: Train a new VQVAE/VQGAN with 4-channel latents

Recommended: **Option A** - Add a learnable projection layer that compresses 256 → 4 channels while preserving information.

---

## 📚 References

- **VQVAE**: [Neural Discrete Representation Learning (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)
- **VQGAN**: [Taming Transformers for High-Resolution Image Synthesis (Esser et al., 2021)](https://arxiv.org/abs/2012.09841)
- **LPIPS**: [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (Zhang et al., 2018)](https://arxiv.org/abs/1801.03924)
