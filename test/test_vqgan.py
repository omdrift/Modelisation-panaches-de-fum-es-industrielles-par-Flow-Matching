# test_vqgan.py
# Simple test script for VQGAN model based on test_vqvae.py
import torch
from model.vqgan.vqvae import build_vqvae
from lutils.dict_wrapper import DictWrapper
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration for VQGAN
# NOTE: Update these parameters to match your VQGAN training config
config = DictWrapper({
    "encoder": {"in_channels": 3, "out_channels": 256, "mid_channels": 128},
    "decoder": {"in_channels": 256, "out_channels": 3, "mid_channels": 128},
    "vector_quantizer": {"embedding_dimension": 256, "num_embeddings": 1024, "commitment_cost": 0.1}
})

# Build VQGAN model (uses same architecture as VQVAE but trained with GAN loss)
vqgan = build_vqvae(config, convert_to_sequence=False)

#  Update this checkpoint path to your trained VQGAN checkpoint
checkpoint_path = "runs_vqgan/vqgan_smoke_vqgan_v2/vqgan_epoch_10.ckpt"
print(f"Loading VQGAN checkpoint: {checkpoint_path}")
vqgan.load_from_ckpt(checkpoint_path)
vqgan.eval()

# Load original frame from video
video_path = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/smoke_videos/view_0-0/6_0-0-2018-06-11-6304-964-6807-1467-180-180-3470-1528712115-1528712290.mp4"
reader = imageio.get_reader(video_path)
frame_original = reader.get_data(0)
reader.close()

# Downscale to 64x64 for the model
frame_64 = cv2.resize(frame_original, (64, 64), interpolation=cv2.INTER_CUBIC)
tensor = torch.from_numpy(frame_64).permute(2, 0, 1).unsqueeze(0).float() / 255.0

print(f"Input shape: {tensor.shape}")


# STEP 1: Encoder - image → continuous latents
with torch.no_grad():
    latents_continuous = vqgan.encoder(tensor)
    print(f"Encoded latents (continuous): {latents_continuous.shape}")
    
    # STEP 2: Vector Quantizer - continuous → discrete latents
    quantized_output = vqgan.vector_quantizer(latents_continuous)
    
    print(f"\nDEBUG - Type of output: {type(quantized_output)}")
    print(f"DEBUG - Number of outputs: {len(quantized_output)}")
    for i, out in enumerate(quantized_output):
        print(f"  Output[{i}] shape: {out.shape if hasattr(out, 'shape') else type(out)}")
    
    # Extract outputs: (loss, quantized_latents, perplexity/encodings)
    vq_loss = quantized_output[0]
    latents_quantized = quantized_output[1]
    
    print(f"\nQuantized latents: {latents_quantized.shape}")
    print(f"VQ loss shape: {vq_loss.shape}")
    print(f"VQ loss mean: {vq_loss.mean().item():.6f}")
    
    # STEP 3: Decoder - discrete latents → reconstructed image
    reconstructed = vqgan.decoder(latents_quantized)
    print(f"Reconstructed shape: {reconstructed.shape}")

# Convert to numpy and upscale for visualization
original_64 = (tensor[0].permute(1, 2, 0).numpy() * 255).astype('uint8')
recon_64 = (reconstructed[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype('uint8')

# Upscale to 512x512 for better visualization
scale = 512
original_native_big = cv2.resize(frame_original, (scale, scale), interpolation=cv2.INTER_CUBIC)
original_big = cv2.resize(original_64, (scale, scale), interpolation=cv2.INTER_CUBIC)
recon_big = cv2.resize(recon_64, (scale, scale), interpolation=cv2.INTER_CUBIC)

# Add labels to images
def add_label(img, text):
    img_copy = img.copy()
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    return img_copy

original_native_big = add_label(original_native_big, "Original 180x180")
original_big = add_label(original_big, "Downscaled 64x64")
recon_big = add_label(recon_big, "VQGAN Reconstructed")

# Create side-by-side comparison
side_by_side = np.concatenate([original_native_big, original_big, recon_big], axis=1)

# Save result
output_path = "vqgan_test.png"
cv2.imwrite(output_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR), 
            [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(f"\nSaved {output_path} ({scale*3}x{scale})")
print(f"Reconstruction range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

# Compute reconstruction metrics
mse = torch.mean((tensor - reconstructed) ** 2).item()
psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
print(f"\nReconstruction Metrics:")
print(f"MSE: {mse:.6f}")
print(f"PSNR: {psnr:.2f} dB")

print("\nTest completed successfully!")
