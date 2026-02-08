# test_vqgan_simple.py
import torch
from model.vqgan.vqvae import build_vqvae
from lutils.dict_wrapper import DictWrapper
import imageio
import cv2
import numpy as np
import glob
import os

# Configuration matching config_vqgan.yaml
config = DictWrapper({
    "encoder": {"in_channels": 3, "out_channels": 256, "mid_channels": 128},
    "decoder": {"in_channels": 256, "out_channels": 3, "mid_channels": 128},
    "vector_quantizer": {"embedding_dimension": 256, "num_embeddings": 1024, "commitment_cost": 0.1}
})

vqgan = build_vqvae(config, convert_to_sequence=False)
vqgan.load_from_ckpt("runs_vqgan/vqgan_smoke_vqgan_v2/vqgan_epoch_10.ckpt")
vqgan.eval()

# Charger frame originale depuis une vidéo (comme test_vqvae.py)
video_dir = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/smoke_videos"
video_files = []

# Chercher dans les sous-dossiers view_*
for view_folder in glob.glob(os.path.join(video_dir, "view_*")):
    video_files.extend(glob.glob(os.path.join(view_folder, "*.mp4")))

# Si pas de vidéos dans view_*, chercher directement
if not video_files:
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))

if not video_files:
    print("ERROR: No MP4 files found in smoke_videos!")
    exit(1)

# Prendre la première vidéo
video_path = video_files[0]
print(f"Testing on: {video_path}")

reader = imageio.get_reader(video_path)
frame_original = reader.get_data(0)
reader.close()

print(f"Original frame shape: {frame_original.shape}")

# Downscale à 64x64 pour le modèle
frame_64 = cv2.resize(frame_original, (64, 64), interpolation=cv2.INTER_CUBIC)

# Convertir en tensor [0, 1]
tensor = torch.from_numpy(frame_64).permute(2, 0, 1).unsqueeze(0).float() / 255.0

print(f"Input tensor shape: {tensor.shape}")
print(f"Input tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")

# ÉTAPE 1: Encoder - image → latents continus
with torch.no_grad():
    latents_continuous = vqgan.encoder(tensor)
    print(f"\nEncoded latents (continuous): {latents_continuous.shape}")
    
    # ÉTAPE 2: Quantizer - quantization vectorielle
    quantized_output = vqgan.vector_quantizer(latents_continuous)
    
    print(f"\nDEBUG - Type of output: {type(quantized_output)}")
    print(f"DEBUG - Number of outputs: {len(quantized_output)}")
    for i, out in enumerate(quantized_output):
        if hasattr(out, 'shape'):
            print(f"  Output[{i}] shape: {out.shape}, dtype: {out.dtype}")
        else:
            print(f"  Output[{i}] type: {type(out)}")
    
    # Format: (vq_loss, quantized_latents, quantized_latents_ids)
    vq_loss = quantized_output[0]
    latents_quantized = quantized_output[1]
    latents_ids = quantized_output[2]
    
    print(f"\nQuantized latents: {latents_quantized.shape}")
    print(f"Latents IDs: {latents_ids.shape}")
    print(f"VQ loss: {vq_loss.item():.6f}")
    print(f"Unique codes used: {torch.unique(latents_ids).numel()} / 1024")
    
    # ÉTAPE 3: Decoder - latents discrets → image reconstruite
    reconstructed = vqgan.decoder(latents_quantized)
    print(f"\nReconstructed shape: {reconstructed.shape}")
    print(f"Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

# Calculer MSE
mse = torch.mean((tensor - reconstructed) ** 2).item()
psnr = 10 * np.log10(1.0 / (mse + 1e-8))
print(f"\nMSE: {mse:.6f}")
print(f"PSNR: {psnr:.2f} dB")

# Convertir en numpy pour visualisation
original_64 = (tensor[0].permute(1,2,0).numpy() * 255).astype('uint8')
recon_64 = (reconstructed[0].permute(1,2,0).numpy().clip(0, 1) * 255).astype('uint8')

# Upscale à 512x512 pour bien voir
scale = 512
original_native_big = cv2.resize(frame_original, (scale, scale), interpolation=cv2.INTER_CUBIC)
original_big = cv2.resize(original_64, (scale, scale), interpolation=cv2.INTER_CUBIC)
recon_big = cv2.resize(recon_64, (scale, scale), interpolation=cv2.INTER_CUBIC)

# Ajouter labels
def add_label(img, text):
    img_copy = img.copy()
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    return img_copy

original_native_big = add_label(original_native_big, f"Original {frame_original.shape[0]}x{frame_original.shape[1]}")
original_big = add_label(original_big, "Downscaled 64x64")
recon_big = add_label(recon_big, f"VQGAN (MSE={mse:.4f})")

# Créer comparaison
side_by_side = np.concatenate([original_native_big, original_big, recon_big], axis=1)

# Sauvegarder
output_path = "vqgan_test.png"
cv2.imwrite(output_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR), 
            [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(f"\n✅ Saved {output_path} ({scale*3}x{scale})")
print(f"Video source: {os.path.basename(video_path)}")

# Visualiser les codes utilisés dans le codebook
unique_codes = torch.unique(latents_ids).cpu().numpy()
print(f"\nCodebook usage: {len(unique_codes)}/1024 codes ({len(unique_codes)/1024*100:.1f}%)")
print(f"Code range: [{unique_codes.min()}, {unique_codes.max()}]")

# Statistiques par canal
print("\n=== Per-channel statistics ===")
for i in range(3):
    orig_ch = tensor[0, i].flatten()
    recon_ch = reconstructed[0, i].flatten()
    print(f"Channel {i}: orig=[{orig_ch.min():.3f}, {orig_ch.max():.3f}], "
          f"recon=[{recon_ch.min():.3f}, {recon_ch.max():.3f}]")
