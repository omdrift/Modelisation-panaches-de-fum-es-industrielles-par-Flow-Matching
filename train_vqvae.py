# test_vqvae.py
import torch
from model.vqgan.vqvae import build_vqvae
from lutils.dict_wrapper import DictWrapper
import imageio
import cv2
import numpy as np

config = DictWrapper({
    "encoder": {"in_channels": 3, "out_channels": 4, "mid_channels": 32},
    "decoder": {"in_channels": 4, "out_channels": 3, "mid_channels": 32},
    "vector_quantizer": {"embedding_dimension": 4, "num_embeddings": 16384, "commitment_cost": 0.25}
})

vqvae = build_vqvae(config, convert_to_sequence=False)
vqvae.load_from_ckpt("runs/vqvae_smoke_vqvae/vqvae_final.ckpt")
vqvae.eval()

# Charger une frame à résolution native
reader = imageio.get_reader("braddock1_2019-02-03_frame7452_3.mp4")
frame_original = reader.get_data(0)
reader.close()

# Downscale pour le modèle (64x64)
frame_64 = cv2.resize(frame_original, (64, 64), interpolation=cv2.INTER_CUBIC)
tensor = torch.from_numpy(frame_64).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# Encode + Decode
with torch.no_grad():
    encoded = vqvae.encode(tensor)
    reconstructed = vqvae.decode_from_latents(encoded)

# Convertir en numpy
original_64 = (tensor[0].permute(1,2,0).numpy() * 255).astype('uint8')
recon_64 = (reconstructed[0].permute(1,2,0).numpy().clip(0, 1) * 255).astype('uint8')

# Upscale ÉNORME pour visualiser (512x512)
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

original_native_big = add_label(original_native_big, "Original 180x180")
original_big = add_label(original_big, "Downscaled 64x64")
recon_big = add_label(recon_big, "VQVAE Reconstructed")

# Créer image de comparaison side-by-side
side_by_side = np.concatenate([original_native_big, original_big, recon_big], axis=1)

# Sauvegarder en très haute qualité
cv2.imwrite("vqvae_test.png", cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR), 
            [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(f"Saved vqvae_test.png ({scale*3}x{scale})")
print(f"  • Left:   Original native (180x180 → {scale}x{scale})")
print(f"  • Middle: Original downscaled (64x64 → {scale}x{scale})")
print(f"  • Right:  VQVAE reconstruction (64x64 → {scale}x{scale})")
print(f"Reconstruction range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")