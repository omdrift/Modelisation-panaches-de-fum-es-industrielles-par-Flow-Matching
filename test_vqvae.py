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
vqvae.load_from_ckpt("/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/runs/vqvae_smoke_vqvae_test/vqvae_epoch_55.ckpt")
vqvae.eval()

# Charger frame originale
reader = imageio.get_reader("/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/smoke_videos/6_0-0-2018-06-11-6304-964-6807-1467-180-180-3470-1528712115-1528712290.mp4")
frame_original = reader.get_data(0)
reader.close()

# Downscale à 64x64 pour le modèle
frame_64 = cv2.resize(frame_original, (64, 64), interpolation=cv2.INTER_CUBIC)
tensor = torch.from_numpy(frame_64).permute(2, 0, 1).unsqueeze(0).float() / 255.0

print(f"Input shape: {tensor.shape}")

# ÉTAPE 1: Encoder - image → latents continus
with torch.no_grad():
    latents_continuous = vqvae.encoder(tensor)
    print(f"Encoded latents (continuous): {latents_continuous.shape}")
    
    # ÉTAPE 2: Quantizer - debug pour voir ce qui est retourné
    quantized_output = vqvae.vector_quantizer(latents_continuous)
    
    print(f"\nDEBUG - Type of output: {type(quantized_output)}")
    print(f"DEBUG - Number of outputs: {len(quantized_output)}")
    for i, out in enumerate(quantized_output):
        print(f"  Output[{i}] shape: {out.shape if hasattr(out, 'shape') else type(out)}")
    
    # Essayons différentes combinaisons
    # Souvent: (loss, quantized_latents, perplexity, encodings)
    vq_loss = quantized_output[0]
    latents_quantized = quantized_output[1]
    
    print(f"\nQuantized latents: {latents_quantized.shape}")
    print(f"VQ loss shape: {vq_loss.shape}")
    print(f"VQ loss mean: {vq_loss.mean().item():.6f}")
    
    # ÉTAPE 3: Decoder - latents discrets → image reconstruite
    reconstructed = vqvae.decoder(latents_quantized)
    print(f"Reconstructed shape: {reconstructed.shape}")

# Convertir en numpy et upscale pour visualisation
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

original_native_big = add_label(original_native_big, "Original 180x180")
original_big = add_label(original_big, "Downscaled 64x64")
recon_big = add_label(recon_big, "VQVAE Reconstructed")

# Créer comparaison
side_by_side = np.concatenate([original_native_big, original_big, recon_big], axis=1)

# Sauvegarder
cv2.imwrite("vqvae_test.png", cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR), 
            [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(f"\nSaved vqvae_test.png ({scale*3}x{scale})")
print(f"Reconstruction range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")