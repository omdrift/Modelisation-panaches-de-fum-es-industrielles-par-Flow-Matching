#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import imageio
from lutils.configuration import Configuration
from model import Model
from torchvision import transforms

# --- Chemins spécifiques ---
VIDEO_PATH = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/smoke_videos/55710_0-7-2019-03-14-3544-899-4026-1381-180-180-9603-1552594030-1552594205.mp4"
VQVAE_CKPT = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/runs/vqvae_quicktestv1/vqvae_epoch_10.ckpt"
FLOW_CKPT = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/runs/smoke_dataset_run-flow_run1/checkpoints/final_step_16800.pth"
CONFIG_PATH = "configs/smoke_dataset.yaml" # Assure-toi que le chemin vers ta config est correct
OUTPUT_NAME = "resultat_fumee_reelle.mp4"
OUTPUT_DIR = "runs/test_results_flow_run1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "frames_original"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "frames_predicted"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "frames_fg_original"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "frames_fg_predicted"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "side_by_side"), exist_ok=True)

def load_checkpoints(model, flow_path, vqvae_path):
    # Chargement Flow Matching
    print(f"Chargement Flow Matching: {flow_path}")
    ckpt = torch.load(flow_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    
    # Chargement VQ-VAE (dans le sous-module du modèle)
    print(f"Chargement VQ-VAE: {vqvae_path}")
    vq_ckpt = torch.load(vqvae_path, map_location="cpu")
    # On cherche où est le vqvae dans ton objet Model (souvent model.ae ou model.vqvae)
    target_vq = model.ae if hasattr(model, 'ae') else model.vqvae

    # Extract actual state_dict from checkpoint
    sd = None
    if isinstance(vq_ckpt, dict):
        if 'model' in vq_ckpt:
            sd = vq_ckpt['model']
        elif 'state_dict' in vq_ckpt:
            sd = vq_ckpt['state_dict']
        else:
            sd = vq_ckpt
    else:
        sd = vq_ckpt

    # If wrapped (DataParallel) keys start with 'module.', strip it
    def strip_module_prefix(d):
        return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in d.items()}

    sd = strip_module_prefix(sd)

    # Try direct load first
    try:
        target_vq.load_state_dict(sd)
        print("VQ-VAE loaded with strict=True")
        return
    except Exception as e:
        print(f"Direct load failed: {e}")

    # Helper to remap prefixes
    def remap_prefix(state_dict, prefix_from, prefix_to):
        new = {}
        for k, v in state_dict.items():
            if k.startswith(prefix_from):
                new_key = prefix_to + k[len(prefix_from):]
            else:
                new_key = k
            new[new_key] = v
        return new

    target_keys = set(target_vq.state_dict().keys())
    ck_keys = set(sd.keys())

    # Try adding 'backbone.' prefix (common mismatch)
    attempt = remap_prefix(sd, '', 'backbone.')
    try:
        target_vq.load_state_dict(attempt)
        print("VQ-VAE loaded by adding 'backbone.' prefix to checkpoint keys")
        return
    except Exception:
        pass

    # Try removing 'backbone.' if present in ckpt
    if any(k.startswith('backbone.') for k in sd.keys()):
        attempt2 = remap_prefix(sd, 'backbone.', '')
        try:
            target_vq.load_state_dict(attempt2)
            print("VQ-VAE loaded by removing 'backbone.' prefix from checkpoint keys")
            return
        except Exception:
            pass

    # Last resort: load with strict=False and report missing/unexpected keys
    try:
        res = target_vq.load_state_dict(sd, strict=False)
        print("VQ-VAE loaded with strict=False")
        print("Missing keys:", res.missing_keys)
        print("Unexpected keys:", res.unexpected_keys)
    except Exception as e:
        print("Failed to load VQ-VAE checkpoint:", e)
        raise

def process_video_and_bg(path, img_size):
    cap = cv2.VideoCapture(path)
    frames_raw = []
    frames_for_model = []
    masks = []
    while len(frames_raw) < 100:
        ret, frame = cap.read()
        if not ret: break
        frames_raw.append(frame) # BGR original
        
        # Keep colors in RGB (cv2 returns BGR)
        bad_colors = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bad_colors = cv2.resize(bad_colors, (img_size, img_size))
        # store in [0,1] for now; will convert to [-1,1] before model
        frames_for_model.append(bad_colors.astype(np.float32) / 255.0)
        # placeholder mask (computed after background estimation)
        masks.append(None)
    cap.release()
    
    # Background propre (médiane sur BGR original)
    indices = np.linspace(0, len(frames_raw)-1, min(len(frames_raw), 30), dtype=int)
    background = np.median(np.stack([frames_raw[i] for i in indices]), axis=0).astype(np.uint8)

    # Compute foreground masks by background subtraction
    for i, frame in enumerate(frames_raw):
        diff = cv2.absdiff(frame, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks[i] = mask
    
    return frames_raw, frames_for_model, background, masks

def main():
    config = Configuration(CONFIG_PATH)
    img_size = config["data"].get("input_size", 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialisation Modèle
    model = Model(config["model"])
    load_checkpoints(model, FLOW_CKPT, VQVAE_CKPT)
    model.to(device).eval()

    # 2. Préparation Data
    raw_bgr, model_input, bg_bgr, masks = process_video_and_bg(VIDEO_PATH, img_size)
    h_orig, w_orig = raw_bgr[0].shape[:2]

    # Tensor pour l'IA [1, K, C, H, W]
    first_k = 10 
    # Apply masks to create foreground inputs (outside mask -> black)
    fg_inputs = []
    for i in range(min(first_k, len(model_input))):
        m = masks[i]
        # ensure mask matches model input resolution
        if m is None:
            mask_resized = np.zeros((img_size, img_size), dtype=np.uint8)
        else:
            mask_resized = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask_norm = (mask_resized.astype(np.float32) / 255.0)[..., None]
        fg = model_input[i] * mask_norm
        fg_inputs.append(fg)

    if len(fg_inputs) == 0:
        raise RuntimeError('No frames to initialize from video')

    # Convert to tensor and to model expected range [-1,1]
    init_np = np.stack(fg_inputs)  # [K, H, W, C] in [0,1]
    init_np = init_np * 2.0 - 1.0
    init_tensor = torch.from_numpy(init_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    # 3. Génération
    print("Génération de la fumée...")
    with torch.no_grad():
        # Génère 30 frames après les 10 premières
        generated = model.generate_frames(init_tensor, num_frames=30)
        full_seq = torch.cat([init_tensor, generated], dim=1)[0] # [T, C, H, W]

    # 4. Recomposition + save frames
    final_video = []
    side_by_side_frames = []
    print("Correction couleurs et fusion sur background...")

    for t in range(full_seq.shape[0]):
        # Sortie IA (en float, couleurs inversées)
        img_ia = full_seq[t].permute(1, 2, 0).cpu().numpy()
        img_ia = (np.clip(img_ia, 0, 1) * 255).astype(np.uint8)
        
        # img_ia is already RGB (0-255)
        smoke_res = cv2.resize(img_ia, (w_orig, h_orig))

        # Original frame (RGB) for t if exists
        if t < len(raw_bgr):
            orig_bgr = raw_bgr[t]
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        else:
            orig_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

        # Save original frame
        orig_path = os.path.join(OUTPUT_DIR, "frames_original", f"frame_{t:04d}.png")
        imageio.imwrite(orig_path, orig_rgb)

        # Déterminer le masque pour cette image (redimensionner le masque existant
        # ou le calculer à partir de la prédiction si absent)
        if t < len(masks) and masks[t] is not None:
            mask = masks[t]
        else:
            # smoke_res est en RGB (0-255)
            gray_smoke = cv2.cvtColor(smoke_res, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray_smoke, 10, 255, cv2.THRESH_BINARY)

        # Assurer que le masque a la même taille que le background
        if mask.shape != bg_bgr.shape[:2]:
            mask = cv2.resize(mask, (bg_bgr.shape[1], bg_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convertir la fumée prédite en BGR pour les opérations OpenCV
        smoke_bgr = cv2.cvtColor(smoke_res, cv2.COLOR_RGB2BGR)

        # Composite predicted over static background
        mask_inv = cv2.bitwise_not(mask)
        bg_part = cv2.bitwise_and(bg_bgr, bg_bgr, mask=mask_inv)
        fg_part = cv2.bitwise_and(smoke_bgr, smoke_bgr, mask=mask)
        final_frame = cv2.add(bg_part, fg_part)

        # Save predicted composite full frame (RGB)
        pred_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        pred_path = os.path.join(OUTPUT_DIR, "frames_predicted", f"pred_{t:04d}.png")
        imageio.imwrite(pred_path, pred_rgb)

        # --- FUSION (utiliser masque si disponible sinon threshold sur prédiction) ---
        if t < len(masks):
            mask = masks[t]
        else:
            gray_smoke = cv2.cvtColor(smoke_res, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_smoke, 10, 255, cv2.THRESH_BINARY)

        # Save foreground original and predicted (both RGB)
        orig_fg = cv2.bitwise_and(orig_rgb, orig_rgb, mask=mask)
        fg_orig_path = os.path.join(OUTPUT_DIR, "frames_fg_original", f"fg_orig_{t:04d}.png")
        imageio.imwrite(fg_orig_path, orig_fg)

        fg_pred = cv2.bitwise_and(pred_rgb, pred_rgb, mask=mask)
        fg_pred_path = os.path.join(OUTPUT_DIR, "frames_fg_predicted", f"fg_pred_{t:04d}.png")
        imageio.imwrite(fg_pred_path, fg_pred)

        # final_video expects RGB frames
        final_video.append(pred_rgb)

        # Side-by-side: original | predicted composite
        sbs = np.concatenate([orig_rgb, pred_rgb], axis=1)
        sbs_path = os.path.join(OUTPUT_DIR, "side_by_side", f"sbs_{t:04d}.png")
        imageio.imwrite(sbs_path, sbs)
        side_by_side_frames.append(sbs)

    # 5. Sauvegarde
    # Save composite video (original background + predicted foreground)
    imageio.mimsave(OUTPUT_NAME, final_video, fps=15)
    # Save side-by-side comparison video
    sbs_out = os.path.join(OUTPUT_DIR, "comparison_side_by_side.mp4")
    imageio.mimsave(sbs_out, side_by_side_frames, fps=15)

    print(f"Terminé ! Vidéo sauvegardée sous : {OUTPUT_NAME}")
    print(f"Side-by-side saved: {sbs_out}")

if __name__ == "__main__":
    main()