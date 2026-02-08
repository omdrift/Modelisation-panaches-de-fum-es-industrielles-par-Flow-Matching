#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import imageio
from lutils.configuration import Configuration
from model import Model
from torchvision import transforms


def _pad_to_block(img, block=16):
    """Pad image on right/bottom so both dimensions are divisible by `block`.
    Returns a contiguous uint8 array."""
    h, w = img.shape[:2]
    new_h = ((h + block - 1) // block) * block
    new_w = ((w + block - 1) // block) * block
    pad_bottom = new_h - h
    pad_right = new_w - w
    if pad_bottom == 0 and pad_right == 0:
        return np.ascontiguousarray(img)
    padded = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return np.ascontiguousarray(padded)

# --- Chemins spécifiques ---
VIDEO_PATH = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/smoke_videos/55710_0-7-2019-03-14-3544-899-4026-1381-180-180-9603-1552594030-1552594205.mp4"
VQVAE_CKPT = "runs/vqvae_quicktestv1/vqvae_epoch_10.ckpt"
FLOW_CKPT = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/runs/smoke_dataset_run-flow_run1/checkpoints/final_step_16800.pth"
CONFIG_PATH = "configs/smoke_dataset.yaml" # Assure-toi que le chemin vers ta config est correct
OUTPUT_NAME = "resultat_fume.mp4"
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

    target_keys = set(target_vq.state_dict().keys())
    ck_keys = set(sd.keys())

    has_backbone_target = any(k.startswith('backbone.') for k in target_keys)
    has_backbone_ckpt = any(k.startswith('backbone.') for k in ck_keys)

    # If the model expects 'backbone.' prefix but checkpoint doesn't, add it.
    if has_backbone_target and not has_backbone_ckpt:
        sd_prefixed = {('backbone.' + k): v for k, v in sd.items()}
        try:
            target_vq.load_state_dict(sd_prefixed)
            print("VQ-VAE loaded by adding 'backbone.' prefix to checkpoint keys")
            return
        except Exception as e:
            print(f"Load with added 'backbone.' prefix failed: {e}")

    # If the checkpoint has 'backbone.' but the model doesn't, strip it.
    if not has_backbone_target and has_backbone_ckpt:
        sd_stripped = { (k[len('backbone.'): ] if k.startswith('backbone.') else k): v for k, v in sd.items() }
        try:
            target_vq.load_state_dict(sd_stripped)
            print("VQ-VAE loaded by removing 'backbone.' prefix from checkpoint keys")
            return
        except Exception as e:
            print(f"Load with removed 'backbone.' prefix failed: {e}")

    # Fallback: try direct load first, then strict=False load
    try:
        target_vq.load_state_dict(sd)
        print("VQ-VAE loaded with strict=True")
        return
    except Exception as e:
        print("Direct load failed (will try non-strict). Error:", str(e).splitlines()[0])

    try:
        res = target_vq.load_state_dict(sd, strict=False)
        print("VQ-VAE loaded with strict=False")
        if getattr(res, 'missing_keys', None):
            print(f"Missing keys: {len(res.missing_keys)} (use verbose to list)")
        if getattr(res, 'unexpected_keys', None):
            print(f"Unexpected keys: {len(res.unexpected_keys)} (use verbose to list)")
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

#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import imageio
from lutils.configuration import Configuration
from model import Model

OUTPUT_DIR_FRAMES = "runs/generated_smoke_frames"

# --- FONCTIONS UTILITAIRES ---

def _pad_to_block(img, block=16):
    """Assure que les dimensions sont divisibles par 16 pour l'encodage vidéo."""
    h, w = img.shape[:2]
    new_h = ((h + block - 1) // block) * block
    new_w = ((w + block - 1) // block) * block
    if h == new_h and w == new_w: return img
    return cv2.copyMakeBorder(img, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def load_checkpoints(model, flow_path, vqvae_path):
    print(f"Chargement Flow Matching: {flow_path}")
    ckpt = torch.load(flow_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    
    print(f"Chargement VQ-VAE: {vqvae_path}")
    vq_ckpt = torch.load(vqvae_path, map_location="cpu")
    target_vq = model.ae if hasattr(model, 'ae') else model.vqvae
    
    sd = vq_ckpt['model'] if 'model' in vq_ckpt else vq_ckpt
    # Nettoyage prefix module si présent
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}
    
    try:
        target_vq.load_state_dict(sd, strict=False)
        print("VQ-VAE chargé avec succès.")
    except Exception as e:
        print(f"Erreur chargement VQ-VAE: {e}")

def process_video_and_bg(path, img_size):
    cap = cv2.VideoCapture(path)
    frames_raw = []
    frames_for_model = []
    masks = []
    while len(frames_raw) < 100:
        ret, frame = cap.read()
        if not ret: break
        frames_raw.append(frame)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (img_size, img_size))
        frames_for_model.append(rgb_resized.astype(np.float32) / 255.0)
    cap.release()
    
    # Background (médiane pour le calcul du masque de départ)
    indices = np.linspace(0, len(frames_raw)-1, min(len(frames_raw), 30), dtype=int)
    background = np.median(np.stack([frames_raw[i] for i in indices]), axis=0).astype(np.uint8)

    for i, frame in enumerate(frames_raw):
        diff = cv2.absdiff(frame, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        masks.append(mask)
    
    return frames_raw, frames_for_model, background, masks

# --- MAIN ---

def main():
    # Setup dossiers
    os.makedirs(OUTPUT_DIR_FRAMES, exist_ok=True)
    
    config = Configuration(CONFIG_PATH)
    img_size = 64  # Résolution cible 128x128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialisation
    model = Model(config["model"])
    load_checkpoints(model, FLOW_CKPT, VQVAE_CKPT)
    model.to(device).eval()

    # 2. Data
    raw_bgr, model_input, _, masks = process_video_and_bg(VIDEO_PATH, img_size)
    # Frame 0 statique pour le décor de prédiction
    base_static_bgr = cv2.resize(raw_bgr[0], (img_size, img_size))

    # Contexte de 15 frames
    first_k = 15 
    fg_inputs_model = []
    for i in range(min(first_k, len(model_input))):
        m = masks[i]
        mask_resized = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST) if m is not None else np.zeros((img_size, img_size), dtype=np.uint8)
        mask_norm = (mask_resized.astype(np.float32) / 255.0)[..., None]
        fg = model_input[i] * mask_norm
        fg_inputs_model.append(fg)

    init_np = np.stack(fg_inputs_model) 
    init_tensor = torch.from_numpy(init_np * 2.0 - 1.0).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    # 3. Génération
    print(f"Génération (Context: {first_k} frames)...")
    with torch.no_grad():
        generated = model.generate_frames(init_tensor, num_frames=45)
        # Tronquer si le modèle répète l'input
        if generated.shape[1] >= first_k:
            generated = generated[:, first_k:]
        full_seq = torch.cat([init_tensor, generated], dim=1)[0]

    # 4. Traitement des frames et Vidéo
    side_by_side_frames = []
    print(f"Sauvegarde des frames dans {OUTPUT_DIR_FRAMES}")

    for t in range(full_seq.shape[0]):
        # Post-process Prédiction (Sortie directe IA en RGB)
        img_ia_float = full_seq[t].permute(1, 2, 0).cpu().numpy()
        img_ia_float = np.clip((img_ia_float + 1.0) / 2.0, 0, 1)
        img_ia_u8 = (img_ia_float * 255).astype(np.uint8)
        
        # --- SAUVEGARDE IMAGE INDIVIDUELLE ---
        imageio.imwrite(os.path.join(OUTPUT_DIR_FRAMES, f"prediction_{t:04d}.png"), img_ia_u8)
        # Save original frame (RGB) for comparison
        try:
            imageio.imwrite(os.path.join(OUTPUT_DIR_FRAMES, f"original_{t:04d}.png"), left_rgb)
        except Exception:
            # fallback: save the base static if original not available
            imageio.imwrite(os.path.join(OUTPUT_DIR_FRAMES, f"original_{t:04d}.png"), cv2.cvtColor(base_static_bgr, cv2.COLOR_BGR2RGB))

        # Préparation BGR pour OpenCV
        smoke_bgr = cv2.cvtColor(img_ia_u8, cv2.COLOR_RGB2BGR)

        # GAUCHE : Vidéo normale
        if t < len(raw_bgr):
            left_bgr = cv2.resize(raw_bgr[t], (img_size, img_size))
        else:
            left_bgr = base_static_bgr
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

        # DROITE : Normal jusqu'à 15, puis Prediction sur Frame 0
        if t < first_k:
            right_rgb = left_rgb
        else:
            # Composition sur la frame 0 statique
            combined_bgr = cv2.add(base_static_bgr, smoke_bgr)
            right_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            cv2.putText(
                right_rgb,                # Image cible
                "PREDICTION",             # Le texte
                (5, 15),                  # Position (x, y) en haut à gauche
                cv2.FONT_HERSHEY_SIMPLEX, # Police de caractères
                0.4,                      # Échelle (taille de la police, petit pour du 64x64)
                (0, 255, 0),              # Couleur (Vert vif RGB)
                1,                        # Épaisseur du trait
                cv2.LINE_AA               # Anti-aliasing pour lisser le texte
            )
            # -------------
        # Side-by-Side
        sbs = np.concatenate([left_rgb, right_rgb], axis=1)
        side_by_side_frames.append(_pad_to_block(sbs, block=16))

    # 5. Save Video
    imageio.mimsave(OUTPUT_NAME, side_by_side_frames, fps=15)
    print(f"Extraction et vidéo terminées : {OUTPUT_NAME}")

if __name__ == "__main__":
    main()