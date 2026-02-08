import os
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from pymatting import estimate_alpha_cf

from model.vqgan.vqvae import build_vqvae
from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper


def extract_foreground(video_path, num_frames=5):
    """Extract smoke foreground using pymatting to avoid fog."""
    cap = cv2.VideoCapture(str(video_path))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    
    if len(frames) < num_frames:
        return None, None, None
    
    # Select evenly spaced frames
    indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    selected_frames = [frames[i] for i in indices]
    
    # Estimate background using median
    video_stack = np.stack(frames)
    background = np.median(video_stack, axis=0)
    
    # Calculate temporal statistics for smoke detection
    std_dev = np.std(video_stack, axis=0)
    turbulence = np.max(std_dev, axis=2)
    turbulence = cv2.normalize(turbulence, None, 0, 1, cv2.NORM_MINMAX)
    
    # Extract foreground for selected frames using pymatting
    foregrounds = []
    alphas = []
    originals = []
    
    for frame in selected_frames:
        h, w = frame.shape[:2]
        
        # Calculate smoke score with emphasis on temporal turbulence (not fog)
        diff = np.max(np.abs(frame - background), axis=2)
        smoke_score = (diff * 0.6) + (turbulence * 0.4)  # More weight on turbulence
        max_s = np.max(smoke_score)
        
        # Skip if too weak - seuil baissé pour capturer plus
        if max_s < 0.03:  # Baissé de 0.06 à 0.03
            foregrounds.append(np.zeros_like(frame))
            alphas.append(np.zeros((h, w)))
            originals.append(frame)
            continue
        
        # Create trimap - moins strict pour capturer plus de fumée
        trimap = np.full((h, w), 0.5, dtype=np.float32)
        trimap[smoke_score < (max_s * 0.20)] = 0.0  # Background - baissé de 0.25
        trimap[smoke_score > (max_s * 0.65)] = 1.0  # Foreground - baissé de 0.80
        
        # Check if we have enough smoke - seuil réduit
        fg_pixels = np.sum(trimap == 1.0)
        if fg_pixels < 50:  # Réduit de 100 à 50
            foregrounds.append(np.zeros_like(frame))
            alphas.append(np.zeros((h, w)))
            originals.append(frame)
            continue
        
        try:
            # Resize for faster matting
            small_frame = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)
            small_trimap = cv2.resize(trimap, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            
            alpha_small = estimate_alpha_cf(small_frame.astype(np.float64), 
                                           small_trimap.astype(np.float64))
            
            alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = np.clip(alpha, 0, 1)
            alpha = np.where(alpha < 0.08, 0, alpha)  # Seuil baissé de 0.15 à 0.08
            
            # Check average alpha - critère assoupli
            avg_alpha = np.mean(alpha[alpha > 0.08])
            if avg_alpha < 0.25:  # Baissé de 0.4 à 0.25 pour capturer fumée plus légère
                foregrounds.append(np.zeros_like(frame))
                alphas.append(np.zeros((h, w)))
                originals.append(frame)
                continue
            
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)
            alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            
            # Extract foreground
            fg = frame * alpha_3d
            
            # Check if extracted smoke has bright regions - seuil réduit
            max_intensity = np.max(fg)
            if max_intensity < 0.15:  # Baissé de 0.3 à 0.15 pour fumée plus faible
                foregrounds.append(np.zeros_like(frame))
                alphas.append(np.zeros((h, w)))
                originals.append(frame)
                continue
            
            foregrounds.append(fg)
            alphas.append(alpha)
            originals.append(frame)
            
        except Exception as e:
            print(f"Error in matting: {e}")
            foregrounds.append(np.zeros_like(frame))
            alphas.append(np.zeros((h, w)))
            originals.append(frame)
    
    return foregrounds, alphas, originals, background


def load_vqgan(checkpoint_path, config_path):
    """Load trained VQGAN model."""
    config = Configuration(config_path)
    
    vqgan_params = DictWrapper({
        "encoder": {
            "in_channels": 3,
            "out_channels": config["encoder"]["out_channels"], 
            "mid_channels": config["encoder"]["mid_channels"]
        },
        "decoder": {
            "in_channels": config["encoder"]["out_channels"],
            "out_channels": 3,
            "mid_channels": config["decoder"]["mid_channels"]
        },
        "vector_quantizer": config["vector_quantizer"]
    })
    
    vqgan = build_vqvae(vqgan_params)
    
    # Load checkpoint - VQGAN uses 'vqvae' key
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'vqvae' in checkpoint:
        vqgan.load_state_dict(checkpoint['vqvae'])
    elif 'model' in checkpoint:
        vqgan.load_state_dict(checkpoint['model'])
    else:
        vqgan.load_state_dict(checkpoint)
    
    vqgan.eval()
    
    return vqgan


def process_frame(fg, alpha, background, vqgan, device, crop_size=64):
    """Process frame with background-dominant composition."""
    # Prepare transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Convert foreground to tensor
    fg_uint8 = (np.clip(fg, 0, 1) * 255).astype(np.uint8)
    fg_tensor = transform(fg_uint8).unsqueeze(0).to(device)
    
    # Reconstruct foreground with VQGAN
    with torch.no_grad():
        output = vqgan(fg_tensor)
        recon_tensor = output.reconstructed_images
    
    # Convert back to numpy
    recon = recon_tensor.squeeze(0).cpu()
    recon = (recon * 0.5 + 0.5).clamp(0, 1)  # Denormalize
    recon = recon.permute(1, 2, 0).numpy()
    
    # Resize back to original size
    h, w = background.shape[:2]
    recon_resized = cv2.resize(recon, (w, h))
    fg_resized = cv2.resize(fg, (w, h))
    alpha_resized = cv2.resize(alpha, (w, h))
    alpha_3d = alpha_resized[:, :, np.newaxis]
    
    # Colonne 4: FG reconstruit + BG rigoureux
    # On applique le background de façon stricte pour overwrite 
    # les parties que le modèle a prédites alors que c'est censé être noir (mask)
    
    # Où est-ce que le FG extrait est noir (mask) ?
    fg_is_mask = (np.max(fg_resized, axis=2) < 0.05).astype(np.float32)
    fg_is_mask_3d = fg_is_mask[:, :, np.newaxis]
    
    # Où le FG extrait est noir → force background
    # Où le FG extrait a de la fumée → utilise reconstruction
    composite_corrected = background * fg_is_mask_3d + recon_resized * (1 - fg_is_mask_3d)
    
    # Colonne 5: Composition finale (optionnel, même chose ou autre logique)
    # Pour l'instant identique à colonne 4
    composite_final = composite_corrected.copy()
    
    return fg_resized, recon_resized, composite_corrected, composite_final


def visualize_comparison(video_path, checkpoint_path, config_path, output_path, num_frames=5):
    """Create comparison visualization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading VQGAN from {checkpoint_path}")
    vqgan = load_vqgan(checkpoint_path, config_path).to(device)
    
    # Extract foregrounds
    print(f"Processing video: {video_path}")
    foregrounds, alphas, originals, background = extract_foreground(video_path, num_frames)
    
    if foregrounds is None:
        print("Error: Not enough frames in video")
        return
    
    # Process each frame
    results = []
    for i, (fg, alpha, original) in enumerate(zip(foregrounds, alphas, originals)):
        print(f"Processing frame {i+1}/{num_frames}")
        fg_viz, recon_fg, comp_orig, comp_recon = process_frame(fg, alpha, background, vqgan, device)
        results.append({
            'original': original,
            'fg_extracted': fg_viz,
            'fg_reconstructed': recon_fg,
            'composite_original': comp_orig,
            'composite_reconstructed': comp_recon
        })
    
    # Create visualization
    fig, axes = plt.subplots(num_frames, 5, figsize=(20, 4*num_frames))
    
    if num_frames == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        # Original frame
        axes[i][0].imshow(cv2.cvtColor((result['original'] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[i][0].set_title('Original Frame', fontsize=10)
        axes[i][0].axis('off')
        
        # Extracted foreground
        axes[i][1].imshow(cv2.cvtColor((result['fg_extracted'] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[i][1].set_title('Extracted FG', fontsize=10)
        axes[i][1].axis('off')
        
        # Reconstructed foreground
        axes[i][2].imshow(cv2.cvtColor((result['fg_reconstructed'] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[i][2].set_title('VQGAN Reconstructed FG', fontsize=10)
        axes[i][2].axis('off')
        
        # VQGAN + BG Correction
        axes[i][3].imshow(cv2.cvtColor((result['composite_original'] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[i][3].set_title('VQGAN + BG Overwrite', fontsize=10)
        axes[i][3].axis('off')
        
        # Final composite
        axes[i][4].imshow(cv2.cvtColor((result['composite_reconstructed'] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[i][4].set_title('Final Result', fontsize=10)
        axes[i][4].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize VQGAN reconstruction on video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VQGAN checkpoint")
    parser.add_argument("--config", type=str, default="configs/config_vqgan.yaml", 
                        help="Path to config file")
    parser.add_argument("--output", type=str, default="vqgan_reconstruction_comparison.png",
                        help="Output PNG path")
    parser.add_argument("--num-frames", type=int, default=5, 
                        help="Number of frames to visualize")
    
    args = parser.parse_args()
    
    visualize_comparison(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        num_frames=args.num_frames
    )


if __name__ == "__main__":
    main()
