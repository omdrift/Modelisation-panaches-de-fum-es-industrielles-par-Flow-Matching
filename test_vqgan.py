"""
Test script for trained VQGAN model.
Evaluates reconstruction quality and generates visualizations.
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset.text_based_video_dataset import TextBasedVideoDataset
from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper
from model.vqgan.vqvae import build_vqvae


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained VQGAN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], 
                       help="Dataset split to test on")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-image", type=str, help="Path to specific image to test")
    parser.add_argument("--visualize-latents", action="store_true", help="Visualize latent codes")
    parser.add_argument("--save-all", action="store_true", help="Save all reconstructions")
    return parser.parse_args()


def compute_metrics(original, reconstructed):
    """
    Compute reconstruction metrics.
    
    Args:
        original: [B, C, H, W] tensor in range [-1, 1]
        reconstructed: [B, C, H, W] tensor in range [-1, 1]
    
    Returns:
        dict of metrics
    """
    # Convert to [0, 1] range
    original = (original + 1) / 2
    reconstructed = torch.clamp((reconstructed + 1) / 2, 0, 1)
    
    # MSE
    mse = F.mse_loss(reconstructed, original).item()
    
    # PSNR
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    
    # MAE
    mae = F.l1_loss(reconstructed, original).item()
    
    # SSIM (simplified version)
    # For proper SSIM, install pytorch-msssim: pip install pytorch-msssim
    try:
        from pytorch_msssim import ssim
        ssim_val = ssim(reconstructed, original, data_range=1.0).item()
    except ImportError:
        ssim_val = None
    
    return {
        "mse": mse,
        "psnr": psnr,
        "mae": mae,
        "ssim": ssim_val
    }


def visualize_codebook_usage(latent_ids, num_embeddings, save_path):
    """
    Visualize which codebook entries are being used.
    
    Args:
        latent_ids: [B, H, W, num_embeddings] one-hot vectors
        num_embeddings: int, size of codebook
        save_path: where to save visualization
    """
    # Get used codes
    codes = latent_ids.argmax(dim=-1).cpu().numpy().flatten()
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(codes, bins=min(num_embeddings, 100), edgecolor='black', alpha=0.7)
    plt.xlabel('Codebook Index')
    plt.ylabel('Frequency')
    plt.title(f'Codebook Usage (Unique codes: {len(np.unique(codes))}/{num_embeddings})')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    
    return len(np.unique(codes))


def test_single_image(model, image_path, output_dir, device, visualize_latents=False):
    """Test on a single image"""
    print(f"\nTesting on single image: {image_path}")
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        recon = output.reconstructed_images
    
    # Compute metrics
    metrics = compute_metrics(img_tensor, recon)
    print(f"Metrics: MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.2f}dB, MAE={metrics['mae']:.6f}")
    if metrics['ssim'] is not None:
        print(f"         SSIM={metrics['ssim']:.4f}")
    
    # Save comparison
    comparison = torch.cat([img_tensor.cpu(), recon.cpu()], dim=0)
    grid = make_grid(comparison, nrow=1, normalize=True, value_range=(-1, 1))
    save_image(grid, output_dir / "single_image_reconstruction.png")
    
    # Visualize latents if requested
    if visualize_latents:
        latents = output.latents.cpu()
        quantized_latents = output.quantized_latents.cpu()
        
        # Plot latent statistics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original latents
        axes[0, 0].imshow(latents[0].mean(dim=0), cmap='viridis')
        axes[0, 0].set_title('Original Latents (mean across channels)')
        axes[0, 0].axis('off')
        
        # Quantized latents
        axes[0, 1].imshow(quantized_latents[0].mean(dim=0), cmap='viridis')
        axes[0, 1].set_title('Quantized Latents (mean across channels)')
        axes[0, 1].axis('off')
        
        # Latent histogram
        axes[1, 0].hist(latents[0].numpy().flatten(), bins=50, alpha=0.7, label='Original')
        axes[1, 0].hist(quantized_latents[0].numpy().flatten(), bins=50, alpha=0.7, label='Quantized')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Latent Value Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Codebook usage
        latent_ids = output.quantized_latents_ids
        codes = latent_ids.argmax(dim=-1).cpu().numpy().flatten()
        axes[1, 1].hist(codes, bins=min(50, codes.max()+1), edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Codebook Index')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Codebook Usage ({len(np.unique(codes))} unique codes)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "single_image_latents.png", dpi=150)
        plt.close()
    
    print(f"Results saved to {output_dir}")


def test_dataset(model, dataloader, output_dir, device, num_samples, save_all=False, 
                visualize_latents=False, num_embeddings=1024):
    """Test on dataset"""
    print(f"\nTesting on dataset ({num_samples} samples)...")
    
    model.eval()
    
    all_metrics = []
    all_latent_ids = []
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, frames in enumerate(tqdm(dataloader, desc="Testing")):
            if samples_processed >= num_samples:
                break
            
            if frames.dim() == 5:
                frames = frames.squeeze(1)
            frames = frames.to(device)
            
            # Forward pass
            output = model(frames)
            recon = output.reconstructed_images
            
            # Compute metrics
            batch_metrics = compute_metrics(frames, recon)
            all_metrics.append(batch_metrics)
            
            # Collect latent IDs for codebook analysis
            all_latent_ids.append(output.quantized_latents_ids.cpu())
            
            # Save reconstructions for first few batches or if save_all is True
            if batch_idx < 5 or save_all:
                # Create comparison grid
                orig_viz = frames.cpu()
                recon_viz = torch.clamp(recon.cpu(), -1.0, 1.0)
                
                n_show = min(8, frames.size(0))
                comparison = torch.cat([orig_viz[:n_show], recon_viz[:n_show]], dim=0)
                grid = make_grid(comparison, nrow=n_show, normalize=True, value_range=(-1, 1))
                
                save_image(grid, output_dir / f"reconstruction_batch_{batch_idx:03d}.png")
            
            samples_processed += frames.size(0)
    
    # Aggregate metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics if m[key] is not None])
        for key in all_metrics[0].keys()
    }
    
    print("\n" + "="*60)
    print("RECONSTRUCTION METRICS")
    print("="*60)
    print(f"MSE:  {avg_metrics['mse']:.6f}")
    print(f"PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"MAE:  {avg_metrics['mae']:.6f}")
    if avg_metrics['ssim'] is not None and not np.isnan(avg_metrics['ssim']):
        print(f"SSIM: {avg_metrics['ssim']:.4f}")
    print("="*60)
    
    # Analyze codebook usage
    if visualize_latents:
        print("\nAnalyzing codebook usage...")
        all_latent_ids = torch.cat(all_latent_ids, dim=0)
        unique_codes = visualize_codebook_usage(
            all_latent_ids, 
            num_embeddings, 
            output_dir / "codebook_usage.png"
        )
        print(f"Codebook usage: {unique_codes}/{num_embeddings} codes ({100*unique_codes/num_embeddings:.1f}%)")
    
    # Save metrics to file
    with open(output_dir / "metrics.txt", "w") as f:
        f.write("RECONSTRUCTION METRICS\n")
        f.write("="*60 + "\n")
        for key, value in avg_metrics.items():
            if value is not None and not np.isnan(value):
                f.write(f"{key.upper()}: {value:.6f}\n")
        f.write("="*60 + "\n")
        if visualize_latents:
            f.write(f"\nCodebook usage: {unique_codes}/{num_embeddings} codes ({100*unique_codes/num_embeddings:.1f}%)\n")
    
    # Plot metrics distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist([m['mse'] for m in all_metrics], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('MSE Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist([m['psnr'] for m in all_metrics], bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('PSNR (dB)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('PSNR Distribution')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist([m['mae'] for m in all_metrics], bins=30, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('MAE')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('MAE Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_distribution.png", dpi=150)
    plt.close()
    
    print(f"\nResults saved to {output_dir}")


def main():
    args = parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config = Configuration(args.config)
    
    # Build model
    vqvae_params = DictWrapper({
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
    vqvae = build_vqvae(vqvae_params).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "vqvae" in checkpoint:
        state_dict = checkpoint["vqvae"]
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (DDP training)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    vqvae.load_state_dict(state_dict)
    vqvae.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in vqvae.parameters()):,}")
    
    # Test on single image if provided
    if args.test_image:
        test_single_image(
            vqvae, 
            args.test_image, 
            output_dir, 
            device,
            visualize_latents=args.visualize_latents
        )
    else:
        # Test on dataset
        data_config = config["data"]
        dataset = TextBasedVideoDataset(
            data_path=data_config["data_root"],
            file_list=f"{args.split}.txt",
            input_size=data_config["input_size"],
            crop_size=data_config["crop_size"],
            frames_per_sample=1,
            random_horizontal_flip=False,
            random_time=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        print(f"Testing on {args.split} split: {len(dataset)} samples")
        
        test_dataset(
            vqvae,
            dataloader,
            output_dir,
            device,
            num_samples=args.num_samples,
            save_all=args.save_all,
            visualize_latents=args.visualize_latents,
            num_embeddings=config["vector_quantizer"]["num_embeddings"]
        )


if __name__ == "__main__":
    main()
