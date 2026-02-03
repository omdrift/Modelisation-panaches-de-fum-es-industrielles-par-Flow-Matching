"""
VQGAN Training Script
Trains a Vector-Quantized GAN with adversarial and perceptual losses.
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from PIL import Image

from dataset.text_based_video_dataset import TextBasedVideoDataset
from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper
from model.vqgan.vqvae import build_vqvae
from model.vqgan.discriminator import build_discriminator, hinge_d_loss, adopt_weight
from model.vqgan.losses import build_perceptual_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory")
    parser.add_argument("--test-image", type=str, help="Path to a specific image to test every epoch")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-disc", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    
    # Loss weights
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="Weight for perceptual loss")
    parser.add_argument("--adversarial-weight", type=float, default=0.5, help="Weight for adversarial loss")
    parser.add_argument("--disc-start", type=int, default=10000, help="Start discriminator after N steps")
    parser.add_argument("--disc-weight-max", type=float, default=0.75, help="Max discriminator loss weight")
    
    # Smoke filtering
    parser.add_argument("--min-smoke-pixels", type=float, default=0.0003, help="Minimum smoke pixel ratio (0.01%)")
    parser.add_argument("--max-smoke-pixels", type=float, default=0.85, help="Maximum smoke pixel ratio (85%)")
    
    return parser.parse_args()


def train_vqgan():
    args = parse_args()
    
    # --- Setup ---
    config = Configuration(args.config)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("runs_vqgan") / f"vqgan_{args.run_name}"
    recons_dir = output_dir / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    recons_dir.mkdir(parents=True, exist_ok=True)
    
    if args.wandb:
        wandb.init(project="smoke-vqgan", name=args.run_name, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    def filter_smoke_content(frames):
        """
        Filter frames based on smoke pixel content.
        Returns mask of valid frames (True = keep, False = discard).
        Smoke is detected as non-background pixels (> -0.99).
        """
        # frames: [B, C, H, W] in range [-1, 1]
        # Background is typically -1 (black after normalization)
        smoke_mask = (frames > -0.99).any(dim=1)  # [B, H, W]
        total_pixels = smoke_mask[0].numel()  # H * W
        smoke_pixel_counts = smoke_mask.sum(dim=(1, 2)).float()  # [B]
        smoke_ratios = smoke_pixel_counts / total_pixels
        
        valid = (smoke_ratios >= args.min_smoke_pixels) & (smoke_ratios <= args.max_smoke_pixels)
        return valid, smoke_ratios
    
    # --- Prepare fixed test image ---
    data_config = config["data"]
    custom_test_tensor = None
    if args.test_image and os.path.exists(args.test_image):
        print(f"Loading custom test image: {args.test_image}")
        test_transform = transforms.Compose([
            transforms.Resize(data_config["input_size"]),
            transforms.CenterCrop(data_config["crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        img = Image.open(args.test_image).convert("RGB")
        custom_test_tensor = test_transform(img).unsqueeze(0).to(device)

    # --- Datasets ---
    train_dataset = TextBasedVideoDataset(
        data_path=data_config["data_root"],
        file_list="train.txt",
        input_size=data_config["input_size"],
        crop_size=data_config["crop_size"],
        frames_per_sample=1,
        random_horizontal_flip=data_config.get("random_horizontal_flip", True),
        random_time=True
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    val_dataset = TextBasedVideoDataset(
        data_path=data_config["data_root"],
        file_list="val.txt",
        input_size=data_config["input_size"],
        crop_size=data_config["crop_size"],
        frames_per_sample=1,
        random_horizontal_flip=False,
        random_time=True  # Randomize time for variety
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,  # Always shuffle to show different samples
        num_workers=args.num_workers, 
        pin_memory=True
    )

    test_dataset = TextBasedVideoDataset(
        data_path=data_config["data_root"],
        file_list="test.txt",
        input_size=data_config["input_size"],
        crop_size=data_config["crop_size"],
        frames_per_sample=1,
        random_horizontal_flip=False,
        random_time=True  # Randomize time for variety
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,  # Always shuffle to show different samples
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # --- Models ---
    # Generator (VQVAE)
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
    
    # Discriminator
    discriminator = build_discriminator(input_nc=3, ndf=64, n_layers=3).to(device)
    
    # Perceptual Loss
    perceptual_loss_fn = build_perceptual_loss(simplified=True).to(device)
    perceptual_loss_fn.eval()
    
    print(f"Generator parameters: {sum(p.numel() for p in vqvae.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # --- Optimizers ---
    optimizer_g = AdamW(vqvae.parameters(), lr=args.lr, weight_decay=1e-6)
    optimizer_d = AdamW(discriminator.parameters(), lr=args.lr_disc, weight_decay=1e-6)
    
    # --- Training Loop ---
    history = {
        "train_loss": [], "train_recon_loss": [], "train_perceptual_loss": [],
        "train_vq_loss": [], "train_g_loss": [], "train_d_loss": [],
        "val_loss": [], "test_loss": []
    }
    global_step = 0
    
    # Statistics for filtered images
    total_filtered = 0
    total_processed = 0
    
    def evaluate_split(split_name, loader, num_examples=4):
        """Evaluate on validation/test set"""
        vqvae.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_perceptual = 0.0
        total_vq = 0.0
        count = 0
        example_grid = None
        split_filtered = 0
        split_processed = 0
        
        with torch.no_grad():
            for i, frames in enumerate(loader):
                if frames.dim() == 5:
                    frames = frames.squeeze(1)
                frames = frames.to(device)
                
                # Filter frames based on smoke content
                valid_mask, smoke_ratios = filter_smoke_content(frames)
                split_processed += frames.size(0)
                split_filtered += (~valid_mask).sum().item()
                
                if not valid_mask.any():
                    continue
                
                # Keep only valid frames
                frames = frames[valid_mask]
                
                output = vqvae(frames)
                recon = output.reconstructed_images
                
                # Reconstruction loss (weighted by intensity)
                weights = (frames.max(dim=1, keepdim=True)[0] + 1.0) / 2.0
                weights = torch.pow(weights, 0.5)
                sq_error = (recon - frames) ** 2
                recon_loss = (sq_error * weights).sum() / (weights.sum() + 1e-8)
                
                # Perceptual loss
                perceptual_loss = perceptual_loss_fn(recon, frames)
                
                # VQ loss
                vq_loss = output.vq_loss
                if isinstance(vq_loss, torch.Tensor):
                    vq_loss = vq_loss.sum()
                
                total_loss += (recon_loss.item() + args.perceptual_weight * perceptual_loss.item() + vq_loss)
                total_recon += recon_loss.item()
                total_perceptual += perceptual_loss.item()
                total_vq += vq_loss if isinstance(vq_loss, float) else vq_loss.item()
                count += frames.size(0)
                
                if i == 0:
                    # Create visualization grid
                    orig_viz = frames.cpu()
                    recon_viz = torch.clamp(recon.cpu(), -1.0, 1.0)
                    take = min(orig_viz.size(0), num_examples)
                    comparison = torch.cat([orig_viz[:take], recon_viz[:take]], dim=0)
                    grid = make_grid(comparison, nrow=take, normalize=True, value_range=(-1, 1))
                    example_grid = grid
        
        # Save visualization
        print(f"  {split_name.capitalize()} filtered: {split_filtered}/{split_processed} images ({100*split_filtered/max(1,split_processed):.2f}%)")
        
        if example_grid is not None:
            img_name = f"recon_{split_name}_epoch_{epoch+1}.png"
            save_image(example_grid, recons_dir / img_name)
            if args.wandb:
                wandb.log({
                    f"recon_{split_name}": [wandb.Image(example_grid, caption=f"{split_name} epoch {epoch+1}")]
                }, step=global_step)
        
        avg_loss = total_loss / max(1, count)
        avg_recon = total_recon / max(1, count)
        avg_perceptual = total_perceptual / max(1, count)
        avg_vq = total_vq / max(1, count)
        
        return avg_loss, avg_recon, avg_perceptual, avg_vq
    
    # --- Main Training Loop ---
    for epoch in range(args.epochs):
        vqvae.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_recon_loss = 0
        epoch_perceptual_loss = 0
        epoch_vq_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for frames in pbar:
            if frames.dim() == 5:
                frames = frames.squeeze(1)
            frames = frames.to(device)
            
            # Filter frames based on smoke content
            valid_mask, smoke_ratios = filter_smoke_content(frames)
            total_processed += frames.size(0)
            total_filtered += (~valid_mask).sum().item()
            
            if not valid_mask.any():
                # Skip batch if all frames are invalid
                continue
            
            # Keep only valid frames
            frames = frames[valid_mask]
            
            # ==================== Train Generator ====================
            optimizer_g.zero_grad()
            
            output = vqvae(frames)
            recon = output.reconstructed_images
            
            # 1. Reconstruction Loss (masked)
            mask = (frames > -0.99).any(dim=1)  # [B, H, W]
            if mask.sum() > 0:
                mask_expand = mask.unsqueeze(1).float()
                sq = (recon - frames) ** 2
                recon_loss = (sq * mask_expand).sum() / mask_expand.sum()
            else:
                recon_loss = F.mse_loss(recon, frames, reduction='mean')
            
            # 2. Perceptual Loss
            perceptual_loss = perceptual_loss_fn(recon, frames)
            
            # 3. VQ Loss
            vq_loss = output.vq_loss
            if isinstance(vq_loss, torch.Tensor):
                vq_loss = vq_loss.sum()
            
            # 4. Adversarial Loss (Generator)
            # Discriminator weight scheduling
            disc_factor = adopt_weight(
                args.adversarial_weight, 
                global_step, 
                threshold=args.disc_start
            )
            
            if disc_factor > 0:
                logits_fake = discriminator(recon)
                g_loss = -torch.mean(logits_fake)  # Generator wants high scores
            else:
                g_loss = torch.tensor(0.0, device=device)
            
            # Total Generator Loss
            total_g_loss = (
                recon_loss + 
                args.perceptual_weight * perceptual_loss + 
                vq_loss + 
                disc_factor * g_loss
            )
            
            total_g_loss.backward()
            optimizer_g.step()
            
            # ==================== Train Discriminator ====================
            if disc_factor > 0:
                optimizer_d.zero_grad()
                
                # Real images
                logits_real = discriminator(frames.detach())
                
                # Fake images
                logits_fake = discriminator(recon.detach())
                
                # Hinge loss
                d_loss = hinge_d_loss(logits_real, logits_fake)
                
                # Adaptive discriminator weight
                # Prevent discriminator from dominating
                d_loss_weighted = args.disc_weight_max * d_loss
                d_loss_weighted.backward()
                optimizer_d.step()
            else:
                d_loss = torch.tensor(0.0)
            
            # Update metrics
            epoch_g_loss += total_g_loss.item()
            epoch_d_loss += d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss
            epoch_recon_loss += recon_loss.item()
            epoch_perceptual_loss += perceptual_loss.item()
            epoch_vq_loss += vq_loss if isinstance(vq_loss, float) else vq_loss.item()
            
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'g_loss': f'{total_g_loss.item():.4f}',
                'd_loss': f'{d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss:.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'disc_factor': f'{disc_factor:.3f}'
            })
            
            # Log to wandb
            if args.wandb and global_step % 100 == 0:
                wandb.log({
                    "train/g_loss": total_g_loss.item(),
                    "train/d_loss": d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
                    "train/recon_loss": recon_loss.item(),
                    "train/perceptual_loss": perceptual_loss.item(),
                    "train/vq_loss": vq_loss if isinstance(vq_loss, float) else vq_loss.item(),
                    "train/disc_factor": disc_factor,
                }, step=global_step)
        
        # Save epoch metrics
        n_batches = len(train_loader)
        history["train_loss"].append(epoch_g_loss / n_batches)
        print(f"  Filtered: {total_filtered}/{total_processed} images ({100*total_filtered/max(1,total_processed):.2f}%)")
        
        # Reset filter stats for next epoch
        total_filtered = 0
        total_processed = 0
        
        # --- Visualize random train sample ---
        vqvae.eval()
        with torch.no_grad():
            # Get one random batch from train set
            train_iter = iter(train_loader)
            train_sample = next(train_iter)
            if train_sample.dim() == 5:
                train_sample = train_sample.squeeze(1)
            train_sample = train_sample.to(device)
            
            # Filter and visualize first valid sample
            valid_mask, _ = filter_smoke_content(train_sample)
            if valid_mask.any():
                train_sample = train_sample[valid_mask][:1]  # Take first valid
                output_train = vqvae(train_sample)
                recon_train = output_train.reconstructed_images
                orig_viz = train_sample.cpu()
                recon_viz = torch.clamp(recon_train.cpu(), -1.0, 1.0)
                comparison = torch.cat([orig_viz, recon_viz], dim=0)
                grid = make_grid(comparison, nrow=1, normalize=True, value_range=(-1, 1))
                img_name = f"train_sample_epoch_{epoch+1}.png"
                save_image(grid, recons_dir / img_name)
                if args.wandb:
                    wandb.log({
                        "train_sample": [wandb.Image(grid, caption=f"Random train epoch {epoch+1}")]
                    }, step=global_step)
        history["train_recon_loss"].append(epoch_recon_loss / n_batches)
        history["train_perceptual_loss"].append(epoch_perceptual_loss / n_batches)
        history["train_vq_loss"].append(epoch_vq_loss / n_batches)
        history["train_g_loss"].append(epoch_g_loss / n_batches)
        history["train_d_loss"].append(epoch_d_loss / n_batches)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  G Loss: {epoch_g_loss/n_batches:.6f}")
        print(f"  D Loss: {epoch_d_loss/n_batches:.6f}")
        print(f"  Recon: {epoch_recon_loss/n_batches:.6f}")
        print(f"  Perceptual: {epoch_perceptual_loss/n_batches:.6f}")
        print(f"  VQ: {epoch_vq_loss/n_batches:.6f}")
        
        # --- Test on custom image ---
        vqvae.eval()
        with torch.no_grad():
            if custom_test_tensor is not None:
                output_custom = vqvae(custom_test_tensor)
                recon_custom = output_custom.reconstructed_images
                orig_viz = custom_test_tensor.cpu()
                recon_viz = torch.clamp(recon_custom.cpu(), -1.0, 1.0)
                comparison = torch.cat([orig_viz, recon_viz], dim=0)
                grid = make_grid(comparison, nrow=1, normalize=True, value_range=(-1, 1))
                img_name = f"custom_recon_epoch_{epoch+1}.png"
                save_image(grid, recons_dir / img_name)
                if args.wandb:
                    wandb.log({
                        "custom_test": [wandb.Image(grid, caption=f"Epoch {epoch+1}")]
                    }, step=global_step)
        
        # --- Evaluate ---
        if (epoch + 1) % args.eval_every == 0:
            val_loss, val_recon, val_perceptual, val_vq = evaluate_split('val', val_loader)
            test_loss, test_recon, test_perceptual, test_vq = evaluate_split('test', test_loader)
            
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)
            
            print(f"  Val Loss: {val_loss:.6f} (recon={val_recon:.6f}, perceptual={val_perceptual:.6f}, vq={val_vq:.6f})")
            print(f"  Test Loss: {test_loss:.6f} (recon={test_recon:.6f}, perceptual={test_perceptual:.6f}, vq={test_vq:.6f})")
            
            if args.wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/recon_loss": val_recon,
                    "val/perceptual_loss": val_perceptual,
                    "val/vq_loss": val_vq,
                    "test/loss": test_loss,
                    "test/recon_loss": test_recon,
                    "test/perceptual_loss": test_perceptual,
                    "test/vq_loss": test_vq,
                }, step=global_step)
        
        # --- Checkpoint ---
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "vqvae": vqvae.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
            }
            torch.save(checkpoint, output_dir / f"vqgan_epoch_{epoch+1}.ckpt")
            print(f"  Checkpoint saved: vqgan_epoch_{epoch+1}.ckpt")
    
    # --- Final Plots ---
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history["train_recon_loss"], label="Recon Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history["train_perceptual_loss"], label="Perceptual Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Perceptual Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(history["train_vq_loss"], label="VQ Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VQ Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(history["train_g_loss"], label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(history["train_d_loss"], label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    if len(history["val_loss"]) > 0:
        plt.plot(history["val_loss"], label="Val Loss")
    if len(history["test_loss"]) > 0:
        plt.plot(history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation & Test Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves_all.png")
    
    # --- Save Final Models ---
    final_checkpoint = {
        "epoch": args.epochs,
        "global_step": global_step,
        "vqvae": vqvae.state_dict(),
        "discriminator": discriminator.state_dict(),
    }
    torch.save(final_checkpoint, output_dir / "vqgan_final.ckpt")
    
    # Also save generator only for easy loading
    torch.save({"model": vqvae.state_dict()}, output_dir / "vqgan_generator_final.ckpt")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    train_vqgan()
