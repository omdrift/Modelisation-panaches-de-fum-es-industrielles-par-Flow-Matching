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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory (overrides runs/* path)")
    parser.add_argument("--test-image", type=str, help="Path to a specific image to test every epoch")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    return parser.parse_args()


def train_vqvae():
    args = parse_args()
    
    # --- Setup ---
    config = Configuration(args.config)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("runs") / f"vqvae_{args.run_name}"
    recons_dir = output_dir / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    recons_dir.mkdir(parents=True, exist_ok=True)
    
    if args.wandb:
        wandb.init(project="river-vqvae", name=args.run_name, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Préparation de l'image de test fixe ---
    data_config = config["data"]
    custom_test_tensor = None
    if args.test_image and os.path.exists(args.test_image):
        print(f"Loading custom test image: {args.test_image}")
        test_transform = transforms.Compose([
            transforms.Resize(data_config["input_size"]),
            transforms.CenterCrop(data_config["crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Range [-1, 1]
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    # Validation and Test loaders
    val_dataset = TextBasedVideoDataset(
        data_path=data_config["data_root"],
        file_list="val.txt",
        input_size=data_config["input_size"],
        crop_size=data_config["crop_size"],
        frames_per_sample=1,
        random_horizontal_flip=False,
        random_time=False
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    test_dataset = TextBasedVideoDataset(
        data_path=data_config["data_root"],
        file_list="test.txt",
        input_size=data_config["input_size"],
        crop_size=data_config["crop_size"],
        frames_per_sample=1,
        random_horizontal_flip=False,
        random_time=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # --- Model ---
# --- Model (Updated to use YAML config) ---
# We map the YAML structure to the format build_vqvae expects
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
    optimizer = AdamW(vqvae.parameters(), lr=args.lr, weight_decay=1e-6)
    
    history = {"train_loss": [], "train_eval_loss": [], "val_loss": [], "test_loss": []}
    global_step = 0

    def evaluate_split(split_name, loader, num_examples=4):
        vqvae.eval()
        total = 0.0
        count = 0
        example_grid = None
        with torch.no_grad():
            for i, frames in enumerate(loader):
                if frames.dim() == 5: frames = frames.squeeze(1)
                frames = frames.to(device)
# --- Replacement Loss Logic ---
                output = vqvae(frames)
                recon = output.reconstructed_images

                # Create a weight map based on pixel intensity (0 to 1)
                # Smoke pixels are brighter than the background [-1, 1] range
                with torch.no_grad():
                    # Use max across RGB channels to find the smoke "signal"
                    weights = (frames.max(dim=1, keepdim=True)[0] + 1.0) / 2.0
                    # Boost weight for faint smoke so the model doesn't ignore it
                    weights = torch.pow(weights, 0.5) 

                # Apply weighted MSE
                sq_error = (recon - frames) ** 2
                recon_loss = (sq_error * weights).sum() / (weights.sum() + 1e-8)

                vq_loss = getattr(output, 'vq_loss', 0.0)
                if isinstance(vq_loss, torch.Tensor):
                    vq_loss = vq_loss.sum().item()

                total += float(recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss) + vq_loss
                count += frames.size(0)

                if i == 0:
                    # build example grid from first batch
                    orig_viz = frames.cpu()
                    recon_viz = torch.clamp(recon.cpu(), -1.0, 1.0)
                    # take up to num_examples
                    take = min(orig_viz.size(0), num_examples)
                    comparison = torch.cat([orig_viz[:take], recon_viz[:take]], dim=0)
                    grid = make_grid(comparison, nrow=take, normalize=True, value_range=(-1, 1))
                    example_grid = grid

        avg_loss = total / max(1, count)
        # save example grid if exists
        if example_grid is not None:
            img_name = f"recon_{split_name}_epoch_{epoch+1}.png"
            save_image(example_grid, recons_dir / img_name)
            if args.wandb:
                wandb.log({f"recon_{split_name}": [wandb.Image(example_grid, caption=f"{split_name} epoch {epoch+1}")]}, step=global_step)

        return avg_loss

    # --- Training Loop ---
    for epoch in range(args.epochs):
        vqvae.train()
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for frames in pbar:
            if frames.dim() == 5: frames = frames.squeeze(1)
            frames = frames.cuda()
            
            output = vqvae(frames)
            recon = output.reconstructed_images
            # Masked reconstruction loss during training: ignore near-black pixels
            mask = (frames > -0.99).any(dim=1)  # [B, H, W]
            if mask.sum() > 0:
                mask_expand = mask.unsqueeze(1).float()
                sq = (recon - frames) ** 2
                recon_loss = (sq * mask_expand).sum() / mask_expand.sum()
            else:
                recon_loss = F.mse_loss(recon, frames, reduction='mean')

            vq_loss = output.vq_loss
            total_loss = recon_loss + (vq_loss if not isinstance(vq_loss, torch.Tensor) else vq_loss.sum())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            global_step += 1
            
        history["train_loss"].append(epoch_train_loss / len(train_loader))

        # --- Test sur l'image personnalisée ---
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
                    wandb.log({"custom_test": [wandb.Image(grid, caption=f"Epoch {epoch+1}")]}, step=global_step)

        # --- Evaluate on train (subset), val and test ---
        if (epoch + 1) % args.eval_every == 0:
            train_eval_loss = evaluate_split('train', train_loader)
            val_loss = evaluate_split('val', val_loader)
            test_loss = evaluate_split('test', test_loader)
            history["train_eval_loss"].append(train_eval_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)
            print(f"Eval Epoch {epoch+1}: train={train_eval_loss:.6f}, val={val_loss:.6f}, test={test_loss:.6f}")

        # --- Checkpoint ---
        if (epoch + 1) % args.save_every == 0:
            torch.save({"model": vqvae.state_dict()}, output_dir / f"vqvae_epoch_{epoch+1}.ckpt")

    # --- Plot Final des Pertes ---
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="Train (train loop)")
    if len(history.get("train_eval_loss", [])) > 0:
        plt.plot(epochs, history["train_eval_loss"], label="Train (eval)")
    if len(history.get("val_loss", [])) > 0:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    if len(history.get("test_loss", [])) > 0:
        plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolution de la perte")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curves_all.png")
    
    torch.save({"model": vqvae.state_dict()}, output_dir / "vqvae_final.ckpt")
    print(f"Entraînement terminé. Résultats dans : {output_dir}")
    if args.wandb: wandb.finish()


if __name__ == "__main__":
    train_vqvae()