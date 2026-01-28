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
    parser.add_argument("--test-image", type=str, help="Path to a specific image to test every epoch")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    return parser.parse_args()


def train_vqvae():
    args = parse_args()
    
    # --- Setup ---
    config = Configuration(args.config)
    output_dir = Path("runs") / f"vqvae_{args.run_name}"
    recons_dir = output_dir / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    recons_dir.mkdir(parents=True, exist_ok=True)
    
    if args.wandb:
        wandb.init(project="river-vqvae", name=args.run_name, config=vars(args))
    
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
        custom_test_tensor = test_transform(img).unsqueeze(0).cuda()

    # --- Datasets ---
    train_dataset = TextBasedVideoDataset(
        data_path=data_config["data_root"],
        file_list="train_files.txt",
        input_size=data_config["input_size"],
        crop_size=data_config["crop_size"],
        frames_per_sample=1,
        random_horizontal_flip=data_config.get("random_horizontal_flip", True),
        random_time=True
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    
    # --- Model ---
    vqvae_config = DictWrapper({
        "encoder": {"in_channels": 3, "out_channels": 4, "mid_channels": 32},
        "decoder": {"in_channels": 4, "out_channels": 3, "mid_channels": 32},
        "vector_quantizer": {
            "embedding_dimension": 4,
            "num_embeddings": 16384,
            "commitment_cost": 0.25
        }
    })
    
    vqvae = build_vqvae(vqvae_config).cuda()
    optimizer = AdamW(vqvae.parameters(), lr=args.lr, weight_decay=1e-6)
    
    history = {"train_loss": [], "val_loss": [], "val_recon": []}
    global_step = 0

    # --- Training Loop ---
    for epoch in range(args.epochs):
        vqvae.train()
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for frames in pbar:
            if frames.dim() == 5: frames = frames.squeeze(1)
            frames = frames.cuda()
            
            output = vqvae(frames)
            recon_loss = F.mse_loss(output.reconstructed_images, frames)
            vq_loss = output.vq_loss
            total_loss = recon_loss + vq_loss
            
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
                
                # Debugging values if it's still white
                # print(f"Min: {recon_custom.min()}, Max: {recon_custom.max()}")

                # Sécurité pour l'affichage : clamp et concaténation
                # On remet en [0, 1] pour save_image
                orig_viz = custom_test_tensor.cpu()
                recon_viz = torch.clamp(recon_custom.cpu(), -1.0, 1.0)
                
                comparison = torch.cat([orig_viz, recon_viz], dim=0)
                grid = make_grid(comparison, nrow=1, normalize=True, value_range=(-1, 1))
                
                img_name = f"custom_recon_epoch_{epoch+1}.png"
                save_image(grid, recons_dir / img_name)
                
                if args.wandb:
                    wandb.log({"custom_test": [wandb.Image(grid, caption=f"Epoch {epoch+1}")]}, step=global_step)

        # --- Checkpoint ---
        if (epoch + 1) % args.save_every == 0:
            torch.save({"model": vqvae.state_dict()}, output_dir / f"vqvae_epoch_{epoch+1}.ckpt")

    # --- Plot Final des Pertes ---
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolution de la perte")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curves.png")
    
    torch.save({"model": vqvae.state_dict()}, output_dir / "vqvae_final.ckpt")
    print(f"Entraînement terminé. Résultats dans : {output_dir}")
    if args.wandb: wandb.finish()


if __name__ == "__main__":
    train_vqvae()