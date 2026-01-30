import os
import torch
from torch.utils.data import Dataset
from lutils.configuration import Configuration
from model.vqgan.vqvae import build_vqvae


class LatentWrapperDataset(Dataset):
    """Wrap a frame-based dataset and return VQ-VAE latents instead of raw images.

    Args:
        base_dataset: a Dataset that returns a torch.Tensor video of shape [n, c, h, w] in [-1,1]
        vq_config: dict/Configuration for the VQ-VAE backbone
        vq_ckpt: path to the VQ-VAE checkpoint
        device: device to run encoding on (default: 'cpu')
        cache_dir: optional folder to store per-sample latents as .pt files
    """

    def __init__(self, base_dataset: Dataset, vq_config: Configuration, vq_ckpt: str, device: str = "cpu",
                 cache_dir: str = None):
        self.base = base_dataset
        self.device = torch.device(device)
        self.cache_dir = cache_dir

        # Build VQ-VAE backbone for encoding
        self.vq = build_vqvae(vq_config, convert_to_sequence=False).to(self.device)
        # load checkpoint (VQVAE implements load_from_ckpt)
        if vq_ckpt is not None:
            try:
                self.vq.load_from_ckpt(vq_ckpt)
            except Exception:
                # Try loading state_dict if different format
                sd = torch.load(vq_ckpt, map_location="cpu")
                if isinstance(sd, dict) and "model" in sd:
                    self.vq.load_state_dict(sd["model"])
                else:
                    self.vq.load_state_dict(sd)
        self.vq.eval()

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Try cache
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"latent_{idx:08d}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path)

        # Get raw video: [n, c, h, w], values in [-1,1]
        video = self.base[idx]

        # Move to device and rescale to [-1,1] already; VQ-VAE expects [-1,1]
        with torch.no_grad():
            v = video.to(self.device).float()
            # Flatten frames to batch for encoder
            b = v.shape[0]
            flat = v
            # If input in [-1,1], keep as-is
            encoded = self.vq.encoder(flat)
            # Quantize latents
            _, quantized_latents, _ = self.vq.vector_quantizer(encoded)

            # quantized_latents: [n, e_dim, h_lat, w_lat]
            # Return as [n, e_dim, h_lat, w_lat] on cpu
            out = quantized_latents.cpu()

        if self.cache_dir is not None:
            torch.save(out, cache_path)

        return out
