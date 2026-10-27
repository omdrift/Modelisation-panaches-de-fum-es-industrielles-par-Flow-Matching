"""Nearest-neighbor VQ bottleneck with a straight-through encoder gradient."""
import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """Minimize codebook MSE plus beta times encoder commitment MSE."""

    def __init__(self, n_e: int, e_dim: int, beta: float):
        super().__init__()
        self.n_e, self.e_dim, self.beta = n_e, e_dim, beta
        self.embedding = nn.Embedding(n_e, e_dim)
        nn.init.uniform_(self.embedding.weight, -1 / n_e, 1 / n_e)

    def forward(self, latents: torch.Tensor):
        if latents.ndim != 4 or latents.shape[1] != self.e_dim:
            raise ValueError('Expected [batch, embedding_dimension, height, width] latents')
        vectors = latents.movedim(1, -1).contiguous()
        with torch.no_grad():
            distances = torch.cdist(vectors.reshape(-1, self.e_dim), self.embedding.weight)
            indices = distances.argmin(dim=1).reshape(vectors.shape[:-1])
        nearest = self.embedding(indices)
        commitment = F.mse_loss(vectors, nearest.detach())
        codebook = F.mse_loss(nearest, vectors.detach())
        loss = codebook + self.beta * commitment
        straight_through = vectors + (nearest - vectors).detach()
        codes = F.one_hot(indices, num_classes=self.n_e).to(vectors.dtype)
        return loss, straight_through.movedim(-1, 1).contiguous(), codes

    def get_latents_from_ids(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 4 or codes.shape[-1] != self.n_e:
            raise ValueError('Expected [batch, height, width, codebook_size] codes')
        return torch.einsum('bhwk,kc->bchw', codes.to(self.embedding.weight.dtype), self.embedding.weight)


def build_vector_quantizer(config: dict) -> VectorQuantizer:
    return VectorQuantizer(config['num_embeddings'], config['embedding_dimension'], float(config['commitment_cost']))
