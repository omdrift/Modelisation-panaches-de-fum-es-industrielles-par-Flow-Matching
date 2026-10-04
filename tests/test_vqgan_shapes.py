import torch

from lutils.dict_wrapper import DictWrapper
from model.autoencoder_adapter import AutoencoderAdapter
from model.layers.utils import SequenceConverter
from model.vqgan.vqvae import build_vqvae


def test_vqgan_encoder_latent_and_decoder_shapes():
    config = DictWrapper(
        {
            "encoder": {"in_channels": 3, "out_channels": 256, "mid_channels": 128},
            "decoder": {"in_channels": 256, "out_channels": 3, "mid_channels": 128},
            "vector_quantizer": {
                "embedding_dimension": 256,
                "num_embeddings": 1024,
                "commitment_cost": 0.1,
            },
        }
    )
    model = build_vqvae(config).eval()

    with torch.inference_mode():
        images = torch.randn(1, 3, 64, 64)
        encoded = model.encoder(images)
        output = model(images)
        adapter = AutoencoderAdapter(SequenceConverter(model), "ours")
        latents = adapter.encode(images.unsqueeze(1))
        decoded = adapter.decode(latents)

    assert encoded.shape == (1, 256, 8, 8)
    assert output.quantized_latents.shape == (1, 256, 8, 8)
    assert output.reconstructed_images.shape == (1, 3, 64, 64)
    assert latents.shape == (1, 1, 256, 8, 8)
    assert decoded.shape == (1, 1, 3, 64, 64)
