"""
PatchGAN Discriminator for VQGAN training.
Adapted from Taming Transformers: https://github.com/CompVis/taming-transformers
"""
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator for VQGAN adversarial training"""
    
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """
        Args:
            input_nc: Number of input channels (3 for RGB)
            ndf: Number of base filters
            n_layers: Number of discriminator layers
        """
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers
        
        norm_layer = nn.BatchNorm2d

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Final layer outputs a single-channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)
        self.apply(weights_init)
    
    def forward(self, input):
        """
        Args:
            input: [B, C, H, W] image tensor
        Returns:
            [B, 1, H', W'] prediction map (higher = more real)
        """
        return self.model(input)


def build_discriminator(input_nc=3, ndf=64, n_layers=3):
    """
    Factory function to build discriminator.
    
    Args:
        input_nc: Number of input channels
        ndf: Base number of filters
        n_layers: Number of convolutional layers
    
    Returns:
        NLayerDiscriminator instance
    """
    return NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers)
