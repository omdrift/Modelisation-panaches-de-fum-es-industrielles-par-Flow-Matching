"""
PatchGAN Discriminator for VQGAN training.
Adapted from Taming Transformers: https://github.com/CompVis/taming-transformers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator for VQGAN adversarial training"""
    
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """
        Args:
            input_nc: Number of input channels (3 for RGB)
            ndf: Number of base filters
            n_layers: Number of discriminator layers
            use_actnorm: Use activation normalization instead of batch norm
        """
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers
        
        if use_actnorm:
            norm_layer = nn.Identity  # Actnorm not implemented, fallback
        else:
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


def adopt_weight(weight, global_step, threshold=0, value=0.):
    """
    Gradually increase discriminator weight during training.
    Helps stabilize early training by letting generator learn first.
    """
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    """
    Hinge loss for discriminator.
    Encourages D(real) > 1 and D(fake) < -1
    """
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    """
    Vanilla GAN loss for discriminator.
    Standard binary cross-entropy formulation.
    """
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


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
