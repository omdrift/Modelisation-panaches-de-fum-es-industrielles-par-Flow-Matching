"""
Loss functions for VQGAN training.
Includes perceptual loss (LPIPS) and adversarial losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS)
    Uses pre-trained VGG network to compute perceptual distance.
    
    This is a simplified version - for production, use the official lpips package:
    pip install lpips
    """
    
    def __init__(self, use_dropout=True):
        super(LPIPS, self).__init__()
        # Use VGG16 features
        from torchvision import models
        
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.slice1 = nn.Sequential(*list(vgg.features[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.features[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.features[9:16])) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg.features[16:23]))# relu4_3
        
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False
        
        # Learnable weights for each layer
        self.lins = nn.ModuleList([
            nn.Sequential(
                nn.Dropout() if use_dropout else nn.Identity(),
                nn.Conv2d(64, 1, 1, bias=False),
            ),
            nn.Sequential(
                nn.Dropout() if use_dropout else nn.Identity(),
                nn.Conv2d(128, 1, 1, bias=False),
            ),
            nn.Sequential(
                nn.Dropout() if use_dropout else nn.Identity(),
                nn.Conv2d(256, 1, 1, bias=False),
            ),
            nn.Sequential(
                nn.Dropout() if use_dropout else nn.Identity(),
                nn.Conv2d(512, 1, 1, bias=False),
            ),
        ])
        
        # Initialize weights to 1.0
        for lin in self.lins:
            if isinstance(lin[-1], nn.Conv2d):
                lin[-1].weight.data.fill_(1.0)
    
    def normalize_tensor(self, x):
        """Normalize to ImageNet stats"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std
    
    def forward(self, input, target):
        """
        Args:
            input: [B, 3, H, W] reconstructed image in range [-1, 1]
            target: [B, 3, H, W] target image in range [-1, 1]
        Returns:
            perceptual_loss: scalar
        """
        # Convert from [-1, 1] to [0, 1]
        input = (input + 1) / 2
        target = (target + 1) / 2
        
        # Normalize for VGG
        input = self.normalize_tensor(input)
        target = self.normalize_tensor(target)
        
        # Extract features
        h_input = [input]
        h_target = [target]
        
        for slice_fn in [self.slice1, self.slice2, self.slice3, self.slice4]:
            h_input.append(slice_fn(h_input[-1]))
            h_target.append(slice_fn(h_target[-1]))
        
        h_input = h_input[1:]
        h_target = h_target[1:]
        
        # Compute perceptual distance
        loss = 0
        for feat_input, feat_target, lin in zip(h_input, h_target, self.lins):
            # Normalize features
            feat_input = feat_input / (feat_input.norm(dim=1, keepdim=True) + 1e-10)
            feat_target = feat_target / (feat_target.norm(dim=1, keepdim=True) + 1e-10)
            
            # Compute weighted L2 distance
            diff = (feat_input - feat_target) ** 2
            diff = lin(diff)
            loss = loss + diff.mean()
        
        return loss


class SimplifiedLPIPS(nn.Module):
    """
    Simplified perceptual loss using frozen VGG features without learnable weights.
    More stable and easier to train.
    """
    
    def __init__(self):
        super(SimplifiedLPIPS, self).__init__()
        from torchvision import models
        
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16])) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg[16:23]))# relu4_3
        
        # Freeze all weights
        for param in self.parameters():
            param.requires_grad = False
        
        self.weights = [1.0, 1.0, 1.0, 1.0]  # Equal weighting
    
    def normalize_tensor(self, x):
        """Normalize to ImageNet stats"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std
    
    def forward(self, input, target):
        """
        Args:
            input: [B, 3, H, W] reconstructed image in range [-1, 1]
            target: [B, 3, H, W] target image in range [-1, 1]
        Returns:
            perceptual_loss: scalar
        """
        # Convert from [-1, 1] to [0, 1]
        input = (input + 1) / 2
        target = (target + 1) / 2
        
        # Normalize for VGG
        input = self.normalize_tensor(input)
        target = self.normalize_tensor(target)
        
        # Extract and compare features
        loss = 0
        feats_input = []
        feats_target = []
        
        for slice_fn in [self.slice1, self.slice2, self.slice3, self.slice4]:
            if len(feats_input) == 0:
                feats_input.append(slice_fn(input))
                feats_target.append(slice_fn(target))
            else:
                feats_input.append(slice_fn(feats_input[-1]))
                feats_target.append(slice_fn(feats_target[-1]))
        
        for feat_input, feat_target, weight in zip(feats_input, feats_target, self.weights):
            loss += weight * F.mse_loss(feat_input, feat_target)
        
        return loss


def build_perceptual_loss(simplified=True):
    """
    Factory function to build perceptual loss.
    
    Args:
        simplified: If True, use SimplifiedLPIPS. Otherwise use full LPIPS with learnable weights.
    
    Returns:
        Perceptual loss module
    """
    if simplified:
        return SimplifiedLPIPS()
    else:
        return LPIPS()
