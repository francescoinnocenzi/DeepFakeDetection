import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import math

def get_frequency_spectrum(x_rgb):
    """
    Computes the 2D FFT log-magnitude and phase spectrum of an RGB image batch.
    Normalizes magnitude to [0, 1] per sample/channel, and phase linearly to [0, 1].
    Args:
        x_rgb (Tensor): Input batch of shape (B, C, H, W)
    Returns:
        Tensor: Concatenated spectrum of shape (B, 2*C, H, W) containing:
                - Normalized log-magnitude (channels 0-2)
                - Normalized phase (channels 3-5)
    """
    # 1. Compute 2D Fast Fourier Transform
    x_fft = torch.fft.fft2(x_rgb)
    
    # 2. Shift zero-frequency component to the center of the spectrum
    x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
    
    # 3. Calculate absolute magnitude (amplitude) and apply log transform
    magnitude = torch.abs(x_fft_shift)
    magnitude_log = torch.log(magnitude + 1e-8)
    
    # Min-Max normalization per image and channel for magnitude
    min_val = magnitude_log.amin(dim=(-2, -1), keepdim=True)
    max_val = magnitude_log.amax(dim=(-2, -1), keepdim=True)
    normalized_magnitude = (magnitude_log - min_val) / (max_val - min_val + 1e-8)
    
    # 4. Calculate phase and normalize linearly from [-pi, pi] to [0, 1]
    phase = torch.angle(x_fft_shift)
    normalized_phase = (phase + math.pi) / (2.0 * math.pi)
    
    # 5. Concatenate normalized magnitude and phase along the channel dimension
    normalized_spectrum = torch.cat([normalized_magnitude, normalized_phase], dim=1)
    
    return normalized_spectrum

class FrequencyBranch(nn.Module):
    """A lightweight CNN to extract features from the 2D frequency spectrum."""
    def __init__(self, in_channels=6):
        super(FrequencyBranch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 224 -> 112
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 -> 56
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 -> 28
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # -> 512 channels
        )
        
    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1) # Output shape: (B, 512)

class DualBranchMTLModel(nn.Module):
    """
    Two-Stream Multi-Task Learning model combining:
    - Spatial Branch: Pretrained ResNet-50
    - Frequency Branch: Custom CNN processing the normalized FFT magnitude + phase spectrum
    """
    def __init__(self):
        super(DualBranchMTLModel, self).__init__()
        
        # 1. Spatial Branch: Pretrained ResNet-50
        self.spatial_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        spatial_features = self.spatial_backbone.fc.in_features # 2048
        self.spatial_backbone.fc = nn.Identity() # Remove default head
        
        # 2. Frequency Branch: Custom CNN
        self.frequency_backbone = FrequencyBranch(in_channels=6)
        freq_features = 512
        
        # Total combined features: 2048 (ResNet) + 512 (FFT-CNN) = 2560
        combined_features = spatial_features + freq_features
        
        # 3. Dual Classification Heads
        self.head_real_fake = nn.Linear(combined_features, 1) # Task 1: Real vs Fake (BCE)
        self.head_transform = nn.Linear(combined_features, 3) # Task 2: Transformation type (CE)
        
    def forward(self, x):
        # Stream 1: Spatial features
        feat_spatial = self.spatial_backbone(x) # Shape: (B, 2048)
        
        # Stream 2: Frequency features
        freq_spectrum = get_frequency_spectrum(x) # Shape: (B, 6, 224, 224)
        feat_freq = self.frequency_backbone(freq_spectrum) # Shape: (B, 512)
        
        # Fusion: Concatenate feature vectors
        fused_features = torch.cat([feat_spatial, feat_freq], dim=1) # Shape: (B, 2560)
        
        # Compute Task Logits
        logits_rf = self.head_real_fake(fused_features) # Shape: (B, 1)
        logits_tf = self.head_transform(fused_features) # Shape: (B, 3)
        
        return logits_rf, logits_tf

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict to handle backward compatibility:
        - Maps older checkpoint keys starting with 'backbone.' to 'spatial_backbone.'
        - Handles head weight size mismatches (pads older checkpoints like 2048 or 2176 features to 2560 with zero weights)
        - Drops mismatching keys from other components (such as older frequency backbones) and switches to strict=False
        """
        # 1. Map old backbone keys to spatial_backbone keys
        has_spatial = any(k.startswith("spatial_backbone.") for k in state_dict.keys())
        has_old_backbone = any(k.startswith("backbone.") for k in state_dict.keys())
        
        if not has_spatial and has_old_backbone:
            print("Mapping old checkpoint keys to new architecture...")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith("backbone."):
                    new_key = k.replace("backbone.", "spatial_backbone.")
                new_state_dict[new_key] = v
            state_dict = new_state_dict
            
        # 2. Inspect state_dict keys for shape mismatches and handle them
        current_sd = self.state_dict()
        keys_to_delete = []
        is_modified = False
        
        for k in list(state_dict.keys()):
            if k in current_sd:
                chk_shape = state_dict[k].shape
                curr_shape = current_sd[k].shape
                if chk_shape != curr_shape:
                    if k in ["head_real_fake.weight", "head_transform.weight"]:
                        print(f"Adapting weight shape for {k} from {chk_shape} to {curr_shape} (padding new channels with zero)...")
                        new_weight = torch.zeros(curr_shape, device=state_dict[k].device)
                        min_in_features = min(chk_shape[1], curr_shape[1])
                        new_weight[:, :min_in_features] = state_dict[k][:, :min_in_features]
                        state_dict[k] = new_weight
                        is_modified = True
                    else:
                        print(f"Shape mismatch for {k}: checkpoint shape {chk_shape}, model shape {curr_shape}. Excluding this key from loading.")
                        keys_to_delete.append(k)
                        is_modified = True
                        
        for k in keys_to_delete:
            del state_dict[k]
                    
        # 3. If loading an old checkpoint or we filtered incompatible keys, use strict=False
        use_strict = strict if (not is_modified and has_spatial and not has_old_backbone) else False
        return super(DualBranchMTLModel, self).load_state_dict(state_dict, strict=use_strict)

