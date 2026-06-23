import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_frequency_spectrum(x_rgb):
    """
    Computes the 2D FFT log-magnitude spectrum of an RGB image batch.
    Normalizes the spectrum per sample and channel to [0, 1] for training stability.
    Args:
        x_rgb (Tensor): Input batch of shape (B, C, H, W)
    Returns:
        Tensor: Normalized log-magnitude frequency spectrum of shape (B, C, H, W)
    """
    # 1. Compute 2D Fast Fourier Transform
    x_fft = torch.fft.fft2(x_rgb)
    
    # 2. Shift zero-frequency component to the center of the spectrum
    x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
    
    # 3. Calculate absolute magnitude (amplitude)
    magnitude = torch.abs(x_fft_shift)
    
    # 4. Apply log transform to scale values
    magnitude_log = torch.log(magnitude + 1e-8)
    
    # 5. Min-Max normalization per image and channel
    min_val = magnitude_log.amin(dim=(-2, -1), keepdim=True)
    max_val = magnitude_log.amax(dim=(-2, -1), keepdim=True)
    
    normalized_spectrum = (magnitude_log - min_val) / (max_val - min_val + 1e-8)
    
    return normalized_spectrum

class FrequencyBranch(nn.Module):
    """A lightweight CNN to extract features from the 2D frequency spectrum."""
    def __init__(self, in_channels=3):
        super(FrequencyBranch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 224 -> 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 -> 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # -> 128 channels
        )
        
    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1) # Output shape: (B, 128)

class DualBranchMTLModel(nn.Module):
    """
    Two-Stream Multi-Task Learning model combining:
    - Spatial Branch: Pretrained ResNet-50
    - Frequency Branch: Custom CNN processing the normalized FFT spectrum
    """
    def __init__(self):
        super(DualBranchMTLModel, self).__init__()
        
        # 1. Spatial Branch: Pretrained ResNet-50
        self.spatial_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        spatial_features = self.spatial_backbone.fc.in_features # 2048
        self.spatial_backbone.fc = nn.Identity() # Remove default head
        
        # 2. Frequency Branch: Custom CNN
        self.frequency_backbone = FrequencyBranch()
        freq_features = 128
        
        # Total combined features: 2048 (ResNet) + 128 (FFT-CNN) = 2176
        combined_features = spatial_features + freq_features
        
        # 3. Dual Classification Heads
        self.head_real_fake = nn.Linear(combined_features, 1) # Task 1: Real vs Fake (BCE)
        self.head_transform = nn.Linear(combined_features, 3) # Task 2: Transformation type (CE)
        
    def forward(self, x):
        # Stream 1: Spatial features
        feat_spatial = self.spatial_backbone(x) # Shape: (B, 2048)
        
        # Stream 2: Frequency features
        freq_spectrum = get_frequency_spectrum(x) # Shape: (B, 3, 224, 224)
        feat_freq = self.frequency_backbone(freq_spectrum) # Shape: (B, 128)
        
        # Fusion: Concatenate feature vectors
        fused_features = torch.cat([feat_spatial, feat_freq], dim=1) # Shape: (B, 2176)
        
        # Compute Task Logits
        logits_rf = self.head_real_fake(fused_features) # Shape: (B, 1)
        logits_tf = self.head_transform(fused_features) # Shape: (B, 3)
        
        return logits_rf, logits_tf

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict to handle backward compatibility:
        - Maps older checkpoint keys starting with 'backbone.' to 'spatial_backbone.'
        - Handles head weight size mismatches (pads 2048 features to 2176 with zero weights)
        - Gracefully loads old single-stream checkpoints using strict=False
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
            
        # 2. Handle size mismatch in the classification heads
        current_sd = self.state_dict()
        for head_name in ["head_real_fake.weight", "head_transform.weight"]:
            if head_name in state_dict and head_name in current_sd:
                chk_shape = state_dict[head_name].shape
                curr_shape = current_sd[head_name].shape
                if chk_shape != curr_shape:
                    print(f"Adapting weight shape for {head_name} from {chk_shape} to {curr_shape} (padding new channels with zero)...")
                    new_weight = torch.zeros(curr_shape, device=state_dict[head_name].device)
                    new_weight[:, :chk_shape[1]] = state_dict[head_name]
                    state_dict[head_name] = new_weight
                    
        # 3. If loading an old checkpoint, use strict=False to ignore missing frequency stream keys
        is_old = (not has_spatial and has_old_backbone)
        return super(DualBranchMTLModel, self).load_state_dict(state_dict, strict=strict if not is_old else False)

