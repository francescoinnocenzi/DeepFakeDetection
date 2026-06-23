import torch
from torch.utils.data import DataLoader, random_split
from .dataset import RRDataset
from .transforms import train_transforms, val_transforms
from PIL import Image
from ..globals import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    SAMPLES_PER_CLASS,
    TRAIN_VAL_SPLIT,
)

class DatasetSubset(torch.utils.data.Dataset):
    """
    Wrapper dataset to apply separate transforms to train/validation splits.
    This avoids applying random augmentations (like crop and flip) to the validation set.
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label_rf, label_trans = self.dataset.samples[real_idx]
        
        image = Image.open(img_path).convert("RGB")
        
        # Apply the specific base transform (e.g. train_transforms or val_transforms)
        if self.transform:
            image = self.transform(image)
            
        # Apply degradation transforms only for transfer class
        if self.dataset.degradation_transform and label_trans == 1:
            image = self.dataset.degradation_transform(image)
            
        # Convert to tensor and normalize
        if self.dataset.to_tensor_transform:
            image = self.dataset.to_tensor_transform(image)
            
        return image, label_rf, label_trans

def get_dataloaders(
    data_dir,
    batch_size=BATCH_SIZE,
    total_samples_per_class=SAMPLES_PER_CLASS,
    train_val_split=TRAIN_VAL_SPLIT,
):
    """
    Get DataLoaders for the Real vs. Fake Image Classification task.
    Args:
        data_dir (str): Root directory of the dataset.
        batch_size (int): Batch size for the DataLoader.
        total_samples_per_class (int): Total number of samples per class.
        train_val_split (float): Proportion of samples used for training.
    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    # Load the balanced subset without any base transforms (applied in SubsetWrapper instead)
    full_dataset = RRDataset(
        root_dir=data_dir,
        samples_per_class=total_samples_per_class,
        transform=None,
        seed=RANDOM_SEED,
    )
    
    # Split into Train and Validation indices
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    # Wrap with DatasetSubset to apply train/val specific transforms
    train_dataset = DatasetSubset(full_dataset, train_subset.indices, train_transforms)
    val_dataset = DatasetSubset(full_dataset, val_subset.indices, val_transforms)
    
    # Create the Loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS,      # Aumentato per caricamento parallelo
        pin_memory=PIN_MEMORY         # Ottimizzato per la tua RTX 5070
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    return train_loader, val_loader