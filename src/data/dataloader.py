import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .dataset import RRDataset
from ..globals import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    SAMPLES_PER_CLASS,
    TRAIN_VAL_SPLIT,
)

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
    # Transform Pipeline
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
    ])
    
    # Load the balanced subset
    full_dataset = RRDataset(
        root_dir=data_dir,
        samples_per_class=total_samples_per_class,
        transform=data_transform,
        seed=RANDOM_SEED,
    )
    
    # Split into Train and Validation (e.g., 80% train, 20% val)
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
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