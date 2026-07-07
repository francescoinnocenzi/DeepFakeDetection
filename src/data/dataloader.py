import os
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
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
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
            
        # Convert to tensor and normalize
        if self.dataset.to_tensor_transform:
            image = self.dataset.to_tensor_transform(image)
            
        return image, label_rf, label_trans

def get_dataloaders(
    data_dir,
    batch_size=BATCH_SIZE,
    total_samples_per_class=SAMPLES_PER_CLASS,
    train_split=TRAIN_SPLIT,
    val_split=VAL_SPLIT,
    test_split=TEST_SPLIT,
):
    """
    Get DataLoaders for the Real vs. Fake Image Classification task.
    Args:
        data_dir (str): Root directory of the dataset.
        batch_size (int): Batch size for the DataLoader.
        total_samples_per_class (int): Total number of samples per class.
        train_split (float): Proportion of parent IDs used for training.
        val_split (float): Proportion of parent IDs used for validation.
        test_split (float): Proportion of parent IDs used for testing. Only used to sanity-check
            that train_split + val_split + test_split == 1.0; the test set itself is still built
            from the remainder of parent IDs after the train/val slices.
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation, and final evaluation.
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, (
        f"train_split ({train_split}) + val_split ({val_split}) + test_split ({test_split}) "
        f"must sum to 1.0"
    )

    # Load the balanced subset without any base transforms (applied in SubsetWrapper instead)
    full_dataset = RRDataset(
        root_dir=data_dir,
        samples_per_class=total_samples_per_class,
        transform=None,
        seed=RANDOM_SEED,
    )
    
    # Extract parent IDs for all samples to prevent data leakage
    parent_ids = []
    for filepath, _, _ in full_dataset.samples:
        filename = os.path.basename(filepath)
        if filename.startswith("transfer_"):
            filename = filename[len("transfer_"):]
        elif filename.startswith("redigital_"):
            filename = filename[len("redigital_"):]
        parent_id, _ = os.path.splitext(filename)
        parent_ids.append(parent_id)
        
    unique_parents = sorted(list(set(parent_ids)))
    
    import random as py_random
    rng = py_random.Random(RANDOM_SEED)
    rng.shuffle(unique_parents)
    
    train_end = int(train_split * len(unique_parents))
    val_end   = train_end + int(val_split * len(unique_parents))
    train_parents = set(unique_parents[:train_end])
    val_parents   = set(unique_parents[train_end:val_end])

    # Partition indices based on unique parent IDs
    train_indices = [idx for idx, pid in enumerate(parent_ids) if pid in train_parents]
    val_indices   = [idx for idx, pid in enumerate(parent_ids) if pid in val_parents]
    test_indices  = [idx for idx, pid in enumerate(parent_ids) if pid not in train_parents and pid not in val_parents]

    # Wrap with DatasetSubset to apply train/val/test specific transforms
    train_dataset = DatasetSubset(full_dataset, train_indices, train_transforms)
    val_dataset   = DatasetSubset(full_dataset, val_indices,   val_transforms)
    test_dataset  = DatasetSubset(full_dataset, test_indices,  val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    return train_loader, val_loader, test_loader