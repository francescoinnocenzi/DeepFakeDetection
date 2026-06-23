import os
from src.data.dataloader import get_dataloaders

def clean_filename(filename):
    if filename.startswith("transfer_"):
        filename = filename[len("transfer_"):]
    elif filename.startswith("redigital_"):
        filename = filename[len("redigital_"):]
    parent_id, _ = os.path.splitext(filename)
    return parent_id

# Load the loaders
train_loader, val_loader = get_dataloaders(data_dir="src/data/RRDataset_final", batch_size=64)

# Extract parent IDs for Train
train_parents = set()
train_dataset = train_loader.dataset
for idx in train_dataset.indices:
    filepath = train_dataset.dataset.samples[idx][0]
    train_parents.add(clean_filename(os.path.basename(filepath)))

# Extract parent IDs for Val
val_parents = set()
val_dataset = val_loader.dataset
for idx in val_dataset.indices:
    filepath = val_dataset.dataset.samples[idx][0]
    val_parents.add(clean_filename(os.path.basename(filepath)))

# Calculate overlap
overlap = train_parents.intersection(val_parents)

print("\n--- NEW LEAKAGE VERIFICATION ---")
print(f"Unique Train Parent Images: {len(train_parents)}")
print(f"Unique Val Parent Images:   {len(val_parents)}")
print(f"LEAKED PARENT IMAGES:       {len(overlap)}")

if len(overlap) == 0:
    print("SUCCESS: Zero leakage detected. Group splitting is working perfectly!")
else:
    print(f"WARNING: Leakage detected! {len(overlap)} parent images overlap.")
