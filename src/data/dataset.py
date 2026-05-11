import os
import random
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None #To block warning on size of an image

class RRDataset(Dataset):
    """
    Custom Dataset for Real vs. Fake Image Classification with Transform Labels.
    Expects a directory structure like:
    - root_dir/
        - original/
            - ai/
            - real/
        - transfer/
            - ai/
            - real/
        - redigital/
            - ai/
            - real/
    Each subfolder should contain the respective images.
    
    Args:
        root_dir (str): Root directory of the dataset.
        samples_per_class (int): Number of samples to load per class (to ensure balance).
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir, samples_per_class=1000, transform=None, seed=42):    
        """Initialize the dataset by crawling the directory structure and loading a balanced subset of image paths and labels."""
        self.root_dir = root_dir
        self.transform = transform
        
        self.samples = []  # Single list of tuples (image_path, rf_label, transform_label)
        
        transform_dict = {'original': 0, 'transfer': 1, 'redigital': 2}
        rf_dict = {'ai': 0, 'real': 1}
        
        random.seed(seed) # Ensure reproducibility
        
        # Crawl the directories
        for trans_folder in ['original', 'transfer', 'redigital']:
            for rf_folder in ['ai', 'real']:
                folder_path = os.path.join(root_dir, trans_folder, rf_folder)
                
                if not os.path.exists(folder_path):
                    print(f"Warning: Folder not found {folder_path}")
                    continue
                
                # Get all images in this specific sub-folder
                all_images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                # Randomly sample to ensure perfect balance
                sampled_images = random.sample(all_images, min(samples_per_class, len(all_images)))
                
                for img_name in sampled_images:
                    # Store the image path and labels as a tuple in a single list for easier access
                    self.samples.append(
                        (
                            os.path.join(folder_path, img_name),
                            rf_dict[rf_folder],
                            transform_dict[trans_folder]
                        )
                    )
                       
        print(f"Loaded {len(self.samples)} images.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""

        return len(self.samples)

    def __getitem__(self, idx):
        """Returns a tuple: (image, real/fake label, transform label)"""

        img_path, label_rf, label_trans = self.samples[idx]  # Unpack the tuple for the current index
        
        image = Image.open(img_path).convert("RGB") # Force RGB to avoid grayscale crashes
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_rf, label_trans