import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root_dir = "src/data/RRDataset_final"
parent_id = "Culture_&_Religion_000001"

# Define paths dynamically
original_path = os.path.join(root_dir, "original", "ai", f"{parent_id}.png")
if not os.path.exists(original_path):
    original_path = os.path.join(root_dir, "original", "ai", f"{parent_id}.jpg")

transfer_path = os.path.join(root_dir, "transfer", "ai", f"transfer_{parent_id}.jpg")
redigital_path = os.path.join(root_dir, "redigital", "ai", f"redigital_{parent_id}.jpg")

paths = [original_path, transfer_path, redigital_path]
titles = ["Original", "Internet-Transmitted", "Re-digitized"]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for ax, path, title in zip(axes, paths, titles):
    if os.path.exists(path):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(f"{title}\nResolution: {img.size[0]}x{img.size[1]}\nFormat: {img.format}", fontsize=11, fontweight='bold', pad=10)
    else:
        ax.text(0.5, 0.5, f"File Not Found:\n{os.path.basename(path)}", 
                ha='center', va='center', fontsize=12, color='red')
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle(f"Comparison of Parent Image: {parent_id}", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('parent_image_comparison.png', bbox_inches='tight', dpi=300)
print("Trio comparison plot saved successfully as 'parent_image_comparison.png'.")
