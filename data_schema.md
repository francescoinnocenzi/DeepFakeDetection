***

### The Shared API Contract (The Agreement)

This is the bridge between Peer 1’s data and Peer 2’s neural network. Both of your scripts must strictly respect these formats.

**1. Label Mappings**
*   **Task 1 (Real/Fake):** `0` = AI-Generated, `1` = Real Photograph
*   **Task 2 (Transformation):** `0` = Original, `1` = Internet-Transmitted, `2` = Re-digitized

**2. Tensor Specifications**
Assuming `B` is your batch size (e.g., 32 or 64).

| Variable | Provided By | PyTorch `dtype` | Tensor Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Peer 1 | `torch.float32` | `(B, 3, 224, 224)` | Resized & Normalized RGB images. |
| `labels_real_fake` | Peer 1 | `torch.float32` | `(B, 1)` | Float format for `BCEWithLogitsLoss`. |
| `labels_transform` | Peer 1 | `torch.long` | `(B,)` | 1D Integer format for `CrossEntropyLoss`. |
| `logits_real_fake` | Peer 2 | `torch.float32` | `(B, 1)` | Raw network output (No Sigmoid). |
| `logits_transform` | Peer 2 | `torch.float32` | `(B, 3)` | Raw network output (No Softmax). |

---

### Peer 1: Data & Analytics Architect
**Primary Deliverable:** A PyTorch `DataLoader` that outputs the exact tensors defined above, and functions to evaluate the results.

**Independent Tasks:**
1.  **Data Subsetting & Balancing:** The RRDataset is massive. Write a script to randomly sample a subset of the dataset that fits your GPU constraints (e.g., 1,000 images per class). You *must* ensure a strict 50/50 balance between Real and AI within all three transformation categories.
2.  **The Transform Pipeline:** Build the `torchvision.transforms` composition. You must force all images (even the weird 764x764 ones) to exactly `224x224`, convert them to tensors, and apply standard ImageNet normalization.
3.  **Custom Dataset Class:** Write the logic that maps the file paths to the correct integer labels (`0`, `1`, `2`) based on the folder structures.
4.  **Evaluation Metrics:** Build functions that take Peer 2's raw mock logits, apply Sigmoid/Softmax, convert them to predictions, and calculate Accuracy, Precision, Recall, and F1-Scores.

---

### Peer 2: Model & Training Engineer
**Primary Deliverable:** A PyTorch multi-task neural network and a training loop that processes `(B, 3, 224, 224)` images.

**Independent Tasks:**
1.  **Shared Backbone:** Load a pre-trained ImageNet model (e.g., `resnet50`). Strip off its final classification layer.
2.  **Dual Heads:** Attach two separate `nn.Linear` layers to the backbone: one outputting 1 feature (Real/Fake), and one outputting 3 features (Transformation).
3.  **Combined Loss Function:** Build the training loop. Calculate `BCEWithLogitsLoss` for Task 1 and `CrossEntropyLoss` for Task 2. Combine them: $Total\_Loss = (\alpha \times Loss_{real/fake}) + (\beta \times Loss_{transform})$.
4.  **Ablation Engine:** Write a script to automate training runs using different values for $\alpha$ and $\beta$ (e.g., 0.5/0.5, 0.8/0.2) and log the accuracy trade-offs.

---
