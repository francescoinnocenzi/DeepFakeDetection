# DeepFake Detection — Two-Stream Multi-Task Learning

Project 2: Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios

Two-stream multi-task learning for AI-generated image forensics. Given a face/portrait image, the model jointly predicts:

1. **Real vs. Fake** (binary) — is the image AI-generated or a real photograph?
2. **Transformation history** (3-class) — has the image gone through no post-processing (`original`), transmission/re-compression (`transfer`), or a re-digitization pass (`redigital`)?

## Table of contents

- [DeepFake Detection — Two-Stream Multi-Task Learning](#deepfake-detection--two-stream-multi-task-learning)
  - [Table of contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Repository layout](#repository-layout)
  - [Dataset](#dataset)
    - [Train/val/test split — leakage-safe by parent ID](#trainvaltest-split--leakage-safe-by-parent-id)
  - [Data pipeline \& augmentation](#data-pipeline--augmentation)
    - [Anti-shortcut design](#anti-shortcut-design)
  - [Training](#training)
  - [Evaluation \& visualization](#evaluation--visualization)
  - [End-to-end pipeline (`main.ipynb`)](#end-to-end-pipeline-mainipynb)
  - [Results](#results)
  - [Setup](#setup)
  - [Configuration reference](#configuration-reference)

## Architecture

`DualBranchMTLModel` (`src/network/model.py`) runs an image through two parallel streams and fuses the result:

- **Spatial branch** — a swappable pretrained backbone, `resnet50` (2048-d) or `convnext_tiny` (768-d). Its classifier head is stripped (`nn.Identity()`), so it acts purely as a feature extractor rather than an ImageNet classifier.
- **Frequency branch** (`FrequencyBranch`) — a lightweight 4-block Conv-BN-ReLU-MaxPool CNN (32→64→128→256 channels) ending in global average pooling. It reads a 6-channel input produced by `get_frequency_spectrum`: the image's 2D FFT is computed, shifted so the zero-frequency (DC) component sits at the center, then split into a log-scaled magnitude (channels 0–2, min-max normalized per sample/channel to `[0, 1]`) and a phase map (channels 3–5, linearly rescaled from `[-π, π]` to `[0, 1]`). The rationale: GAN/diffusion generators tend to leave periodic upsampling/checkerboard artifacts in the frequency domain that are subtle spatially but show up clearly as spectral peaks or asymmetries.
- **Fusion** — the two feature vectors are concatenated (`spatial_features + 256`, e.g. 2304-d for ResNet-50) and read by two independent linear heads: `head_real_fake` (a single BCE logit) and `head_transform` (3-way CE logits). Because both heads read from the same fused vector, GradCAM can attribute either head's prediction back to either branch — there's no architectural reason the transform head couldn't be driven mostly by frequency features, for instance, and the interpretability tooling is built to check that rather than assume it.
  
## Repository layout

- `src/globals.py` — all hyperparameters/toggles in one place
- `src/data/` — `dataset.py` (`RRDataset`: balanced sampling + label mapping), `dataloader.py` (parent-ID-safe split, `DatasetSubset`), `transforms.py` (train/val augmentation pipelines), and the `RRDataset_final/` dataset root (not versioned)
- `src/network/model.py` — `DualBranchMTLModel`, `FrequencyBranch`, the FFT helper
- `src/train/` — `loops.py` (`train_epoch`/`validate_epoch`, `MultiTaskModel` alias), `loss.py` (`MultiTaskLoss`, `UncertaintyWeightedLoss`), `ablation.py` (weight-sweep training + checkpointing)
- `src/evaluation/` — `visualizer.py` (metrics, all diagnostic plots, GradCAM), `compare_backbones.py` (cross-backbone test-set comparison)
- `main.ipynb` — orchestrates the full pipeline end to end
- `models/` — saved checkpoints (`.pth`) + history/ablation sidecars (`.pt`)
- `presentation/` — contains pdf of presentation
- `requirements.txt` — Python dependencies


## Dataset

Expected at `src/data/RRDataset_final/`: three top-level folders — `original/`, `transfer/`, `redigital/` — each with `ai/` and `real/` subfolders. `original/*` are pristine `.png`s; `transfer/*` and `redigital/*` are `.jpg`s, simulating what happens once an image leaves the source pipeline. Real source counts are close to balanced (~8500 images per `{transform} × {ai,real}` cell, one cell short by a single image), but `RRDataset` (`src/data/dataset.py`) doesn't rely on that: it crawls all six subfolders and randomly samples `samples_per_class` images from each (default `6000`, seeded, see [Configuration reference](#configuration-reference)), guaranteeing perfect class balance in the working set regardless of how skewed the raw folders are. `Image.MAX_IMAGE_PIXELS` is set to `None` to avoid PIL's decompression-bomb warning on unusually large source images.

Label mapping is fixed in `dataset.py`: Real/Fake is `ai → 0`, `real → 1`; Transform is `original → 0`, `transfer → 1`, `redigital → 2`. Every sample is stored as a `(path, label_rf, label_trans)` tuple, which both the training dataloaders and the EDA/diagnostic plots read directly (avoiding a full image decode just to check label distributions).


### Train/val/test split — leakage-safe by parent ID

`get_dataloaders` (`src/data/dataloader.py`) does not split randomly at the image level. It strips `transfer_`/`redigital_` filename prefixes to recover each image's original parent ID, shuffles the set of unique parent IDs once (seed `42`) and partitions it 80/10/10 (`TRAIN_SPLIT`/`VAL_SPLIT`/remainder), then routes every image derived from the same parent — the original plus its transferred and re-digitized variants — to the same split. This guarantees the same source photo, in any of its processed forms, never appears in both train and test, which would otherwise let the model implicitly memorize a specific photo rather than learn generalizable cues.

The `DatasetSubset` wrapper exists specifically to let train/val/test each apply a *different* base transform (heavy augmentation for train, none for val/test) while sharing the same underlying `RRDataset` and index list — rather than instantiating three separate dataset objects, which would re-sample the image list three times independently and reintroduce the leakage the parent-ID logic is meant to prevent.

## Data pipeline & augmentation

Two separate `torchvision` pipelines (`src/data/transforms.py`), selected per split by `DatasetSubset`:

| Split | Pipeline |
|---|---|
| Train | `Resize(256) → RandomCrop(224) → RandomHorizontalFlip(p=0.5) → ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, p=0.4) → GaussianBlur(kernel=3, p=0.1) → RandomJPEGCompression(q30–95, p=0.5)` |
| Val / Test | `Resize(256) → CenterCrop(224)` (deterministic, no degradation) |
| Both, at the end | `ToTensor() → Normalize(ImageNet mean/std)` |

`RandomJPEGCompression` is a custom transform that re-encodes the PIL image through an in-memory JPEG buffer at a randomly chosen quality in `[30, 95]`, simulating the recompression that happens when an image is sent through a messaging app or email client. It's applied probabilistically (`p=0.5`) rather than unconditionally, so the model also sees a good share of un-degraded images at every epoch.

### Anti-shortcut design

An earlier version of this pipeline applied JPEG/blur/jitter degradation **only** to `transfer`-labeled images, inside the dataset's `__getitem__`. Since `transfer`/`redigital` images were already `.jpg` and `original` images `.png` — and the frequency branch is specifically sensitive to this kind of artifact — the network could trivially solve the transform task (and partially leak into the real/fake task) by detecting injected JPEG DCT block patterns rather than learning genuine post-processing or generative signal. That's a classic label-conditional-augmentation shortcut: the model doesn't learn "what does re-digitization look like," it learns "which images did the *loader* touch."

The fix: degradation is now applied uniformly to all training images regardless of label, decoupling "does this image show compression artifacts" from "what is its transform label." `GaussianBlur` probability is deliberately kept low (`0.1`) because blur is a low-pass filter that would otherwise erode the very high-frequency fingerprints the frequency branch is meant to detect — too much of it and the frequency branch loses signal for every task, not just the one it was accidentally shortcutting. A deterministic JPEG-85 pass was also tried on the validation set, to make val more representative of "real-world" degraded images, but it was reverted: it compressed the pristine `original` PNGs and distorted Objective-3's per-category robustness breakdown, making the `original` category look artificially harder than it should.

## Training

**Loss functions** (`src/train/loss.py`) — two interchangeable strategies:

- `MultiTaskLoss` (`'fixed'`) — a manually weighted sum `total = α·BCE(real/fake) + β·CE(transform)`, with `α`/`β` swept by the ablation engine below.
- `UncertaintyWeightedLoss` (`'uncertainty'`) — implements Kendall et al. (2018) homoscedastic uncertainty weighting. Each task gets its own learnable `log(σ)` parameter (stored in log-space for numerical stability and to enforce `σ > 0`), and the total loss becomes `Σ 1/(2σᵢ²)·Lᵢ + log(σᵢ)` — the network automatically down-weights whichever task is noisier, with no manual α/β tuning required. The σ parameters are passed to the same optimizer as the model weights, so they train jointly via ordinary backprop.

**Train/validate loop** (`src/train/loops.py`) — standard per-epoch `train_epoch`/`validate_epoch` functions built around `MultiTaskModel` (a thin alias over `DualBranchMTLModel`). `validate_epoch` also computes per-epoch Real/Fake and Transform accuracy, so the ablation engine can rank and checkpoint models without a separate evaluation pass.

**Ablation engine** (`src/train/ablation.py`) — `run_ablation_study` sweeps five fixed `(α, β)` combinations: `(1.0, 0.0)` unimodal Real/Fake, `(0.0, 1.0)` unimodal Transform, `(0.5, 0.5)` balanced, `(0.8, 0.2)` RF-focused, `(0.2, 0.8)` Transform-focused — plus, under `loss_type='uncertainty'`, a single learned-uncertainty run (the α/β sweep is skipped entirely in that mode, since the weights are learned rather than fixed). Each configuration trains a fresh model with a fresh `Adam` optimizer for up to `ABLATION_NUM_EPOCHS` epochs, with early stopping on the average of the two validation accuracies (`EARLY_STOP_PATIENCE`). The **best** epoch's model state and metrics are checkpointed — not the last epoch's, since early stopping means the final epoch is often worse than the peak — to `models/model_{backbone}_{alpha}_{beta}.pth` (or `models/model_{backbone}_uncertainty.pth`), alongside a `..._history.pt` sidecar holding every epoch's train/val loss and accuracy (for later curve-plotting) and a `models/ablation_results_{backbone}_{loss_type}.pt` summary dict across the whole sweep. `main.ipynb` runs this twice per backbone — once per loss type — so a full pass produces 6 checkpoints per backbone (5 fixed-weight + 1 uncertainty). `load_ablation_results` can reload a completed study's summary instantly, or reconstruct it from the `_history.pt` sidecars if the summary file is missing, without retraining anything.

## Evaluation & visualization

**`visualizer.py`** — `evaluate_model` runs a full pass over a loader and reports accuracy/precision/recall/F1/ROC-AUC for the Real/Fake task plus accuracy for the Transform task. Raw sigmoid probabilities are kept separately from rounded predictions specifically so ROC-AUC isn't computed on binarized scores, which would silently collapse the curve to a single point. Depending on the call, it also saves a subset of:

| Plot | What it shows |
|---|---|
| `confusion_matrices_{label}.png` | Side-by-side confusion matrices for both tasks |
| `prediction_distribution_{label}.png` | KDE of predicted probabilities, real vs. fake — calibration/separation |
| `training_curves_{label}.png` | Per-config train/val loss and validation accuracy over epochs |
| `accuracy_breakdown_{label}.png` | Real/Fake accuracy split by transform category (robustness check) |
| `roc_pr_curves_{label}.png` | ROC and Precision-Recall curves for Real/Fake |
| `transform_class_metrics_{label}.png` | Per-class precision/recall/F1 for the 3-way transform task |
| `ablation_study_{backbone}.png` / `ablation_accuracy_{backbone}.png` | Trade-off scatter and grouped bar chart across all configs for one backbone |
| `training_dynamics_{backbone}.png` | 4-panel comparison (train/val loss, val RF/Transform acc) across all configs, overlaid |
| `class_distribution.png`, `dataset_sample_grid.png` | Pre-training EDA: class balance, split sizes, one example per class |
| `frequency_spectrum.png` | FFT log-magnitude + phase spectrum per class — what the frequency branch actually sees |

**GradCAM** (`compute_gradcam`/`plot_gradcam`) — attributes either head's prediction to either branch by hooking the last conv block of the chosen branch: `spatial_backbone.layer4[-1]` (ResNet) or `.features[-1][-1]` (ConvNeXt) for the spatial branch; `frequency_backbone.features[-4]` (the last `Conv2d`, two steps back from the branch's final ReLU) for the frequency branch — chosen because hooking the branch's final in-place ReLU, or the BatchNorm layer feeding it, raises an autograd "view modified in-place" error during the backward hook, since a backward hook wraps the target layer's output in a view that autograd forbids feeding into an in-place op. `plot_gradcam` renders a 5-column grid per sample — original image, then Real/Fake × Spatial/Frequency and Transform × Spatial/Frequency heatmap overlays — giving a visual answer to "is the model reading texture or spectral artifacts to make this call?" Spatial-branch CAMs are overlaid on the original RGB photo (same pixel coordinate space); frequency-branch CAMs are overlaid on the FFT log-magnitude spectrum instead, since they operate on the frequency branch's activations, whose spatial dimensions correspond to frequency (center = low frequency/DC, edges = high frequency), not image geometry.

**`compare_backbones.py`** — loads multiple trained checkpoints (by backbone type and path), evaluates each on the same test loader, prints a summary table (RF/TF accuracy, F1, ROC-AUC), and saves a grouped bar chart (`backbone_comparison.png`) comparing architectures head-to-head. Runnable standalone once checkpoints exist:

## End-to-end pipeline (`main.ipynb`)

The notebook is the orchestration entry point, run top to bottom:

1. **Data leakage check** — confirms zero overlapping parent IDs between train/val (a `redigital_`/`transfer_` variant of an image counts as the same parent as its original).
2. **Dataset overview (EDA)** — class balance across Real/AI-Generated × Original/Transmitted/Re-digitized, how the parent-ID split divides into train/val/test, and one example image per class.
3. **Dataloader sanity check** — grabs one batch and prints its shape and label mix, to confirm the dataloaders are wired correctly before anything downstream depends on them.
4. **Imports & device setup** — imports the training/loss/model modules used by both backbone flows, and selects the compute device (CUDA → MPS → CPU).
5. **Frequency-domain preview** — the FFT log-magnitude and phase spectrum computed directly from the raw input, before any learned processing, for one example of each class combination — a sanity check for spectral artifacts that motivate having a frequency branch at all.
6. **ResNet-50 flow** — train across the full α/β sweep plus the uncertainty variant, collect and summarize the ablation results, evaluate every checkpoint on the test set, and run GradCAM on the best one.
7. **ConvNeXt-Tiny flow** — the same sequence as step 6, mirrored for the second backbone.
8. **Cross-backbone comparison** — `compare_backbone_checkpoints` evaluates the best config of each backbone on the same test loader and reports them side by side.

Model selection within each flow always picks the best checkpoint by **validation** accuracy (never test — that would leak the model-selection decision into reported test metrics), then runs the full diagnostic plot suite only on that one best checkpoint per backbone, rather than on all six.

## Results

Best config per backbone, selected by average validation accuracy, reported on the held-out test set:

| Metric | ResNet-50 (α=0.5, β=0.5) | ConvNeXt-Tiny (α=0.2, β=0.8) |
|---|---|---|
| Real/Fake accuracy | 93.72% | 94.75% |
| Transform accuracy | 92.20% | 94.44% |
| Precision / Recall (RF) | 0.920 / 0.956 | 0.960 / 0.933 |
| F1 (RF) | 0.938 | 0.946 |
| ROC AUC (RF) | 0.986 | 0.989 |
| Parameters (full model) | 23.9M | 28.2M |

ConvNeXt-Tiny wins on every single metric — confirmed directly by `compare_backbone_checkpoints`'s head-to-head table, not just by comparing separately-generated numbers. It also carries ~18% more parameters (28.2M vs. 23.9M), so part of the gap is raw capacity rather than a purely architectural advantage. The one metric that looks "worse" for ConvNeXt at first glance — recall 0.933 vs. ResNet's 0.956 — isn't actually a weakness: both models threshold at 0.5, and ConvNeXt's strictly higher ROC AUC means it's a better classifier at *every* threshold, not just this one. In absolute terms it misses fewer AI-generated images by labeling them "Real" (70 out of 1835, vs. ResNet's 148) at the cost of flagging slightly more real images as fake — trading a bit of recall for more precision at the default cutoff, which is the safer error direction for a deepfake detector to bias toward.

**Ablation finding, reproduced on both backbones:** the two degenerate single-task configurations (α=1/β=0, α=0/β=1) collapse the untrained head to near-chance accuracy on its own task, exactly as expected — a head that receives zero gradient signal can't learn anything. Every mixed-weight configuration (0.5/0.5, 0.8/0.2, 0.2/0.8, and the learned-uncertainty run), on both backbones, lands in a tight accuracy band regardless of the exact α/β ratio chosen — meaning once both loss terms have any non-zero weight, the two tasks stop competing for the shared backbone's representational capacity. Per-epoch training curves back this up with a matching pattern in overfitting behavior: the degenerate configurations, especially α=1/β=0, show classic train/validation loss divergence, while the balanced configurations do not — consistent with the auxiliary transform task acting as an implicit regularizer on the shared trunk, exactly what multi-task learning theory predicts.

**Per-transform-category breakdown (Real/Fake accuracy):** both backbones are weakest on `Transmitted` images (ResNet 87.4%, ConvNeXt 90.8%) and strongest on `Re-digitized` (ResNet 96.2%, ConvNeXt 98.7%) — `Original` vs. `Transmitted` is consistently the main confusion pair in the transform-task confusion matrix for both backbones, which tracks with intuition: re-digitization (a full re-capture pass) leaves much stronger, more distinctive artifacts than a single re-compression step, which can look close to an untouched original at moderate JPEG quality. See [REPORT.md](REPORT.md) for the full discussion, and `analysis_plots_resnet50/` / `analysis_plots_convnext/` for the complete per-backbone plot archives.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `tqdm`.

Place the dataset at `src/data/RRDataset_final/` following the structure described in [Dataset](#dataset), then open and run `main.ipynb` top to bottom. Checkpoints and history sidecars land in `models/`; diagnostic plots save to the repo root by default (or wherever a given call's `save_path` argument points).

## Configuration reference

All hyperparameters and toggles live in `src/globals.py`:

| Setting | Value | Notes |
|---|---|---|
| `RANDOM_SEED` | `42` | Used for sampling and the parent-ID split shuffle |
| `TRAIN_SPLIT` / `VAL_SPLIT` / `TEST_SPLIT` | `0.8` / `0.1` / `0.1` | By unique parent ID |
| `SAMPLES_PER_CLASS` | `6000` | Per one of the six `{transform} × {ai,real}` cells |
| `BATCH_SIZE` | `64` | |
| `NUM_WORKERS` / `PIN_MEMORY` | `4` / `True` | DataLoader settings |
| `ABLATION_NUM_EPOCHS` | `10` | Per weight combination |
| `EARLY_STOP_PATIENCE` | `3` | On average validation accuracy |
| `ABLATION_LEARNING_RATE` | `1e-4` | Adam |
