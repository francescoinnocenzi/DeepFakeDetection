import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc, precision_recall_fscore_support

from src.network.model import get_frequency_spectrum


def _sanitize_label(label):
    """Turns a human-readable plot label into a filesystem-safe filename fragment."""
    return label.replace(' ', '_').replace(',', '').replace('=', '').replace('.', '')


def evaluate_model(model, val_loader, device, quiet_mode=False, dataset_name="Validation", combo_label=None, label=None, skip_plots=None):
    """
    Evaluates the trained model on the dataset and generates performance metrics.
    Args:
        model: The trained MultiTaskModel
        val_loader: DataLoader for the evaluation set
        device: torch device (CPU or GPU)
        quiet_mode: If True, suppresses print statements and plots (for ablation study runs)
        dataset_name: Name of the dataset split being evaluated (e.g., 'Validation' or 'Test')
        combo_label: If given (e.g. "resnet50 alpha=0.5, beta=0.5"), only the confusion matrix and
            prediction-distribution plots are drawn, titled/saved per this label instead of the
            full 5-plot suite — for looping over many checkpoints without each one overwriting
            the last's PNGs. If None (default), draws the full 5-plot suite instead.
        label: Optional label shown in the title (and filenames) of the full 5-plot suite,
            e.g. "resnet50 alpha=0.8, beta=0.2" — so it's clear which checkpoint produced them.
            Ignored when combo_label is set (that path always uses combo_label for the title).
        skip_plots: Optional set of plot names to omit from the label/default 5-plot suite —
            e.g. {'confusion_matrix', 'prediction_distribution'} to avoid redrawing plots an
            earlier combo_label call already produced for this same checkpoint. Valid names:
            'confusion_matrix', 'category_breakdown', 'roc_pr', 'prediction_distribution',
            'transform_metrics'. Ignored when combo_label is set.
    Returns:
        acc_rf: Overall accuracy for the Real/Fake task
        acc_trans: Overall accuracy for the Transformation task
    """
    skip_plots = skip_plots or set()
    model.eval()
    
    # Storage for all predictions and ground truths
    all_preds_rf, all_probs_rf, all_labels_rf = [], [], []
    all_preds_trans, all_labels_trans = [], []

    if not quiet_mode:
        print(f"Running evaluation on {dataset_name} set...")
    with torch.no_grad():
        for images, labels_rf, labels_trans in val_loader:
            images = images.to(device)

            logits_rf, logits_trans = model(images)

            probs_rf = torch.sigmoid(logits_rf).squeeze(-1)
            preds_rf = probs_rf.round()
            preds_trans = torch.argmax(logits_trans, dim=1)

            all_probs_rf.extend(probs_rf.cpu().numpy())
            all_preds_rf.extend(preds_rf.cpu().numpy())
            all_labels_rf.extend(labels_rf.cpu().numpy())
            all_preds_trans.extend(preds_trans.cpu().numpy())
            all_labels_trans.extend(labels_trans.cpu().numpy())

    # Convert to numpy arrays for easier math indexing
    all_probs_rf = np.array(all_probs_rf)
    all_preds_rf = np.array(all_preds_rf)
    all_labels_rf = np.array(all_labels_rf)
    all_preds_trans = np.array(all_preds_trans)
    all_labels_trans = np.array(all_labels_trans)

    # --- 1. Overall Accuracy ---
    acc_rf = accuracy_score(all_labels_rf, all_preds_rf)
    acc_trans = accuracy_score(all_labels_trans, all_preds_trans)

    precision_rf = precision_score(all_labels_rf, all_preds_rf)
    recall_rf = recall_score(all_labels_rf, all_preds_rf)
    f1_rf = f1_score(all_labels_rf, all_preds_rf)
    auc_rf = roc_auc_score(all_labels_rf, all_probs_rf)

    if not quiet_mode:
        print(f"\n--- Final Results ---")
        print(f"Overall Real/Fake Accuracy:   {acc_rf * 100:.2f}%")
        print(f"Overall Transform Accuracy:   {acc_trans * 100:.2f}%")

        print(f"\n--- Advanced Metrics (Real/Fake Task) ---")
        print(f"Precision: {precision_rf:.4f}")
        print(f"Recall:    {recall_rf:.4f}")
        print(f"F1 Score:  {f1_rf:.4f}")
        print(f"ROC AUC:   {auc_rf:.4f}")

        if combo_label is not None:
            safe_label = _sanitize_label(combo_label)
            plot_confusion_matrices(all_labels_rf, all_preds_rf, all_labels_trans, all_preds_trans,
                                     title_suffix=combo_label, save_path=f"confusion_matrices_{safe_label}.png")
            plot_prediction_distribution(all_labels_rf, all_probs_rf,
                                          title_suffix=combo_label, save_path=f"prediction_distribution_{safe_label}.png")
        elif label is not None:
            safe_label = _sanitize_label(label)
            if 'confusion_matrix' not in skip_plots:
                plot_confusion_matrices(all_labels_rf, all_preds_rf, all_labels_trans, all_preds_trans,
                                         title_suffix=label, save_path=f"confusion_matrices_{safe_label}.png")
            if 'category_breakdown' not in skip_plots:
                plot_category_breakdown(all_labels_rf, all_preds_rf, all_labels_trans,
                                         title_suffix=label, save_path=f"accuracy_breakdown_{safe_label}.png")
            if 'roc_pr' not in skip_plots:
                plot_roc_and_pr_curves(all_labels_rf, all_probs_rf,
                                        title_suffix=label, save_path=f"roc_pr_curves_{safe_label}.png")
            if 'prediction_distribution' not in skip_plots:
                plot_prediction_distribution(all_labels_rf, all_probs_rf,
                                              title_suffix=label, save_path=f"prediction_distribution_{safe_label}.png")
            if 'transform_metrics' not in skip_plots:
                plot_transform_class_metrics(all_labels_trans, all_preds_trans,
                                              title_suffix=label, save_path=f"transform_class_metrics_{safe_label}.png")
        else:
            plot_confusion_matrices(all_labels_rf, all_preds_rf, all_labels_trans, all_preds_trans)
            plot_category_breakdown(all_labels_rf, all_preds_rf, all_labels_trans)
            plot_roc_and_pr_curves(all_labels_rf, all_probs_rf)
            plot_prediction_distribution(all_labels_rf, all_probs_rf)
            plot_transform_class_metrics(all_labels_trans, all_preds_trans)

    return acc_rf, acc_trans

def plot_confusion_matrices(true_rf, pred_rf, true_trans, pred_trans, title_suffix="", save_path="confusion_matrices.png"):
    """
    Plots confusion matrices for both tasks side by side.
    Args:
        true_rf: Ground truth labels for Real/Fake task
        pred_rf: Predicted labels for Real/Fake task
        true_trans: Ground truth labels for Transformation task
        pred_trans: Predicted labels for Transformation task
        title_suffix: Optional text appended to both subplot titles (e.g. a checkpoint label)
        save_path: Path to save the figure (default: "confusion_matrices.png")
    Returns:
        None (saves and shows the plot)
    """
    suffix = f" — {title_suffix}" if title_suffix else ""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Real/Fake Confusion Matrix
    cm_rf = confusion_matrix(true_rf, pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['AI-Generated', 'Real'],
                yticklabels=['AI-Generated', 'Real'])
    axes[0].set_title(f'Task 1: Real vs Fake Detection{suffix}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # 2. Transformation Confusion Matrix
    cm_trans = confusion_matrix(true_trans, pred_trans)
    sns.heatmap(cm_trans, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Original', 'Transmitted', 'Re-digitized'],
                yticklabels=['Original', 'Transmitted', 'Re-digitized'])
    axes[1].set_title(f'Task 2: Transformation Classification{suffix}')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")

def plot_category_breakdown(true_rf, pred_rf, true_trans, title_suffix="", save_path="accuracy_breakdown.png"):
    """
    Plots a grouped bar chart showing Real/Fake accuracy for all three transformation categories separately.
    Args:
        true_rf: Ground truth labels for Real/Fake task
        pred_rf: Predicted labels for Real/Fake task
        true_trans: Ground truth labels for Transformation task (used to filter by category)
        title_suffix: Optional text appended to the title (e.g. a checkpoint label)
        save_path: Path to save the figure (default: "accuracy_breakdown.png")
    Returns:
        None (saves and shows the plot)
    """
    suffix = f" — {title_suffix}" if title_suffix else ""
    categories = ['Original', 'Transmitted', 'Re-digitized']
    ai_accs = []
    real_accs = []

    for trans_idx in [0, 1, 2]:
        # Filter data for this specific transformation
        mask_trans = (true_trans == trans_idx)
        
        true_rf_sub = true_rf[mask_trans]
        pred_rf_sub = pred_rf[mask_trans]
        
        # Calculate accuracy for AI images (Label 0) in this transformation
        mask_ai = (true_rf_sub == 0)
        acc_ai = accuracy_score(true_rf_sub[mask_ai], pred_rf_sub[mask_ai]) if np.any(mask_ai) else 0
        ai_accs.append(acc_ai * 100)
        
        # Calculate accuracy for Real images (Label 1) in this transformation
        mask_real = (true_rf_sub == 1)
        acc_real = accuracy_score(true_rf_sub[mask_real], pred_rf_sub[mask_real]) if np.any(mask_real) else 0
        real_accs.append(acc_real * 100)

    # Plotting the Grouped Bar Chart
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, ai_accs, width, label='AI Images Accuracy', color='salmon')
    rects2 = ax.bar(x + width/2, real_accs, width, label='Real Images Accuracy', color='skyblue')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Objective 3: Real/Fake Detection Robustness by Transformation{suffix}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 110) # Set to 110 to leave room for the legend
    ax.legend(loc='lower right')

    # Add the text labels above the bars
    ax.bar_label(rects1, padding=3, fmt='%.1f%%')
    ax.bar_label(rects2, padding=3, fmt='%.1f%%')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")

def plot_roc_and_pr_curves(true_rf, probs_rf, title_suffix="", save_path="roc_pr_curves.png"):
    """
    Plots the ROC curve and Precision-Recall curve for binary Real/Fake detection.
    Args:
        true_rf: Ground truth binary labels (0: AI, 1: Real)
        probs_rf: Predicted probabilities for Real class
        title_suffix: Optional text appended to both subplot titles (e.g. a checkpoint label)
        save_path: Path to save the figure (default: "roc_pr_curves.png")
    """
    suffix = f" — {title_suffix}" if title_suffix else ""
    fpr, tpr, _ = roc_curve(true_rf, probs_rf)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(true_rf, probs_rf)
    pr_auc = auc(recall, precision)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title(f'ROC Curve - Real vs Fake Detection{suffix}', fontsize=12, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Precision-Recall Curve
    axes[1].plot(recall, precision, color='#2ca02c', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title(f'Precision-Recall Curve - Real vs Fake Detection{suffix}', fontsize=12, fontweight='bold')
    axes[1].legend(loc="lower left", fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")

def plot_transform_class_metrics(true_trans, pred_trans, title_suffix="", save_path="transform_class_metrics.png"):
    """
    Plots per-class Precision/Recall/F1 for the transformation classification task
    (Original / Transmitted / Re-digitized). This task is 3-way, not binary, so it
    can't use the same ROC/PR curves as the Real/Fake task — per-class bars are the
    multiclass analog.
    Args:
        true_trans: Ground truth labels for the transformation task (0/1/2)
        pred_trans: Predicted labels for the transformation task (0/1/2)
        title_suffix: Optional text appended to the title (e.g. a checkpoint label)
        save_path: Path to save the figure (default: "transform_class_metrics.png")
    Returns:
        None (saves and shows the plot)
    """
    suffix = f" — {title_suffix}" if title_suffix else ""
    categories = ['Original', 'Transmitted', 'Re-digitized']
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_trans, pred_trans, labels=[0, 1, 2], zero_division=0
    )

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width, precision * 100, width, label='Precision', color='#1f77b4')
    r2 = ax.bar(x, recall * 100, width, label='Recall', color='#ff7f0e')
    r3 = ax.bar(x + width, f1 * 100, width, label='F1 Score', color='#2ca02c')

    ax.set_ylabel('Score (%)')
    ax.set_title(f'Transformation Classification — Per-Class Metrics{suffix}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right')

    ax.bar_label(r1, padding=3, fmt='%.1f')
    ax.bar_label(r2, padding=3, fmt='%.1f')
    ax.bar_label(r3, padding=3, fmt='%.1f')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")

def plot_prediction_distribution(true_rf, probs_rf, title_suffix="", save_path="prediction_distribution.png"):
    """
    Plots the probability density distribution for AI-Generated vs Real images.
    Demonstrates model confidence and calibration separation.
    Args:
        title_suffix: Optional text appended to the title (e.g. a checkpoint label)
        save_path: Path to save the figure (default: "prediction_distribution.png")
    """
    suffix = f" — {title_suffix}" if title_suffix else ""
    fig, ax = plt.subplots(figsize=(10, 5))

    ai_probs = probs_rf[true_rf == 0]
    real_probs = probs_rf[true_rf == 1]

    sns.kdeplot(ai_probs, fill=True, color='salmon', label='AI-Generated (True 0)', ax=ax, alpha=0.5, bw_adjust=0.5)
    sns.kdeplot(real_probs, fill=True, color='skyblue', label='Real Images (True 1)', ax=ax, alpha=0.5, bw_adjust=0.5)

    ax.axvline(0.5, color='gray', linestyle='--', lw=1.5, label='Decision Threshold (0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Predicted Probability (Sigmoid Output)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Prediction Confidence Separation (Sigmoid Probability Distribution){suffix}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper center', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")

def plot_ablation_study(results_dict, save_path="ablation_study.png"):
    """
    Plots the trade-off graph for the different alpha/beta weightings.
    Args:
        results_dict: A dictionary where keys are "alpha_beta" strings and values are tuples of (Real/Fake Accuracy, Transformation Accuracy).
        save_path: Path to save the figure (default: "ablation_study.png")
    Returns:
        None (saves and shows the plot)
    """
    labels = list(results_dict.keys())
    rf_accs = [res[0] for res in results_dict.values()]
    trans_accs = [res[1] for res in results_dict.values()]

    plt.figure(figsize=(8, 6))
    plt.scatter(trans_accs, rf_accs, color='purple', s=100)
    plt.plot(trans_accs, rf_accs, linestyle='--', color='gray', alpha=0.5)

    # Add labels to the dots
    for i, label in enumerate(labels):
        if str(label).lower() == 'uncertainty':
            label_text = "Uncertainty Loss"
        elif '_' in str(label):
            label_text = f"α={label.split('_')[0]}, β={label.split('_')[1]}"
        else:
            label_text = str(label)
        plt.annotate(label_text, 
                     (trans_accs[i], rf_accs[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), ha='center')

    plt.title('Objective 4: Ablation Study Trade-off')
    plt.xlabel('Transformation Accuracy (%)')
    plt.ylabel('Real/Fake Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")


def plot_ablation_bar_chart(results_dict, backbone_label, save_path=None):
    """
    Grouped bar chart comparing Real/Fake vs Transform accuracy across all
    alpha/beta weightings and the learned-uncertainty run, for one backbone.
    Args:
        results_dict: Same shape as plot_ablation_study's — keys are "alpha_beta"
            strings or "uncertainty", values are (Real/Fake Accuracy, Transform Accuracy) tuples.
        backbone_label: Backbone name shown in the title (e.g. "resnet50").
        save_path: Path to save the figure (default: "ablation_accuracy_{backbone_label}.png")
    Returns:
        None (saves and shows the plot)
    """
    if save_path is None:
        save_path = f"ablation_accuracy_{backbone_label}.png"

    labels = list(results_dict.keys())
    rf_accs = [res[0] for res in results_dict.values()]
    trans_accs = [res[1] for res in results_dict.values()]

    label_texts = []
    for label in labels:
        if str(label).lower() == 'uncertainty':
            label_texts.append("Learned Uncertainty")
        elif '_' in str(label):
            alpha, beta = str(label).split('_')
            label_texts.append(f"α={alpha}, β={beta}")
        else:
            label_texts.append(str(label))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, rf_accs, width, label='Real/Fake Accuracy (%)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, trans_accs, width, label='Transform Accuracy (%)', color='#2ca02c')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'[{backbone_label}] Ablation Study: Task Accuracy by Loss Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(label_texts)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')

    ax.bar_label(rects1, padding=3, fmt='%.1f%%')
    ax.bar_label(rects2, padding=3, fmt='%.1f%%')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")


def plot_training_curves(history, config_label, save_path=None):
    """
    Plots training/validation loss and validation accuracy over epochs for one
    trained configuration (an alpha/beta combo or the uncertainty run).
    Args:
        history: dict with keys 'train_loss', 'val_loss', 'val_acc_rf', 'val_acc_tf',
            each a list of per-epoch values — as saved by run_ablation_study's
            "<checkpoint>_history.pt" sidecar file.
        config_label: Label shown in the title (e.g. "resnet50 alpha=0.5, beta=0.5").
        save_path: Path to save the figure (default: "training_curves_{config_label}.png")
    Returns:
        None (saves and shows the plot)
    """
    if save_path is None:
        save_path = f"training_curves_{_sanitize_label(config_label)}.png"

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label='Train Loss', color='#1f77b4', marker='o')
    axes[0].plot(epochs, history["val_loss"], label='Val Loss', color='#d62728', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss Curve — {config_label}')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    val_acc_rf_pct = [a * 100 for a in history["val_acc_rf"]]
    val_acc_tf_pct = [a * 100 for a in history["val_acc_tf"]]
    axes[1].plot(epochs, val_acc_rf_pct, label='Val Acc (Real/Fake)', color='#2ca02c', marker='o')
    axes[1].plot(epochs, val_acc_tf_pct, label='Val Acc (Transform)', color='#9467bd', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Validation Accuracy — {config_label}')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")


def plot_training_dynamics_comparison(histories, backbone_label, save_path=None):
    """
    Compares training/validation dynamics across all configs for one backbone in
    a single 4-panel chart: Training Loss, Validation Loss, Validation Real/Fake
    Accuracy, and Validation Transform Accuracy over epochs — one line per config,
    so convergence behavior can be compared directly instead of across separate plots.
    Args:
        histories: dict mapping config_label -> history dict (as saved by
            run_ablation_study's "<checkpoint>_history.pt" sidecar), e.g.
            {"alpha=1.0, beta=0.0": {...}, "Learned Uncertainty": {...}}
        backbone_label: Backbone name shown in the title (e.g. "resnet50").
        save_path: Path to save the figure (default: "training_dynamics_{backbone_label}.png")
    Returns:
        None (saves and shows the plot)
    """
    if save_path is None:
        save_path = f"training_dynamics_{backbone_label}.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for label, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        is_uncertainty = "uncertainty" in label.lower()
        style = '-' if is_uncertainty else '--'
        lw = 2.5 if is_uncertainty else 1.5

        axes[0, 0].plot(epochs, history["train_loss"], style, label=label, linewidth=lw)
        axes[0, 1].plot(epochs, history["val_loss"], style, label=label, linewidth=lw)
        axes[1, 0].plot(epochs, [a * 100 for a in history["val_acc_rf"]], style, label=label, linewidth=lw)
        axes[1, 1].plot(epochs, [a * 100 for a in history["val_acc_tf"]], style, label=label, linewidth=lw)

    panel_specs = [
        (axes[0, 0], 'Training Loss over Epochs', 'Loss'),
        (axes[0, 1], 'Validation Loss over Epochs', 'Loss'),
        (axes[1, 0], 'Validation Real/Fake Accuracy over Epochs', 'Accuracy (%)'),
        (axes[1, 1], 'Validation Transform Accuracy over Epochs', 'Accuracy (%)'),
    ]
    for ax, subtitle, ylabel in panel_specs:
        ax.set_title(subtitle)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(f'[{backbone_label}] Ablation Study: Training & Validation Dynamics',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def _find_example_per_combo(data_loader):
    """
    Scans a DataLoader for one example image of each Real/AI-Generated x
    Original/Transmitted/Re-digitized combination.
    Args:
        data_loader: DataLoader yielding (images, labels_rf, labels_trans) batches
    Returns:
        dict mapping (rf_label, trans_label) -> (1, 3, H, W) image tensor, for
        whichever of the 6 combinations were found (fewer if the loader is small).
    """
    wanted = {(rf, trans) for rf in (0, 1) for trans in (0, 1, 2)}
    found = {}

    for images, labels_rf, labels_trans in data_loader:
        for i in range(images.shape[0]):
            key = (int(labels_rf[i].item()), int(labels_trans[i].item()))
            if key in wanted and key not in found:
                found[key] = images[i:i + 1].clone()
        if len(found) == len(wanted):
            break

    return found


def plot_sample_grid(data_loader, save_path="dataset_sample_grid.png"):
    """
    Shows one example image for each Real/AI-Generated x Original/Transmitted/
    Re-digitized combination found in the given loader — a quick visual check of
    what each class actually looks like, before any training happens.
    Args:
        data_loader: DataLoader yielding (images, labels_rf, labels_trans) batches
        save_path: Path to save the figure (default: "dataset_sample_grid.png")
    Returns:
        None (saves and shows the plot)
    """
    rf_names = {0: 'AI-Generated', 1: 'Real'}
    trans_names = {0: 'Original', 1: 'Transmitted', 2: 'Re-digitized'}
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    found = _find_example_per_combo(data_loader)
    if not found:
        print("No samples found in the given loader — nothing to plot.")
        return

    combos = sorted(found.keys(), key=lambda k: (k[1], k[0]))  # group by transform, then real/fake
    n = len(combos)

    ncols = 3 if n >= 3 else n
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, key in zip(axes, combos):
        rf, trans = key
        img_tensor = found[key]
        img_np = img_tensor[0].permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean).clip(0, 1)
        ax.imshow(img_np)
        ax.set_title(f"{rf_names[rf]} · {trans_names[trans]}", fontsize=11)
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle("Dataset Sample Grid — One Example per Class", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_class_distribution(train_loader, val_loader, test_loader, save_path="class_distribution.png"):
    """
    Plots dataset composition before training: Real/AI-Generated counts broken
    down by transform category (across the full dataset), and how many samples
    ended up in each split. Reads labels directly from the underlying dataset's
    stored (path, label_rf, label_trans) tuples — the same access pattern the
    leakage-check cell uses — so no images are loaded and this is fast regardless
    of dataset size.
    Args:
        train_loader, val_loader, test_loader: DataLoaders as returned by get_dataloaders
        save_path: Path to save the figure (default: "class_distribution.png")
    Returns:
        None (saves and shows the plot)
    """
    def _label_counts(loader):
        subset = loader.dataset
        base_dataset = getattr(subset, "dataset", subset)
        indices = getattr(subset, "indices", range(len(subset)))
        labels_rf = np.array([base_dataset.samples[idx][1] for idx in indices])
        labels_trans = np.array([base_dataset.samples[idx][2] for idx in indices])
        return labels_rf, labels_trans

    splits = {"Train": train_loader, "Val": val_loader, "Test": test_loader}
    trans_names = ['Original', 'Transmitted', 'Re-digitized']

    all_rf_parts, all_trans_parts = [], []
    split_sizes = {}
    for name, loader in splits.items():
        rf, trans = _label_counts(loader)
        all_rf_parts.append(rf)
        all_trans_parts.append(trans)
        split_sizes[name] = len(rf)
    all_rf = np.concatenate(all_rf_parts)
    all_trans = np.concatenate(all_trans_parts)

    # Counts per (rf, trans) combination across the full dataset (all splits combined)
    combo_counts = np.zeros((2, 3), dtype=int)  # rows=rf (0=AI,1=Real), cols=trans
    for rf_val in (0, 1):
        for trans_val in (0, 1, 2):
            combo_counts[rf_val, trans_val] = int(np.sum((all_rf == rf_val) & (all_trans == trans_val)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(trans_names))
    width = 0.35
    rects1 = axes[0].bar(x - width / 2, combo_counts[0], width, label='AI-Generated', color='salmon')
    rects2 = axes[0].bar(x + width / 2, combo_counts[1], width, label='Real', color='skyblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(trans_names)
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Class Balance by Transform Category')
    axes[0].legend()
    axes[0].bar_label(rects1, padding=3)
    axes[0].bar_label(rects2, padding=3)

    split_names = list(split_sizes.keys())
    split_vals = list(split_sizes.values())
    bars = axes[1].bar(split_names, split_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Number of Images')
    axes[1].set_title('Train / Val / Test Split Sizes')
    axes[1].bar_label(bars, padding=3)

    fig.suptitle('Dataset Overview', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_frequency_spectrum(data_loader, save_path="frequency_spectrum.png"):
    """
    Visualizes the FFT log-magnitude and phase spectrum — exactly what the frequency
    branch receives, via the same get_frequency_spectrum() the model's forward pass
    uses, with no learned processing applied — for one example of each
    Real/AI-Generated x Original/Transmitted/Re-digitized combination found in the
    given loader. A preliminary, dataset-level view of spectral differences between
    real and AI-generated content, and across transform pipelines.
    Args:
        data_loader: DataLoader yielding (images, labels_rf, labels_trans) batches.
            Images are used exactly as the model receives them (normalized), since
            get_frequency_spectrum operates on that same tensor internally.
        save_path: Path to save the figure (default: "frequency_spectrum.png")
    Returns:
        None (saves and shows the plot)
    """
    rf_names = {0: 'AI-Generated', 1: 'Real'}
    trans_names = {0: 'Original', 1: 'Transmitted', 2: 'Re-digitized'}
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    found = _find_example_per_combo(data_loader)

    if not found:
        print("No samples found in the given loader — nothing to plot.")
        return

    combos = sorted(found.keys(), key=lambda k: (k[1], k[0]))  # group rows by transform, then real/fake
    n = len(combos)

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    if n == 1:
        axes = axes[None, :]

    for row, key in enumerate(combos):
        rf, trans = key
        img_tensor = found[key]

        with torch.no_grad():
            spectrum = get_frequency_spectrum(img_tensor).squeeze(0).numpy()  # (6, H, W)
        magnitude = spectrum[0:3].mean(axis=0)
        phase = spectrum[3:6].mean(axis=0)

        img_np = img_tensor[0].permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean).clip(0, 1)

        label = f"{rf_names[rf]} · {trans_names[trans]}"
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"Original\n{label}", fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(magnitude, cmap='viridis')
        axes[row, 1].set_title("Log-Magnitude Spectrum", fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(phase, cmap='twilight')
        axes[row, 2].set_title("Phase Spectrum", fontsize=10)
        axes[row, 2].axis('off')

    fig.suptitle("Frequency-Domain View — Input to the Frequency Branch",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
    if n < 6:
        wanted = {(rf, trans) for rf in (0, 1) for trans in (0, 1, 2)}
        missing = wanted - found.keys()
        print(f"Note: could not find examples for {len(missing)} combination(s) in the scanned batches: {missing}")


def compute_gradcam(model, image_tensor, head='real_fake', class_idx=None, device='cpu', branch='spatial'):
    """
    Computes a GradCAM heatmap for a single image.

    Args:
        model       : MultiTaskModel instance (in eval mode)
        image_tensor: (1, 3, 224, 224) normalised tensor
        head        : 'real_fake' or 'transform'
        class_idx   : class to explain (None → uses the predicted class)
        device      : torch device
        branch      : 'spatial' (last conv block of spatial_backbone) or
                      'frequency' (last conv block of frequency_backbone, before
                      the AdaptiveAvgPool) — both heads read fused features from
                      both branches, so either branch can drive either head.

    Returns:
        cam: (224, 224) numpy array with values in [0, 1]
    """
    model.eval()

    gradients = []
    activations = []

    def _fwd_hook(module, input, output):
        activations.append(output)

    def _bwd_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    if branch == 'spatial':
        if hasattr(model.spatial_backbone, 'layer4'):
            target_layer = model.spatial_backbone.layer4[-1]
        elif hasattr(model.spatial_backbone, 'features'):
            target_layer = model.spatial_backbone.features[-1][-1]
        else:
            raise ValueError("Cannot determine target_layer for GradCAM.")
    elif branch == 'frequency':
        # features[-2] is an inplace ReLU. A backward hook wraps its target layer's
        # output in a view that autograd forbids feeding into an inplace op — and
        # that holds whether we hook the ReLU itself or the BatchNorm2d right before
        # it (whose output flows straight into that same inplace ReLU). Hook the
        # Conv2d two steps back instead: its output feeds a non-inplace BatchNorm2d.
        target_layer = model.frequency_backbone.features[-4]
    else:
        raise ValueError(f"Unsupported branch: {branch}. Expected 'spatial' or 'frequency'.")
    fwd_handle = target_layer.register_forward_hook(_fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

    # Forward pass — gradients must flow, so no torch.no_grad()
    img = image_tensor.to(device)
    logits_rf, logits_trans = model(img)

    if head == 'real_fake':
        score = logits_rf[0, 0]
    else:
        if class_idx is None:
            class_idx = logits_trans.argmax(dim=1).item()
        score = logits_trans[0, class_idx]

    model.zero_grad()
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # Weights = global average of gradients over spatial dims
    grad = gradients[0]                            # (1, C, H, W)
    act  = activations[0]                          # (1, C, H, W)
    weights = grad.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

    # Weighted combination + ReLU
    cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, h, w)
    cam = F.relu(cam)

    # Upsample to 224×224
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

    # Normalise to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    return cam


def plot_gradcam(model, val_loader, device, num_samples=4, save_path="gradcam_visualization.png", label=None):
    """
    Visualises GradCAM overlays for both heads on a few validation samples,
    covering both the spatial and frequency branches (the classification heads
    read fused features from both, so either branch can drive either head).
    Saves the figure to the specified path (default: 'gradcam_visualization.png').

    Layout per row:
        [Original] | [RF · Spatial] | [RF · Frequency] | [Transform · Spatial] | [Transform · Frequency]

    The spatial-branch CAMs are overlaid on the original RGB image (they live in the same
    pixel coordinate space). The frequency-branch CAMs are overlaid on the FFT log-magnitude
    spectrum instead — they operate on frequency_backbone's activations, whose spatial
    dimensions correspond to frequency, not image, coordinates (center = low frequency/DC,
    edges = high frequency), so overlaying them on the photo would misrepresent them as a
    spatial attention map over the face/scene, which they are not.

    Args:
        label: Optional text shown in the title (e.g. a checkpoint label), so it's clear
            which model/checkpoint produced this visualization.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    rf_names    = {0: 'AI-Generated', 1: 'Real'}
    trans_names = {0: 'Original', 1: 'Transmitted', 2: 'Re-digitized'}

    # Grab one batch and keep only num_samples
    images, labels_rf, labels_trans = next(iter(val_loader))
    images      = images[:num_samples]
    labels_rf   = labels_rf[:num_samples]
    labels_trans = labels_trans[:num_samples]

    suffix = f" — {label}" if label else ""
    fig, axes = plt.subplots(num_samples, 5, figsize=(23, num_samples * 4))
    fig.suptitle(f"GradCAM — What the model looks at, per head and per branch{suffix}",
                 fontsize=15, fontweight='bold', y=1.01)

    for i in range(num_samples):
        img_tensor = images[i:i+1].clone()  # (1,3,224,224)

        # Denormalise for display
        img_np = images[i].permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean).clip(0, 1)

        # FFT log-magnitude spectrum — the frequency branch's actual input — for the
        # frequency-branch CAMs to be overlaid on, instead of the spatial RGB photo.
        with torch.no_grad():
            spectrum = get_frequency_spectrum(img_tensor).squeeze(0).numpy()  # (6, H, W)
        magnitude = spectrum[0:3].mean(axis=0)  # average over RGB channels -> (H, W), in [0, 1]
        magnitude_rgb = np.stack([magnitude, magnitude, magnitude], axis=-1)

        # Get predictions (no_grad is fine here — only for display labels)
        with torch.no_grad():
            logits_rf, logits_trans = model(img_tensor.to(device))
            pred_rf    = int(torch.sigmoid(logits_rf).round().item())
            pred_trans = int(logits_trans.argmax(dim=1).item())

        # Compute GradCAM for each head, on both the spatial and frequency branches
        cam_rf_spatial    = compute_gradcam(model, img_tensor.clone(), head='real_fake', device=device, branch='spatial')
        cam_rf_freq       = compute_gradcam(model, img_tensor.clone(), head='real_fake', device=device, branch='frequency')
        cam_trans_spatial = compute_gradcam(model, img_tensor.clone(), head='transform',
                                            class_idx=pred_trans, device=device, branch='spatial')
        cam_trans_freq    = compute_gradcam(model, img_tensor.clone(), head='transform',
                                            class_idx=pred_trans, device=device, branch='frequency')

        true_rf    = rf_names[labels_rf[i].item()]
        true_trans = trans_names[labels_trans[i].item()]

        def _overlay(img, cam):
            heatmap = cm.jet(cam)[:, :, :3]
            return (0.55 * img + 0.45 * heatmap).clip(0, 1)

        # Column 0 — Original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Original\nTrue: {true_rf} · {true_trans}", fontsize=10)
        axes[i, 0].axis('off')

        # Column 1 — Real/Fake GradCAM, spatial branch
        axes[i, 1].imshow(_overlay(img_np, cam_rf_spatial))
        axes[i, 1].set_title(f"Real/Fake · Spatial\nPred: {rf_names[pred_rf]}", fontsize=10)
        axes[i, 1].axis('off')

        # Column 2 — Real/Fake GradCAM, frequency branch (overlaid on the FFT magnitude
        # spectrum, not the photo — center = low frequency, edges = high frequency/noise)
        axes[i, 2].imshow(_overlay(magnitude_rgb, cam_rf_freq))
        axes[i, 2].set_title(f"Real/Fake · Frequency (on FFT Magnitude)\nPred: {rf_names[pred_rf]}", fontsize=10)
        axes[i, 2].axis('off')

        # Column 3 — Transform GradCAM, spatial branch
        axes[i, 3].imshow(_overlay(img_np, cam_trans_spatial))
        axes[i, 3].set_title(f"Transform · Spatial\nPred: {trans_names[pred_trans]}", fontsize=10)
        axes[i, 3].axis('off')

        # Column 4 — Transform GradCAM, frequency branch (same FFT-magnitude overlay as column 2)
        axes[i, 4].imshow(_overlay(magnitude_rgb, cam_trans_freq))
        axes[i, 4].set_title(f"Transform · Frequency (on FFT Magnitude)\nPred: {trans_names[pred_trans]}", fontsize=10)
        axes[i, 4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")