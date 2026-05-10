import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model(model, val_loader, device, quiet_mode=False):
    """
    Runs the validation data through the model and generates all required plots.
    """
    model.eval()
    
    # Storage for all predictions and ground truths
    all_preds_rf, all_labels_rf = [], []
    all_preds_trans, all_labels_trans = [], []
    
    print("Running evaluation on validation set...")
    with torch.no_grad():
        for images, labels_rf, labels_trans in val_loader:
            images = images.to(device)
            
            # Get raw logits from model
            logits_rf, logits_trans = model(images)
            
            # Convert logits to predictions
            # Real/Fake is BCE, so use Sigmoid + Round
            preds_rf = torch.sigmoid(logits_rf).squeeze().round() 
            # Transform is CE, so use Argmax
            preds_trans = torch.argmax(logits_trans, dim=1) 
            
            # Move to CPU and save to lists
            all_preds_rf.extend(preds_rf.cpu().numpy())
            all_labels_rf.extend(labels_rf.cpu().numpy())
            all_preds_trans.extend(preds_trans.cpu().numpy())
            all_labels_trans.extend(labels_trans.cpu().numpy())

    # Convert to numpy arrays for easier math indexing
    all_preds_rf = np.array(all_preds_rf)
    all_labels_rf = np.array(all_labels_rf)
    all_preds_trans = np.array(all_preds_trans)
    all_labels_trans = np.array(all_labels_trans)

    # --- 1. Overall Accuracy ---
    acc_rf = accuracy_score(all_labels_rf, all_preds_rf)
    acc_trans = accuracy_score(all_labels_trans, all_preds_trans)

    if not quiet_mode:
        print(f"\n--- Final Results ---")
        print(f"Overall Real/Fake Accuracy:   {acc_rf * 100:.2f}%")
        print(f"Overall Transform Accuracy:   {acc_trans * 100:.2f}%")

        plot_confusion_matrices(all_labels_rf, all_preds_rf, all_labels_trans, all_preds_trans)
        plot_category_breakdown(all_labels_rf, all_preds_rf, all_labels_trans)

    return acc_rf, acc_trans

def plot_confusion_matrices(true_rf, pred_rf, true_trans, pred_trans):
    """Generates two side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Real/Fake Confusion Matrix
    cm_rf = confusion_matrix(true_rf, pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['AI-Generated', 'Real'], 
                yticklabels=['AI-Generated', 'Real'])
    axes[0].set_title('Task 1: Real vs Fake Detection')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # 2. Transformation Confusion Matrix
    cm_trans = confusion_matrix(true_trans, pred_trans)
    sns.heatmap(cm_trans, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Original', 'Transmitted', 'Re-digitized'], 
                yticklabels=['Original', 'Transmitted', 'Re-digitized'])
    axes[1].set_title('Task 2: Transformation Classification')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=300)
    plt.show()
    print("Saved: confusion_matrices.png")

def plot_category_breakdown(true_rf, pred_rf, true_trans):
    """
    Objective 3: Breaks down Real/Fake accuracy across the 3 transformation 
    categories, separated by AI vs Real images.
    """
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
    ax.set_title('Objective 3: Real/Fake Detection Robustness by Transformation')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 110) # Set to 110 to leave room for the legend
    ax.legend(loc='lower right')

    # Add the text labels above the bars
    ax.bar_label(rects1, padding=3, fmt='%.1f%%')
    ax.bar_label(rects2, padding=3, fmt='%.1f%%')

    plt.tight_layout()
    plt.savefig("accuracy_breakdown.png", dpi=300)
    plt.show()
    print("Saved: accuracy_breakdown.png")

def plot_ablation_study(results_dict):
    """
    Objective 4: Plots the trade-off graph for the different alpha/beta weightings.
    Expects a dictionary like: 
    {'1.0_0.0': (90.5, 33.3), '0.5_0.5': (88.0, 85.0), ...}
    where the tuple is (Real/Fake Acc, Trans Acc).
    """
    labels = list(results_dict.keys())
    rf_accs = [res[0] for res in results_dict.values()]
    trans_accs = [res[1] for res in results_dict.values()]

    plt.figure(figsize=(8, 6))
    plt.scatter(trans_accs, rf_accs, color='purple', s=100)
    plt.plot(trans_accs, rf_accs, linestyle='--', color='gray', alpha=0.5)

    # Add labels to the dots
    for i, label in enumerate(labels):
        plt.annotate(f"α={label.split('_')[0]}, β={label.split('_')[1]}", 
                     (trans_accs[i], rf_accs[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), ha='center')

    plt.title('Objective 4: Ablation Study Trade-off')
    plt.xlabel('Transformation Accuracy (%)')
    plt.ylabel('Real/Fake Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("ablation_study.png", dpi=300)
    plt.show()
    print("Saved: ablation_study.png")