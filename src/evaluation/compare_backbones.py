import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.train.loops import MultiTaskModel
from src.data.dataloader import get_dataloaders

def compare_backbone_checkpoints(backbone_models, test_loader=None, data_dir="src/data/RRDataset_final"):
    """
    Evaluates and compares different trained backbones on the test set.
    
    Args:
        backbone_models (dict): Mapping of display name to (backbone_type, checkpoint_path).
                                Example:
                                {
                                    "ResNet-50": ("resnet50", "models/model_05_05.pth"),
                                    "ConvNeXt-Tiny": ("convnext_tiny", "models/model_convnext_tiny_05_05.pth")
                                }
        test_loader: Optional DataLoader. If None, it will be created from data_dir.
        data_dir: Path to dataset root if test_loader is None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running backbone comparison on device: {device}\n")
    
    if test_loader is None:
        print("Loading test dataset...")
        _, _, test_loader = get_dataloaders(data_dir=data_dir)
        
    metrics_summary = {}

    for display_name, (b_type, ckpt_path) in backbone_models.items():
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {display_name}: Checkpoint not found at {ckpt_path}")
            continue
            
        print(f"Evaluating {display_name} ({b_type}) from {ckpt_path}...")
        model = MultiTaskModel(backbone_type=b_type).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        all_probs_rf, all_preds_rf, all_labels_rf = [], [], []
        all_preds_trans, all_labels_trans = [], []
        
        with torch.no_grad():
            for images, labels_rf, labels_trans in test_loader:
                images = images.to(device)
                logits_rf, logits_trans = model(images)
                
                probs_rf = torch.sigmoid(logits_rf).squeeze()
                preds_rf = probs_rf.round()
                preds_trans = torch.argmax(logits_trans, dim=1)
                
                all_probs_rf.extend(probs_rf.cpu().numpy())
                all_preds_rf.extend(preds_rf.cpu().numpy())
                all_labels_rf.extend(labels_rf.cpu().numpy())
                all_preds_trans.extend(preds_trans.cpu().numpy())
                all_labels_trans.extend(labels_trans.cpu().numpy())
                
        all_probs_rf = np.array(all_probs_rf)
        all_preds_rf = np.array(all_preds_rf)
        all_labels_rf = np.array(all_labels_rf)
        all_preds_trans = np.array(all_preds_trans)
        all_labels_trans = np.array(all_labels_trans)
        
        acc_rf = accuracy_score(all_labels_rf, all_preds_rf) * 100
        acc_tf = accuracy_score(all_labels_trans, all_preds_trans) * 100
        precision = precision_score(all_labels_rf, all_preds_rf)
        recall = recall_score(all_labels_rf, all_preds_rf)
        f1 = f1_score(all_labels_rf, all_preds_rf)
        auc = roc_auc_score(all_labels_rf, all_probs_rf)
        
        metrics_summary[display_name] = {
            "acc_rf": acc_rf,
            "acc_tf": acc_tf,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    if not metrics_summary:
        print("\nNo models were successfully evaluated.")
        return metrics_summary

    # Print Summary Table
    print("\n" + "="*80)
    print(f"{'Backbone':<16} | {'RF Acc (%)':<12} | {'TF Acc (%)':<12} | {'F1 Score':<10} | {'ROC AUC':<10}")
    print("="*80)
    for name, m in metrics_summary.items():
        print(f"{name:<16} | {m['acc_rf']:<12.2f} | {m['acc_tf']:<12.2f} | {m['f1']:<10.4f} | {m['auc']:<10.4f}")
    print("="*80)

    # Plot Comparison Chart
    names = list(metrics_summary.keys())
    rf_accs = [m['acc_rf'] for m in metrics_summary.values()]
    tf_accs = [m['acc_tf'] for m in metrics_summary.values()]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, rf_accs, width, label='Real/Fake Accuracy (%)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, tf_accs, width, label='Transform Accuracy (%)', color='#2ca02c')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Backbone Architecture Comparison on Test Set', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    ax.bar_label(rects1, padding=3, fmt='%.2f%%')
    ax.bar_label(rects2, padding=3, fmt='%.2f%%')
    
    plt.tight_layout()
    plot_path = "backbone_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"\nSaved comparison plot to {plot_path}")
    
    return metrics_summary

if __name__ == "__main__":
    # Example usage:
    models_to_compare = {
        "ResNet-50": ("resnet50", "models/model_05_05.pth"),
        "ConvNeXt-Tiny": ("convnext_tiny", "models/model_convnext_tiny_05_05.pth"),
        "ConvNeXt-Base": ("convnext_base", "models/model_convnext_base_05_05.pth"),
    }
    compare_backbone_checkpoints(models_to_compare)
