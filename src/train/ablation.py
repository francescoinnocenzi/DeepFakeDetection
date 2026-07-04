import os

import torch
import src.globals
from src.train.loss import get_loss
from src.train.loops import MultiTaskModel, train_epoch, validate_epoch
from src.globals import ABLATION_LEARNING_RATE, ABLATION_NUM_EPOCHS, EARLY_STOP_PATIENCE

# The ablation engine iterates through different weightings ($\alpha$ and $\beta$) to analyze the trade-offs between the real/fake and transformation classification accuracies.
def run_ablation_study(train_loader, val_loader, backbone_type=None, loss_type=None):
    """
    Run an ablation study by varying the weights of the multi-task loss components.
    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        backbone_type: Optional backbone identifier ('resnet50', 'convnext_tiny', etc.).
        loss_type: Optional 'fixed' or 'uncertainty'. Defaults to src.globals.LOSS_TYPE if not given.
    """
    if backbone_type is None:
        from src.globals import BACKBONE_TYPE
        backbone_type = BACKBONE_TYPE

    if loss_type is None:
        loss_type = src.globals.LOSS_TYPE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example set of weight combinations
    weight_combinations = [
        (1.0, 0.0), # Unimodal Real/Fake
        (0.0, 1.0), # Unimodal Transform
        (0.5, 0.5), # Balanced
        (0.8, 0.2), # RF focused
        (0.2, 0.8), # Transform focused
    ]

    results = {}

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # Under uncertainty loss alpha/beta are ignored — run only once to avoid wasted compute
    if loss_type == 'uncertainty':
        weight_combinations = [(0.5, 0.5)]

    for alpha, beta in weight_combinations:

        if loss_type == 'uncertainty':
            save_name = f"model_{backbone_type}_uncertainty.pth" if backbone_type != 'resnet50' else "model_uncertainty.pth"
            print(f"\n--- Running [{backbone_type}] with learned uncertainty weighting ---")
        else:
            alpha_str = str(alpha).replace('.', '')
            beta_str = str(beta).replace('.', '')
            save_name = f"model_{backbone_type}_{alpha_str}_{beta_str}.pth" if backbone_type != 'resnet50' else f"model_{alpha_str}_{beta_str}.pth"
            print(f"\n--- Running [{backbone_type}] iteration with Alpha={alpha}, Beta={beta} ---")

        full_save_path = os.path.join(save_dir, save_name)

        model = MultiTaskModel(backbone_type=backbone_type).to(device)
        criterion = get_loss(alpha=alpha, beta=beta, loss_type=loss_type).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=ABLATION_LEARNING_RATE
        )

        # Track the best score (average validation accuracy across both tasks)
        best_val_score = -1.0
        best_metrics = None # Metrics of the epoch actually saved to full_save_path

        # Full per-epoch history (unlike best_metrics, keeps every epoch — not just
        # the best one) so later cells can plot loss/accuracy curves over training.
        history = {"train_loss": [], "val_loss": [], "val_acc_rf": [], "val_acc_tf": []}

        patience = EARLY_STOP_PATIENCE # How many epochs to wait before giving up
        patience_counter = 0

        # Train for a few epochs
        num_epochs = ABLATION_NUM_EPOCHS

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc_rf, val_acc_tf = validate_epoch(model, val_loader, criterion, device)
            val_score = (val_acc_rf + val_acc_tf) / 2.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc_rf"].append(val_acc_rf)
            history["val_acc_tf"].append(val_acc_tf)

            sigma_info = ""
            if loss_type == 'uncertainty':
                s1 = torch.exp(criterion.log_sigma1).item()
                s2 = torch.exp(criterion.log_sigma2).item()
                # 5 decimals: if these stay flat at 1.00000 the sigmas aren't training
                sigma_info = f", σ_rf={s1:.5f}, σ_trans={s2:.5f}"
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc (Avg): {val_score*100:.2f}%{sigma_info}")

            # Save the model if it has improved on average validation accuracy
            if val_score > best_val_score:
                best_val_score = val_score
                best_metrics = {"train": train_loss, "val": val_loss, "val_acc_rf": val_acc_rf, "val_acc_tf": val_acc_tf}
                patience_counter = 0 # Reset the counter because the model improved!
                print(f"Saving improved model (Val Acc: {val_score*100:.2f}%) to {full_save_path}...")
                torch.save(model.state_dict(), full_save_path)
                if loss_type == 'uncertainty':
                    # Sigmas live on the loss module, not the model — save them alongside the
                    # checkpoint so later evaluation cells can report them (model.state_dict()
                    # alone doesn't carry them).
                    sigma_path = os.path.splitext(full_save_path)[0] + "_sigmas.pt"
                    torch.save({"sigma_rf": s1, "sigma_trans": s2}, sigma_path)
            else:
                patience_counter += 1 # The model got worse, increase the counter
                print(f"No improvement in Val Acc. Patience: {patience_counter}/{patience}")

                # Check if we have run out of patience
                if patience_counter >= patience:
                    print(f"Early stopping triggered! Moving to next weight combination.")
                    break # This breaks the epoch loop and goes to the next alpha/beta

        # Save the full epoch-by-epoch history alongside the checkpoint (model.state_dict()
        # only carries weights, not training history) so later cells can plot loss/accuracy curves.
        history_path = os.path.splitext(full_save_path)[0] + "_history.pt"
        torch.save(history, history_path)

        # The key of the dict is a tuple of (alpha, beta) or 'uncertainty' and the value is another dict with all the relevant metrics
        # Report the BEST epoch's metrics (the one actually saved to full_save_path), not the last epoch's
        res_key = 'uncertainty' if loss_type == 'uncertainty' else (alpha, beta)
        results[res_key] = best_metrics

    print("\n" + "="*40)
    print("Ablation Study Complete. Summary:")
    print("="*40)
    for key, val in results.items():
        if key == 'uncertainty':
            print("Learned Uncertainty Weighting")
        else:
            print(f"Weights (alpha={key[0]}, beta={key[1]})")
        print(f"  -> Best Train Loss: {val['train']:.4f}")
        print(f"  -> Best Val Loss:   {val['val']:.4f}")
        print(f"  -> Best Val Acc (RF): {val['val_acc_rf']:.4f}")
        print(f"  -> Best Val Acc (TF): {val['val_acc_tf']:.4f}")
    print("="*40)
    return results