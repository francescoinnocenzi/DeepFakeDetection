import os

import torch
from src.train.loss import MultiTaskLoss
from src.train.loops import MultiTaskModel, train_epoch, validate_epoch
from src.globals import ABLATION_LEARNING_RATE, ABLATION_NUM_EPOCHS, EARLY_STOP_PATIENCE

# The ablation engine iterates through different weightings ($\alpha$ and $\beta$) to analyze the trade-offs between the real/fake and transformation classification accuracies.
def run_ablation_study(train_loader, val_loader):
    """
    Run an ablation study by varying the weights of the multi-task loss components.
    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """
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
    
    for alpha, beta in weight_combinations:

        # Create the dynamic filename
        alpha_str = str(alpha).replace('.', '')
        beta_str = str(beta).replace('.', '')

        save_name = f"model_{alpha_str}_{beta_str}.pth"
        full_save_path = os.path.join(save_dir, save_name) 

        print(f"\n--- Running iteration with Alpha={alpha}, Beta={beta} ---")

        model = MultiTaskModel().to(device)
        criterion = MultiTaskLoss(alpha=alpha, beta=beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=ABLATION_LEARNING_RATE) #0.001 old
        
        # Track the best loss for this specific combination
        best_val_loss = float('inf')

        patience = EARLY_STOP_PATIENCE # How many epochs to wait before giving up
        patience_counter = 0

        # Train for a few epochs
        num_epochs = ABLATION_NUM_EPOCHS

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc_rf, val_acc_tf = validate_epoch(model, val_loader, criterion, device)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save the model if it has improved on the validation loss 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0 # Reset the counter because the model improved!
                print(f"Saving improved model to {full_save_path}...")
                torch.save(model.state_dict(), full_save_path)
            else:
                patience_counter += 1 # The model got worse, increase the counter
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                # Check if we have run out of patience
                if patience_counter >= patience:
                    print(f"Early stopping triggered! Moving to next weight combination.")
                    break # This breaks the epoch loop and goes to the next alpha/beta
        
        # The key of the dict is a tuple of (alpha, beta) and the value is another dict with all the relevant metrics
        results[(alpha, beta)] = {"train": train_loss, "val": val_loss, "val_acc_rf": val_acc_rf, "val_acc_tf": val_acc_tf}
        
    print("\n" + "="*40)
    print("Ablation Study Complete. Summary:")
    print("="*40)
    for key, val in results.items():
        print(f"Weights (alpha={key[0]}, beta={key[1]})")
        print(f"  -> Final Train Loss: {val['train']:.4f}")
        print(f"  -> Final Val Loss:   {val['val']:.4f}")
        print(f"  -> Final Val Acc (RF): {val['val_acc_rf']:.4f}")
        print(f"  -> Final Val Acc (TF): {val['val_acc_tf']:.4f}")
    print("="*40)