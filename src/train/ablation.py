import torch
from src.train.loss import MultiTaskLoss
from src.train.loops import MultiTaskModel, train_epoch, validate_epoch

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
        (0.5, 0.5),
        (0.8, 0.2),
        (0.2, 0.8),
    ]
    
    results = {}
    
    for alpha, beta in weight_combinations:
        print(f"\n--- Running iteration with Alpha={alpha}, Beta={beta} ---")
        
        model = MultiTaskModel().to(device)
        criterion = MultiTaskLoss(alpha=alpha, beta=beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few epochs
        num_epochs = 5
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate_epoch(model, val_loader, criterion, device)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
        results[(alpha, beta)] = {"train": train_loss, "val": val_loss}
        
    print("\n" + "="*40)
    print("Ablation Study Complete. Summary:")
    print("="*40)
    for key, val in results.items():
        print(f"Weights (alpha={key[0]}, beta={key[1]})")
        print(f"  -> Final Train Loss: {val['train']:.4f}")
        print(f"  -> Final Val Loss:   {val['val']:.4f}")
    print("="*40)