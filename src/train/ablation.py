import torch
from torch.utils.data import DataLoader, TensorDataset
from src.train.loss import MultiTaskLoss
from src.train.loops import MultiTaskModel, train_epoch



#The ablation engine iterates through different weightings ($\alpha$ and $\beta$) to analyze the trade-offs between the real/fake and transformation classification accuracies.
def run_ablation_study():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example set of weight combinations
    weight_combinations = [
        (0.5, 0.5),
        (0.8, 0.2),
        (0.2, 0.8),
    ]
    
    # Mock data for testing integration (B, 3, 224, 224)
    dummy_images = torch.randn(10, 3, 224, 224)
    dummy_labels_rf = torch.randint(0, 2, (10, 1)).float()
    dummy_labels_tf = torch.randint(0, 3, (10,))
    
    dataset = TensorDataset(dummy_images, dummy_labels_rf, dummy_labels_tf)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    results = {}
    
    for alpha, beta in weight_combinations:
        print(f"Running iteration with Alpha={alpha}, Beta={beta}")
        
        model = MultiTaskModel().to(device)
        criterion = MultiTaskLoss(alpha=alpha, beta=beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for 2 mock epochs
        for epoch in range(2):
            avg_loss = train_epoch(model, dataloader, criterion, optimizer, device)
            
        results[(alpha, beta)] = avg_loss
        
    print("\nAblation Study Complete. Summary of losses:")
    for key, val in results.items():
        print(f"Weights (alpha={key[0]}, beta={key[1]}) -> Average Loss: {val:.4f}")

if __name__ == "__main__":
    run_ablation_study()