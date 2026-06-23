import torch
import torch.nn as nn
from src.network.model import DualBranchMTLModel

class MultiTaskModel(DualBranchMTLModel):
    def __init__(self):
        super(MultiTaskModel, self).__init__()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Args:
        model: The multi-task model to be trained.
        dataloader: DataLoader for the training data.
        criterion: The multi-task loss function.
        optimizer: The optimizer for updating model parameters.
        device: The device (CPU or GPU) to run the training on.
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    
    for images, labels_rf, labels_tf in dataloader:
        # Move tensors to the appropriate device (e.g., GPU if available)
        images = images.to(device)
        labels_rf = labels_rf.to(device)
        labels_tf = labels_tf.to(device)
        
        # Reset gradients for each batch
        optimizer.zero_grad()
        
        logits_rf, logits_tf = model(images) # Forward pass through the model to get predictions for both tasks
        loss, loss_rf, loss_tf = criterion(logits_rf, logits_tf, labels_rf, labels_tf)
        
        loss.backward() # Backpropagation to compute gradients based on the combined loss
        optimizer.step() # Update model parameters using the gradients
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    # Set up counters for accuracy
    correct_rf = 0
    correct_tf = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels_rf, labels_tf in dataloader:
            images = images.to(device)
            labels_rf = labels_rf.to(device)
            labels_tf = labels_tf.to(device)
            
            logits_rf, logits_tf = model(images)
            loss, loss_rf, loss_tf = criterion(logits_rf, logits_tf, labels_rf, labels_tf)
            
            running_loss += loss.item()

            # Calculate Real/Fake Accuracy (Binary)
            preds_rf = torch.sigmoid(logits_rf).round().squeeze()
            
            # Compare predictions to true labels and count the matches
            correct_rf += (preds_rf == labels_rf.squeeze()).sum().item()
            
            # Calculate Transform Accuracy (Multi-class) ---
            preds_tf = torch.argmax(logits_tf, dim=1)
            correct_tf += (preds_tf == labels_tf).sum().item()
            
            # Update total samples count
            total_samples += labels_rf.size(0)
    
    # Calculate Final Averages
    avg_loss = running_loss / len(dataloader)
    acc_rf = correct_rf / total_samples
    acc_tf = correct_tf / total_samples

    return running_loss / len(dataloader), acc_rf, acc_tf