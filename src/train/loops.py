import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        
        # 1. Shared Backbone
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT) # Load pre-trained ResNet-50 (creates a ResNet object)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove the original classification head
        
        # 2. Dual Heads
        self.head_real_fake = nn.Linear(in_features, 1) # 2048 -> 1 for real/fake classification
        self.head_transform = nn.Linear(in_features, 3) # 2048 -> 3 for transformation classification

    def forward(self, x):
        features = self.backbone(x) # each image is transformed into a 2048-dimensional feature vector (runs the forward pass through the ResNet backbone)
        logits_real_fake = self.head_real_fake(features)  # maps the features to a single logit for real/fake classification 
        logits_transform = self.head_transform(features) # maps the features to three logits for transformation classification 
        return logits_real_fake, logits_transform

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
    
    with torch.no_grad():
        for images, labels_rf, labels_tf in dataloader:
            images = images.to(device)
            labels_rf = labels_rf.to(device)
            labels_tf = labels_tf.to(device)
            
            logits_rf, logits_tf = model(images)
            loss, loss_rf, loss_tf = criterion(logits_rf, logits_tf, labels_rf, labels_tf)
            
            running_loss += loss.item()
            
    return running_loss / len(dataloader)