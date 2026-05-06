import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        
        # 1. Shared Backbone
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 2. Dual Heads
        self.head_real_fake = nn.Linear(in_features, 1)
        self.head_transform = nn.Linear(in_features, 3)

    def forward(self, x):
        features = self.backbone(x)
        logits_real_fake = self.head_real_fake(features)
        logits_transform = self.head_transform(features)
        return logits_real_fake, logits_transform

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels_rf, labels_tf in dataloader:
        images = images.to(device)
        labels_rf = labels_rf.to(device)
        labels_tf = labels_tf.to(device)
        
        optimizer.zero_grad()
        
        logits_rf, logits_tf = model(images)
        loss, loss_rf, loss_tf = criterion(logits_rf, logits_tf, labels_rf, labels_tf)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)