import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
        # Task 1: Real vs Fake (binary classification)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Task 2: Transformation type (3 classes)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits_real_fake, logits_transform, labels_real_fake, labels_transform):
        # Ensure labels_real_fake matches the (batch_size, 1) shape and float type for BCE loss
        labels_real_fake = labels_real_fake.view(-1, 1).float()
        
        # Computes the combined total loss
        loss_rf = self.bce_loss(logits_real_fake, labels_real_fake)
        loss_tf = self.ce_loss(logits_transform, labels_transform)
        
        total_loss = (self.alpha * loss_rf) + (self.beta * loss_tf)
        
        return total_loss, loss_rf, loss_tf