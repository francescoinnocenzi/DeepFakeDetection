import torch
import torch.nn as nn
import src.globals


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits_real_fake, logits_transform, labels_real_fake, labels_transform):
        labels_real_fake = labels_real_fake.view(-1, 1).float()
        loss_rf = self.bce_loss(logits_real_fake, labels_real_fake)
        loss_tf = self.ce_loss(logits_transform, labels_transform)
        total_loss = (self.alpha * loss_rf) + (self.beta * loss_tf)
        return total_loss, loss_rf, loss_tf


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with learnable per-task uncertainty weights (Kendall et al. 2018).
    Total Loss = (1/2*sigma1^2)*L_rf + log(sigma1) + (1/2*sigma2^2)*L_trans + log(sigma2)
    sigma1 and sigma2 are learned alongside model weights — no manual alpha/beta needed.
    """
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()
        # Store log(sigma) for numerical stability and to enforce sigma > 0
        self.log_sigma1 = nn.Parameter(torch.zeros(1))  # task 1: real/fake
        self.log_sigma2 = nn.Parameter(torch.zeros(1))  # task 2: transform
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits_real_fake, logits_transform, labels_real_fake, labels_transform):
        labels_real_fake = labels_real_fake.view(-1, 1).float()
        loss_rf = self.bce_loss(logits_real_fake, labels_real_fake)
        loss_tf = self.ce_loss(logits_transform, labels_transform)

        # 1/(2*sigma^2) = exp(-2*log(sigma)) / 2
        w1 = torch.exp(-2 * self.log_sigma1)
        w2 = torch.exp(-2 * self.log_sigma2)

        total_loss = (0.5 * w1 * loss_rf + self.log_sigma1 +
                      0.5 * w2 * loss_tf + self.log_sigma2)

        return total_loss, loss_rf, loss_tf


def get_loss(alpha=0.5, beta=0.5, loss_type=None):
    """Returns the loss instance selected by loss_type, or by LOSS_TYPE in globals.py if not given."""
    if loss_type is None:
        loss_type = src.globals.LOSS_TYPE
    if loss_type == 'uncertainty':
        return UncertaintyWeightedLoss()
    return MultiTaskLoss(alpha=alpha, beta=beta)
