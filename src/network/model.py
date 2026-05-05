import torch.nn as nn

class DualHeadModel(nn.Module):
    def __init__(self, backbone):
        super(DualHeadModel, self).__init__()
        self.backbone = backbone
        # Define heads here
