import torchvision.models as models

def get_backbone():
    return models.resnet50(pretrained=True)
