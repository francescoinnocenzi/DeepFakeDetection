import torchvision.transforms as T
from PIL import Image
import io
import random

class RandomJPEGCompression:
    """Custom transform to simulate WhatsApp/Email transmission artifacts."""
    def __init__(self, min_quality=30, max_quality=95, p=0.7):
        self.min_q = min_quality
        self.max_q = max_quality
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.min_q, self.max_q)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            return Image.open(buf)
        return img

# 1. BASE PIPELINES: Separate for Train and Validation
# Degradation (JPEG, blur, color jitter) is applied to ALL training images, not only
# 'transfer' ones. If it were label-conditional, the model could shortcut the transform
# classification task by detecting artifact presence, and the frequency branch would
# learn JPEG DCT patterns instead of deepfake signatures.
train_transforms = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)], p=0.4),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1),
    RandomJPEGCompression(min_quality=30, max_quality=95, p=0.5),
])

val_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224)
])

# 2. FINALIZATION PIPELINE: Applied to ALL images at the very end
to_tensor_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet standards
])