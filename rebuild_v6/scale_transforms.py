"""Preserve complete QuPath 224 image; no crop/zoom/aspect distortion."""
from torchvision import transforms

class Require224:
    def __call__(self,image):
        if image.size!=(224,224):raise ValueError('Expected original 224x224 QuPath export')
        return image

def build_transforms(train:bool,img_size:int=224):
    if img_size!=224:raise ValueError('Frozen image size is 224')
    stages=[Require224()]
    if train:
        stages.extend([transforms.RandomHorizontalFlip(),transforms.ColorJitter(.1,.1,.1,.05)])
    stages.extend([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return transforms.Compose(stages)
