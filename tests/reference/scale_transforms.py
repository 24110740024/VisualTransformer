# Reference syntax-tree fixtures for the reported v6 implementation.
# Parsed by tests; not executable entrypoints.

class Require224:

    def __call__(self, image):
        if image.size != (224, 224):
            raise ValueError('Expected original 224x224 QuPath export')
        return image

def build_transforms(train: bool, img_size: int=224):
    if img_size != 224:
        raise ValueError('Frozen image size is 224')
    stages = [Require224()]
    if train:
        stages.extend([transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)])
    stages.extend([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transforms.Compose(stages)
