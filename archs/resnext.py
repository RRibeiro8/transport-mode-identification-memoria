import torch.nn as nn
import torchvision.models as models

class ResNeXt:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def resnext50_32x4d(self):
        model = models.resnext50_32x4d(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        return model
    
    def resnext101_32x8d(self):
        model = models.resnext101_32x8d(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        return model