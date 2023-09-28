import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_v2_s, efficientnet_v2_l

class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.model = efficientnet_b0(pretrained=False)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetV2, self).__init__()
        self.model = efficientnet_v2_s(pretrained=False)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x