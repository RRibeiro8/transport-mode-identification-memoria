import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        #print(self.resnet50)
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.resnet50(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=num_classes)
        
    def forward(self, x):
        x = self.resnet18(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        self.resnet101 = models.resnet101(pretrained=False)
        self.resnet101.fc = nn.Linear(in_features=2048, out_features=num_classes)
        
    def forward(self, x):
        x = self.resnet101(x)
        return x

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()
        self.resnet152 = models.resnet152(pretrained=True)
        self.resnet152.fc = nn.Linear(in_features=2048, out_features=num_classes)
        
    def forward(self, x):
        x = self.resnet152(x)
        return x