import torch
import torch.nn as nn
from vit_pytorch import ViT
#from vit_pytorch.vit_for_small_dataset import ViT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch import SimpleViT
from vit_pytorch.cait import CaiT

class ViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViT(
            image_size=64,#64,
            patch_size=32,#32,
            num_classes=5,
            dim=64,#64,
            depth=6,#6,
            heads=8,#8,
            mlp_dim=512,#512,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        x = self.vit(x)
        return x

class CaiTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = CaiT(
            image_size = 64,
            patch_size = 32,
            num_classes = 5,
            dim = 64,
            depth = 6,             # depth of transformer for patch to patch attention only
            cls_depth = 2,          # depth of cross attention of CLS tokens to patch
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05    # randomly dropout 5% of the layers
        )

    def forward(self, x):
        x = self.vit(x)
        return x

class DeepViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = DeepViT(
            image_size=64,
            patch_size=32,
            num_classes=5,
            dim=64,
            depth=12,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        x = self.vit(x)
        return x


class SimpleViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = SimpleViT(
            image_size=32,
            patch_size=8,
            num_classes=5,
            dim=64,
            depth=6,
            heads=16,
            mlp_dim=2048,
        )

    def forward(self, x):
        x = self.vit(x)
        return x

class ViT_G14(nn.Module):
    def __init__(self, image_size, num_classes):
        super(ViT_G14, self).__init__()

        self.patch_size = 14
        self.vit = ViT(
            image_size=image_size,
            patch_size=self.patch_size,
            num_classes=num_classes,
            dim=1280,
            depth=24,
            heads=16,
            mlp_dim=5120,
            dropout=0.1,
            emb_dropout=0.1,
            #channels=3,
            #use_conv=False
        )

    def forward(self, x):
        x = x.view(-1, 3, 64, 64)
        x = self.vit(x)
        return x


