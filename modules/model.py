import torch
import torch.nn as nn
import torch.nn.functional as F
from .LocalFeatureExtraction import LocalFeatureExtraction
from .VisionTransformer import VisionTransformer
from einops.layers.torch import Reduce
import torchsummary


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'), 
            nn.Linear(emb_size, n_classes)
        )

class Model(nn.Sequential):
    def __init__(self, num_classes, depth=1):
        super(Model, self).__init__(
            LocalFeatureExtraction(),
            VisionTransformer(in_channels=32, patch_size=14, num_heads=128 * 7,
                              expansion=4, drop_p=0.1, feed_forward_drop_p=0.1, depth=depth),
            ClassificationHead(emb_size=32 * 14 * 14, n_classes=num_classes)
        )


if __name__ == "__main__":
    x = torch.randn(5, 3, 224, 224).to("cuda")
    model = Model(num_classes=2, depth=2)
    model.to("cuda")

    torchsummary.summary(model, (3, 224, 224))


