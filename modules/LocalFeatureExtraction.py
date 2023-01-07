import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalFeatureExtraction(nn.Sequential):
    def __init__(self):
        super(LocalFeatureExtraction, self).__init__(
            nn.Conv2d(3, 3, 3, padding=1, stride=1),
            nn.Conv2d(3, 5, 3, padding=1, stride=1),
            nn.Conv2d(5, 9, 3, padding=1, stride=1),
            nn.BatchNorm2d(9),
            nn.PReLU(),
            nn.Conv2d(9, 16, 5, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, 5, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((224, 224)),
        )

if __name__ == '__main__':
    x = torch.rand(5, 3, 224, 224)
    model = LocalFeatureExtraction()
    y = model(x)
    print(y.shape)