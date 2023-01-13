import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torchvision.datasets import ImageFolder
from modules.model import Model
from criterion import FocalLoss


if __name__ == '__main__':
    model = Model(num_classes=2, depth=1)
    model = model.to("cuda")
    model.eval()
    x = torch.randn(5, 3, 224, 224).to("cuda")
    y = model(x)
    y_real = torch.tensor([0, 1, 0, 1, 0]).to("cuda")
    print(y.shape)
    loss = FocalLoss()(y, y_real)
    print(loss)
