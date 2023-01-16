import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torchvision.datasets import ImageFolder
from modules.model import Model
from criterion import FocalLoss
import numpy as np


if __name__ == '__main__':
    model = Model(num_classes=2, depth=1)
    model.load_state_dict(torch.load('models/model.pt'))
    model = model.to('cuda')
    test_set = torch.load('loaders/trainloader.pt')
    criterion = FocalLoss()

    model.eval()
    count = 0
    total = 0
    negatives = 0
    a = 0
    for i, (images, labels) in enumerate(test_set):
        images = images.to('cuda')
        labels = labels.to('cuda')
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        labels = labels.cpu()
        predicted = predicted.cpu()
        labels = labels.numpy()
        predicted = predicted.numpy()

        a += np.sum(labels == 1)
        negatives += np.sum(predicted == 1)
    #     total += labels.size(0)
    #     count += (predicted == labels).sum().item()
    #     print(f"Loss: {loss.item():.4f}, Accuracy: {count/total:.4f}")

    # print(f"Accuracy: {count/total:.4f}")
    print(f"Negatives: {negatives}, {a}")