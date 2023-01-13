import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os


def getdataset(path, batch_size):
    # path: path to the dataset
    # batch_size: batch size
    # return: train_loader, test_loader
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = ImageFolder(os.path.join(path, 'train'),
                            transform=train_transform)
    test_set = ImageFolder(os.path.join(path, 'test'),
                           transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import os
    trainloader, testloader = getdataset(
        path=os.getenv('DATAPATH'), batch_size=32)
