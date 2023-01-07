from modules.model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hyper-parameters
num_classes = 10
batch_size = 1
epochs = 1
learning_rate = 1e-3

# Load data
transform = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CIFAR10(root="data", train=True,
                        transform=transform, download=True)
test_dataset = CIFAR10(root="data", train=False,
                       transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = Model(num_classes=num_classes)
model = model.to("cuda")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Train
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to("cuda")
        labels = labels.to("cuda")

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
