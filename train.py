from modules.model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from criterion import FocalLoss

# Hyper-parameters
num_classes = 2
batch_size = 2
epochs = 5
learning_rate = 1e-6

# Load data
transform = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_loader = torch.load(os.path.join('loaders', 'trainloader.pt'))

# Load model
model = Model(num_classes=num_classes)
model = model.to("cuda")

# Loss and optimizer
criterion = FocalLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
# Train
model.train()
gl_loss = 0

for epoch in range(epochs):

    for i, (images, labels) in enumerate(train_loader):
        images = images.to("cuda")
        labels = labels.to("cuda")

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        gl_loss += loss.item()
        # Backward and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print(f"Global Loss: {gl_loss:.4f}")

# Save model
torch.save(model.state_dict(), os.path.join('models', 'model.pt'))
