import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)  # 26x26x4
        self.pool1 = nn.MaxPool2d(2, 2)  # 13x13x4
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)  # 11x11x8
        self.pool2 = nn.MaxPool2d(2, 2)  # 5x5x8
        self.fc1 = nn.Linear(8 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 8 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced augmentation pipeline
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")
    
if __name__ == "__main__":
    train_model() 