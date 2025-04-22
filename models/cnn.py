import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """A simple CNN for image classification (e.g., MNIST)."""
    def __init__(self, input_channels=1, num_classes=10):
        """
        Args:
            input_channels (int): Number of channels in the input image (1 for grayscale, 3 for RGB).
            num_classes (int): Number of output classes.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces dimension by half (e.g., 28x28 -> 14x14)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces dimension again (e.g., 14x14 -> 7x7)

        # Calculate the flattened size after convolutions and pooling
        # For MNIST (1x28x28 input):
        # After pool1: 32 x 14 x 14
        # After pool2: 64 x 7 x 7
        self.flattened_size = 64 * 7 * 7 # Adjust if input size or layers change

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu3 = nn.ReLU()
        # self.dropout = nn.Dropout(0.5) # Optional dropout
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Forward pass"""
        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        # x = self.dropout(x) # Apply dropout if enabled
        x = self.fc2(x)

        return x