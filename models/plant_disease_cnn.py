"""
Convolutional Neural Network model for plant disease classification.

This module defines the PlantDiseaseCNN architecture used for
classifying plant diseases from images.
"""

import torch
import torch.nn as nn


class PlantDiseaseCNN(nn.Module):
    """
    CNN model for plant disease classification.

    Architecture:
        - 3 Convolutional layers (32 -> 64 -> 128 filters)
        - Max pooling after each conv layer
        - 2 Fully connected layers (256 -> num_classes)
        - ReLU activation and Dropout for regularization

    Args:
        num_classes: Number of disease classes to predict
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initialize the PlantDiseaseCNN model.

        Args:
            num_classes: Number of output classes (disease categories)
        """
        super(PlantDiseaseCNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Apply conv layers with ReLU and pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
