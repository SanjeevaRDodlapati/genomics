"""
ChromDragoNN model implementation for genomic sequence analysis.

ChromDragoNN is a deep learning model designed for predicting chromatin features
from DNA sequence, employing a multi-task convolutional architecture with residual
connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with convolutional layers and skip connection."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ChromDragoNNModel(nn.Module):
    """
    ChromDragoNN model for predicting chromatin features from DNA sequences.
    
    The model consists of:
    1. Initial convolutional layer
    2. Multiple residual blocks
    3. Global max pooling
    4. Fully connected layers for prediction
    """
    
    def __init__(
        self,
        sequence_length=1000,
        n_genomic_features=4,  # A, C, G, T
        n_outputs=919,         # Default DeepSEA output shape
        n_filters=300,
        n_residual_blocks=5,
        first_kernel_size=19,
        dropout_rate=0.2
    ):
        """
        Initialize the ChromDragoNN model.
        
        Args:
            sequence_length: Length of input DNA sequence
            n_genomic_features: Number of features per position (4 for DNA)
            n_outputs: Number of output predictions
            n_filters: Number of convolutional filters
            n_residual_blocks: Number of residual blocks
            first_kernel_size: Size of first convolutional filter
            dropout_rate: Dropout rate
        """
        super(ChromDragoNNModel, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv1d(n_genomic_features, n_filters, kernel_size=first_kernel_size)
        self.bn1 = nn.BatchNorm1d(n_filters)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(n_filters) for _ in range(n_residual_blocks)
        ])
        
        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        
        # Calculate the output size of the convolutional part
        conv_output_size = sequence_length - first_kernel_size + 1
        self.fc1 = nn.Linear(n_filters * conv_output_size, 1000)
        self.fc2 = nn.Linear(1000, n_outputs)
        
    def forward(self, x):
        """
        Forward pass for ChromDragoNN.
        
        Args:
            x: Input tensor of shape [batch_size, n_genomic_features, sequence_length]
            
        Returns:
            Model output of shape [batch_size, n_outputs]
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1)
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)