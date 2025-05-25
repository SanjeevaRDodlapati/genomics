"""DeepSEA model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSEA(nn.Module):
    """DeepSEA model architecture.
    
    Based on the original DeepSEA paper: Zhou & Troyanskaya (2015)
    """

    def __init__(
        self,
        sequence_length: int = 1000,
        n_targets: int = 919,
        conv_kernel_sizes: list = None,
        conv_channels: list = None,
        pool_kernel_sizes: list = None,
        dropout_rate: float = 0.2,
    ):
        """Initialize the DeepSEA model.
        
        Args:
            sequence_length: Length of input DNA sequence
            n_targets: Number of output targets
            conv_kernel_sizes: List of kernel sizes for convolutional layers
            conv_channels: List of channel numbers for convolutional layers
            pool_kernel_sizes: List of kernel sizes for pooling layers
            dropout_rate: Dropout rate
        """
        super(DeepSEA, self).__init__()
        
        # Default architecture parameters
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [8, 8, 8]
        if conv_channels is None:
            conv_channels = [320, 480, 960]
        if pool_kernel_sizes is None:
            pool_kernel_sizes = [4, 4, 4]
        
        # Input layer (one-hot encoded DNA: 4 channels)
        self.input_channels = 4
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First convolutional layer
        self.conv_layers.append(
            nn.Conv1d(
                self.input_channels,
                conv_channels[0],
                kernel_size=conv_kernel_sizes[0]
            )
        )
        
        # Additional convolutional layers
        for i in range(1, len(conv_channels)):
            self.conv_layers.append(
                nn.Conv1d(
                    conv_channels[i-1],
                    conv_channels[i],
                    kernel_size=conv_kernel_sizes[i]
                )
            )
        
        # Calculate size after convolutions and pooling
        size = sequence_length
        for i in range(len(conv_kernel_sizes)):
            # Convolution reduces size by kernel_size - 1
            size = size - (conv_kernel_sizes[i] - 1)
            # Pooling reduces size by factor of pool_kernel_sizes
            size = size // pool_kernel_sizes[i]
        
        # Fully connected layers
        self.fc1 = nn.Linear(size * conv_channels[-1], 925)
        self.fc2 = nn.Linear(925, n_targets)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 4, sequence_length)
                representing one-hot encoded DNA sequences
            
        Returns:
            Model predictions of shape (batch_size, n_targets)
        """
        # Apply convolutional layers with max pooling
        for i, conv in enumerate(self.conv_layers):
            x = F.max_pool1d(
                F.relu(conv(x)),
                kernel_size=4
            )
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
