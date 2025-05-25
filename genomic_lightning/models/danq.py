"""
DanQ model implementation for genomic sequence analysis.

DanQ combines CNN and BiLSTM layers for improved feature extraction
from genomic sequences. The model architecture is based on:
"DanQ: a hybrid convolutional and recurrent deep neural network for
quantifying the function of DNA sequences."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DanQModel(nn.Module):
    """
    DanQ hybrid CNN-RNN model for genomic sequence analysis.
    
    The model consists of:
    1. Convolutional layer with max pooling
    2. Bidirectional LSTM
    3. Dense layers
    """
    
    def __init__(
        self,
        sequence_length=1000,
        n_genomic_features=4,  # A, C, G, T
        n_outputs=919,         # Default DeepSEA output shape
        conv_kernel_size=26,
        conv_filters=320,
        pool_size=13,
        pool_stride=13,
        lstm_hidden=320,
        dropout_rate=0.2
    ):
        """
        Initialize the DanQ model.
        
        Args:
            sequence_length: Length of input DNA sequence
            n_genomic_features: Number of features per position (4 for DNA)
            n_outputs: Number of output predictions
            conv_kernel_size: Size of convolutional filters
            conv_filters: Number of convolutional filters
            pool_size: Max pooling size
            pool_stride: Max pooling stride
            lstm_hidden: Number of LSTM hidden units
            dropout_rate: Dropout rate
        """
        super(DanQModel, self).__init__()
        
        # Calculate the size after convolution and pooling
        self.pool_output_size = (sequence_length - conv_kernel_size + 1) // pool_stride
        
        # Convolutional layer
        self.conv = nn.Conv1d(
            n_genomic_features,
            conv_filters,
            kernel_size=conv_kernel_size
        )
        
        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        
        # Dense layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(2 * lstm_hidden * self.pool_output_size, 925)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(925, n_outputs)
        
    def forward(self, x):
        """
        Forward pass for DanQ.
        
        Args:
            x: Input tensor of shape [batch_size, n_genomic_features, sequence_length]
            
        Returns:
            Model output of shape [batch_size, n_outputs]
        """
        # Convolution layer (batch_size, conv_filters, reduced_length)
        x = F.relu(self.conv(x))
        
        # Max pooling
        x = F.max_pool1d(x, kernel_size=self.pool_size, stride=self.pool_stride)
        
        # Transpose to feed into LSTM
        x = x.transpose(1, 2)  # (batch_size, reduced_length, conv_filters)
        
        # LSTM layer
        x, _ = self.bilstm(x)  # (batch_size, reduced_length, 2*lstm_hidden)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Dense layers
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        
        return torch.sigmoid(x)
