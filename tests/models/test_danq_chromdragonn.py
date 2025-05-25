"""
Tests for DanQ and ChromDragoNN models.
"""

import pytest
import torch
import numpy as np
from genomic_lightning.models.danq import DanQModel
from genomic_lightning.models.chromdragonn import ChromDragoNNModel


@pytest.fixture
def sample_input():
    """Create sample input tensor for testing."""
    batch_size = 4
    seq_length = 1000
    n_features = 4  # A, C, G, T
    
    # Create random one-hot encoded DNA sequences
    x = torch.zeros((batch_size, n_features, seq_length))
    for i in range(batch_size):
        for j in range(seq_length):
            # Set one nucleotide to 1 for each position
            x[i, np.random.randint(0, n_features), j] = 1.0
    
    return x


def test_danq_forward(sample_input):
    """Test DanQ model forward pass."""
    n_outputs = 919
    
    # Create model
    model = DanQModel(
        sequence_length=1000,
        n_genomic_features=4,
        n_outputs=n_outputs,
    )
    
    # Run forward pass
    output = model(sample_input)
    
    # Check output shape
    assert output.shape == (sample_input.shape[0], n_outputs)
    
    # Check output values are between 0 and 1 (sigmoid output)
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)
    

def test_chromdragonn_forward(sample_input):
    """Test ChromDragoNN model forward pass."""
    n_outputs = 919
    
    # Create model
    model = ChromDragoNNModel(
        sequence_length=1000,
        n_genomic_features=4,
        n_outputs=n_outputs,
    )
    
    # Run forward pass
    output = model(sample_input)
    
    # Check output shape
    assert output.shape == (sample_input.shape[0], n_outputs)
    
    # Check output values are between 0 and 1 (sigmoid output)
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)


def test_danq_gradient_flow(sample_input):
    """Test that gradients flow correctly through DanQ model."""
    n_outputs = 919
    
    # Create model
    model = DanQModel(
        sequence_length=1000,
        n_genomic_features=4,
        n_outputs=n_outputs,
    )
    
    # Run forward pass with grad enabled
    output = model(sample_input)
    
    # Compute loss and backward
    loss = output.sum()
    loss.backward()
    
    # Check that gradients are computed
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_chromdragonn_gradient_flow(sample_input):
    """Test that gradients flow correctly through ChromDragoNN model."""
    n_outputs = 919
    
    # Create model
    model = ChromDragoNNModel(
        sequence_length=1000,
        n_genomic_features=4,
        n_outputs=n_outputs,
    )
    
    # Run forward pass with grad enabled
    output = model(sample_input)
    
    # Compute loss and backward
    loss = output.sum()
    loss.backward()
    
    # Check that gradients are computed
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
