"""
Tests for genomic metrics.
"""

import pytest
import torch
import numpy as np
from genomic_lightning.metrics.genomic_metrics import GenomicAUPRC, TopKAccuracy, PositionalAUROC


@pytest.fixture
def sample_predictions():
    """Create sample predictions and targets for testing."""
    # Create random predictions and binary targets
    batch_size = 16
    n_classes = 10
    
    # Predictions as probabilities between 0 and 1
    preds = torch.rand(batch_size, n_classes)
    
    # Targets as binary values (mostly 0, some 1s)
    targets = torch.zeros(batch_size, n_classes)
    for i in range(batch_size):
        # Add 1-3 positive classes per sample
        n_positive = torch.randint(1, 4, (1,)).item()
        positive_indices = torch.randperm(n_classes)[:n_positive]
        targets[i, positive_indices] = 1.0
        
    return preds, targets


def test_genomic_auprc(sample_predictions):
    """Test GenomicAUPRC metric."""
    preds, targets = sample_predictions
    n_classes = targets.shape[1]
    
    # Test with macro averaging
    metric = GenomicAUPRC(num_classes=n_classes, average="macro")
    metric.update(preds, targets)
    result = metric.compute()
    
    # Check result is a tensor with correct shape and valid AUPRC values
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0  # Scalar
    assert result >= 0 and result <= 1
    
    # Test with per-class results
    metric = GenomicAUPRC(num_classes=n_classes, average=None)
    metric.update(preds, targets)
    result = metric.compute()
    
    # Check result is a dictionary with values for each class
    assert isinstance(result, dict)
    assert len(result) == n_classes
    for class_idx, auprc in result.items():
        assert 0 <= class_idx < n_classes
        assert auprc >= 0 and auprc <= 1


def test_top_k_accuracy(sample_predictions):
    """Test TopKAccuracy metric."""
    preds, targets = sample_predictions
    
    # Test with k=3
    k = 3
    metric = TopKAccuracy(k=k)
    metric.update(preds, targets)
    result = metric.compute()
    
    # Check result is a tensor with valid accuracy value
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0  # Scalar
    assert result >= 0 and result <= 1
    
    # Manual calculation to verify
    correct = 0
    for i in range(preds.shape[0]):
        # Get top-k indices
        _, top_indices = torch.topk(preds[i], k)
        top_set = set(top_indices.tolist())
        
        # Get positive class indices
        pos_indices = torch.nonzero(targets[i], as_tuple=True)[0]
        pos_set = set(pos_indices.tolist())
        
        # Check if any positive class is in top-k
        if len(pos_set.intersection(top_set)) > 0:
            correct += 1
            
    manual_accuracy = correct / preds.shape[0]
    assert abs(result.item() - manual_accuracy) < 1e-6


def test_positional_auroc():
    """Test PositionalAUROC metric."""
    # Create sample data with position information
    batch_size = 20
    n_classes = 5
    sequence_length = 100
    num_bins = 5
    
    # Create predictions, targets, and positions
    preds = torch.rand(batch_size, n_classes)
    targets = torch.zeros(batch_size, n_classes)
    for i in range(batch_size):
        positive_idx = torch.randint(0, n_classes, (1,)).item()
        targets[i, positive_idx] = 1
        
    # Create positions spanning the sequence
    positions = torch.randint(0, sequence_length, (batch_size,))
    
    # Create metric
    metric = PositionalAUROC(sequence_length=sequence_length, num_bins=num_bins)
    
    # Update metric with data
    metric.update(preds, targets, positions)
    
    # Compute results
    results = metric.compute()
    
    # Check results dictionary
    assert isinstance(results, dict)
    assert "average" in results
    
    # Check result values
    assert results["average"] >= 0 and results["average"] <= 1
    
    # Check all bins
    for bin_idx in range(num_bins):
        bin_key = f"bin_{bin_idx}"
        if bin_key in results:
            assert results[bin_key] >= 0 and results[bin_key] <= 1
