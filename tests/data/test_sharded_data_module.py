"""
Tests for sharded data module.
"""

import pytest
import numpy as np
import h5py
import torch
import os
import tempfile
from genomic_lightning.data.sharded_data_module import ShardedGenomicDataset, ShardedGenomicDataModule


@pytest.fixture
def sample_shard_files():
    """Create temporary HDF5 shards for testing."""
    temp_files = []
    
    # Create 2 shard files for training
    for i in range(2):
        temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        temp_files.append(temp_file.name)
        
        with h5py.File(temp_file.name, 'w') as f:
            # Create sequences dataset
            sequences = np.zeros((100, 4, 1000), dtype=np.float32)
            for j in range(100):
                for k in range(1000):
                    # One-hot encode
                    nuc = np.random.randint(0, 4)
                    sequences[j, nuc, k] = 1.0
                    
            # Create targets dataset
            targets = np.random.rand(100, 919).astype(np.float32)
            
            # Save datasets
            f.create_dataset("sequences", data=sequences)
            f.create_dataset("targets", data=targets)
            
    # Create 1 shard file for validation
    temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    temp_files.append(temp_file.name)
    
    with h5py.File(temp_file.name, 'w') as f:
        # Create sequences dataset
        sequences = np.zeros((50, 4, 1000), dtype=np.float32)
        for j in range(50):
            for k in range(1000):
                # One-hot encode
                nuc = np.random.randint(0, 4)
                sequences[j, nuc, k] = 1.0
                
        # Create targets dataset
        targets = np.random.rand(50, 919).astype(np.float32)
        
        # Save datasets
        f.create_dataset("sequences", data=sequences)
        f.create_dataset("targets", data=targets)
    
    yield temp_files
    
    # Clean up temp files
    for temp_file in temp_files:
        os.unlink(temp_file)


def test_sharded_dataset_init(sample_shard_files):
    """Test ShardedGenomicDataset initialization."""
    dataset = ShardedGenomicDataset(
        shard_paths=sample_shard_files[:2],
        x_dset="sequences",
        y_dset="targets",
    )
    
    # Check total samples
    assert dataset.total_samples == 200  # 2 shards with 100 samples each
    
    # Check shapes
    assert dataset.x_shape == (4, 1000)
    assert dataset.y_shape == (919,)


def test_sharded_dataset_get_shard(sample_shard_files):
    """Test locating samples in shards."""
    dataset = ShardedGenomicDataset(
        shard_paths=sample_shard_files[:2],
        x_dset="sequences",
        y_dset="targets",
    )
    
    # Test first shard
    shard_idx, local_idx = dataset.get_shard_for_index(50)
    assert shard_idx == 0
    assert local_idx == 50
    
    # Test second shard
    shard_idx, local_idx = dataset.get_shard_for_index(150)
    assert shard_idx == 1
    assert local_idx == 50


def test_sharded_dataset_iterator(sample_shard_files):
    """Test ShardedGenomicDataset iterator."""
    dataset = ShardedGenomicDataset(
        shard_paths=sample_shard_files[:2],
        x_dset="sequences",
        y_dset="targets",
        cache_size=50,  # Small cache to test multiple loads
    )
    
    # Count samples from iterator
    count = 0
    for x, y in dataset:
        # Check shapes
        assert x.shape == (4, 1000)
        assert y.shape == (919,)
        
        # Check tensor types
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        
        count += 1
        if count >= 10:  # Just test a few to keep test fast
            break
    
    assert count > 0


def test_sharded_data_module(sample_shard_files):
    """Test ShardedGenomicDataModule."""
    # Use first two files for training, last for validation
    data_module = ShardedGenomicDataModule(
        train_shards=sample_shard_files[:2],
        val_shards=[sample_shard_files[2]],
        batch_size=16,
        num_workers=1,
        cache_size=50,
    )
    
    # Setup data module
    data_module.setup()
    
    # Create data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Check batch size
    batch = next(iter(train_loader))
    assert len(batch) == 2  # (x, y)
    assert batch[0].shape[0] == 16  # Batch size
    
    # Test batch dimensions
    assert batch[0].shape[1:] == (4, 1000)  # (n_features, seq_length)
    assert batch[1].shape[1:] == (919,)  # (n_outputs)
    
    # Check validation loader
    val_batch = next(iter(val_loader))
    assert len(val_batch) == 2
    assert val_batch[0].shape[0] == 16  # Batch size
