"""
Large-scale genomic data module with sharding and streaming capabilities.
"""

import os
import torch
import h5py
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Union, Tuple
from torch.utils.data import DataLoader, IterableDataset

class ShardedGenomicDataset(IterableDataset):
    """
    Iterable dataset for streaming large genomic data from sharded files.
    
    This dataset can read data from multiple sharded files without
    loading everything into memory at once.
    """
    
    def __init__(
        self,
        shard_paths: List[str],
        x_dset: str = "sequences",
        y_dset: str = "targets",
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        cache_size: int = 1000,
        transform=None
    ):
        """
        Initialize the sharded dataset.
        
        Args:
            shard_paths: List of HDF5 shard file paths
            x_dset: Name of inputs dataset in HDF5 files
            y_dset: Name of targets dataset in HDF5 files
            batch_size: Batch size
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
            cache_size: Number of samples to cache in memory
            transform: Optional transform to apply to samples
        """
        super().__init__()
        self.shard_paths = shard_paths
        self.x_dset = x_dset
        self.y_dset = y_dset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.cache_size = cache_size
        self.transform = transform
        
        # Get total dataset size and shapes
        self.total_samples = 0
        self.shard_sizes = []
        self.x_shape = None
        self.y_shape = None
        
        for shard in self.shard_paths:
            with h5py.File(shard, 'r') as f:
                shard_size = f[self.x_dset].shape[0]
                self.shard_sizes.append(shard_size)
                self.total_samples += shard_size
                
                if self.x_shape is None:
                    self.x_shape = f[self.x_dset].shape[1:]
                    self.y_shape = f[self.y_dset].shape[1:]
                    
        # Cumulative sizes for locating samples
        self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)
        
        # Cache to improve performance
        self.cache_x = None
        self.cache_y = None
        self.cache_indices = None
        
    def __len__(self):
        return self.total_samples
    
    def get_shard_for_index(self, idx: int) -> Tuple[int, int]:
        """
        Find which shard contains the given index.
        
        Args:
            idx: Global index
            
        Returns:
            Tuple of (shard_idx, local_idx)
        """
        shard_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        local_idx = idx - self.cumulative_sizes[shard_idx]
        return shard_idx, local_idx
    
    def load_cache(self, start_idx: int, worker_info=None):
        """
        Load a chunk of data into memory cache.
        
        Args:
            start_idx: Starting index
            worker_info: DataLoader worker info
        """
        end_idx = min(start_idx + self.cache_size, self.total_samples)
        cache_indices = list(range(start_idx, end_idx))
        
        # Initialize empty cache arrays
        self.cache_x = np.zeros((len(cache_indices),) + self.x_shape, dtype=np.float32)
        self.cache_y = np.zeros((len(cache_indices),) + self.y_shape, dtype=np.float32)
        
        # Fill cache from appropriate shards
        cache_pos = 0
        prev_shard_idx = -1
        file_handle = None
        
        for idx in cache_indices:
            shard_idx, local_idx = self.get_shard_for_index(idx)
            
            # Only open new file when shard changes
            if shard_idx != prev_shard_idx:
                if file_handle is not None:
                    file_handle.close()
                file_handle = h5py.File(self.shard_paths[shard_idx], 'r')
                prev_shard_idx = shard_idx
                
            # Read data into cache
            self.cache_x[cache_pos] = file_handle[self.x_dset][local_idx]
            self.cache_y[cache_pos] = file_handle[self.y_dset][local_idx]
            cache_pos += 1
            
        if file_handle is not None:
            file_handle.close()
            
        self.cache_indices = cache_indices

    def __iter__(self):
        """
        Iterator for the dataset.
        
        Yields:
            Tuple of (x, y) samples
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # Each worker processes a different part of the dataset
        per_worker = int(np.ceil(self.total_samples / num_workers))
        worker_start = worker_id * per_worker
        worker_end = min(worker_start + per_worker, self.total_samples)
        
        # Set up shuffled indices for this worker
        rng = np.random.RandomState(self.seed)
        indices = list(range(worker_start, worker_end))
        if self.shuffle:
            rng.shuffle(indices)
            
        # Process batches
        for i in range(0, len(indices), self.cache_size):
            cache_indices = indices[i:i + self.cache_size]
            if not cache_indices:
                break
                
            # Load cache for current chunk
            self.load_cache(cache_indices[0], worker_info)
            
            # Return samples from cache
            for idx in cache_indices:
                local_idx = self.cache_indices.index(idx)
                x = torch.from_numpy(self.cache_x[local_idx])
                y = torch.from_numpy(self.cache_y[local_idx])
                
                if self.transform:
                    x, y = self.transform(x, y)
                    
                yield x, y


class ShardedGenomicDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for sharded genomic datasets.
    """
    
    def __init__(
        self,
        train_shards: List[str],
        val_shards: List[str],
        test_shards: Optional[List[str]] = None,
        x_dset: str = "sequences",
        y_dset: str = "targets",
        batch_size: int = 32,
        num_workers: int = 4,
        cache_size: int = 1000,
        transform=None
    ):
        """
        Initialize the sharded genomic data module.
        
        Args:
            train_shards: List of training shard file paths
            val_shards: List of validation shard file paths
            test_shards: List of test shard file paths (optional)
            x_dset: Name of input dataset in HDF5 files
            y_dset: Name of target dataset in HDF5 files
            batch_size: Batch size
            num_workers: Number of data loader workers
            cache_size: Number of samples to cache in memory
            transform: Optional transform to apply to samples
        """
        super().__init__()
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.test_shards = test_shards
        self.x_dset = x_dset
        self.y_dset = y_dset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.transform = transform
        
    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets at the beginning of fit or test.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test')
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = ShardedGenomicDataset(
                shard_paths=self.train_shards,
                x_dset=self.x_dset,
                y_dset=self.y_dset,
                batch_size=self.batch_size,
                shuffle=True,
                cache_size=self.cache_size,
                transform=self.transform
            )
            
            self.val_dataset = ShardedGenomicDataset(
                shard_paths=self.val_shards,
                x_dset=self.x_dset,
                y_dset=self.y_dset,
                batch_size=self.batch_size,
                shuffle=False,
                cache_size=self.cache_size,
                transform=self.transform
            )
            
        if stage == 'test' or stage is None:
            if self.test_shards:
                self.test_dataset = ShardedGenomicDataset(
                    shard_paths=self.test_shards,
                    x_dset=self.x_dset,
                    y_dset=self.y_dset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    cache_size=self.cache_size,
                    transform=self.transform
                )
    
    def train_dataloader(self):
        """Create the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Create the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Create the test data loader."""
        if self.test_shards:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return None
