"""
Utilities for handling and converting genomic data samplers.

This module provides utilities for working with data samplers, especially
for converting between different sampling frameworks and creating data loaders
compatible with PyTorch Lightning.
"""

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List, Callable, Iterator
import logging
import os
import h5py

logger = logging.getLogger(__name__)

class LegacySamplerWrapper(Dataset):
    """
    Wrapper for legacy genomic data samplers to make them compatible with PyTorch DataLoaders.
    """
    
    def __init__(
        self, 
        sampler: Any, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize the wrapper for a legacy sampler.
        
        Args:
            sampler: Legacy sampler object that implements __getitem__ and __len__
            transform: Optional transform to apply to the sequences
            target_transform: Optional transform to apply to the targets
        """
        self.sampler = sampler
        self.transform = transform
        self.target_transform = target_transform
        
        # Check if sampler has required methods
        if not hasattr(self.sampler, '__getitem__'):
            raise ValueError("Sampler must implement __getitem__")
        if not hasattr(self.sampler, '__len__'):
            raise ValueError("Sampler must implement __len__")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (sequence, target)
        """
        # Get sample from legacy sampler
        sample = self.sampler[idx]
        
        # Legacy samplers might return various formats
        if isinstance(sample, tuple) and len(sample) == 2:
            # Standard (sequence, target) format
            sequence, target = sample
        elif isinstance(sample, dict):
            # Dictionary format
            if 'sequence' in sample and 'target' in sample:
                sequence, target = sample['sequence'], sample['target']
            else:
                raise ValueError(f"Unknown dictionary keys in sampler: {list(sample.keys())}")
        else:
            raise ValueError(f"Unknown sample format from sampler: {type(sample)}")
        
        # Convert to torch tensors if needed
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform is not None:
            sequence = self.transform(sequence)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sequence, target


class LegacyIterableSamplerWrapper(IterableDataset):
    """
    Wrapper for legacy genomic data samplers that use iterables instead of indexing.
    """
    
    def __init__(
        self, 
        sampler: Any, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize the wrapper for a legacy iterable sampler.
        
        Args:
            sampler: Legacy sampler object that implements __iter__
            transform: Optional transform to apply to the sequences
            target_transform: Optional transform to apply to the targets
        """
        self.sampler = sampler
        self.transform = transform
        self.target_transform = target_transform
        
        # Check if sampler has required methods
        if not hasattr(self.sampler, '__iter__'):
            raise ValueError("Sampler must implement __iter__")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create an iterator over the dataset.
        
        Returns:
            Iterator yielding (sequence, target) pairs
        """
        sampler_iter = iter(self.sampler)
        
        for sample in sampler_iter:
            # Handle different sample formats
            if isinstance(sample, tuple) and len(sample) == 2:
                sequence, target = sample
            elif isinstance(sample, dict):
                if 'sequence' in sample and 'target' in sample:
                    sequence, target = sample['sequence'], sample['target']
                else:
                    raise ValueError(f"Unknown dictionary keys in sampler: {list(sample.keys())}")
            else:
                raise ValueError(f"Unknown sample format from sampler: {type(sample)}")
            
            # Convert to torch tensors if needed
            if not isinstance(sequence, torch.Tensor):
                sequence = torch.tensor(sequence, dtype=torch.float32)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)
            
            # Apply transforms if provided
            if self.transform is not None:
                sequence = self.transform(sequence)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            yield sequence, target


def create_dataloader_from_legacy_sampler(
    sampler: Any, 
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    is_iterable: bool = False,
    **dataloader_kwargs
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a legacy genomic data sampler.
    
    Args:
        sampler: Legacy sampler object
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for loading
        transform: Optional transform to apply to sequences
        target_transform: Optional transform to apply to targets
        is_iterable: Whether the sampler is iterable-based (vs. index-based)
        **dataloader_kwargs: Additional arguments for DataLoader
        
    Returns:
        PyTorch DataLoader
    """
    if is_iterable:
        dataset = LegacyIterableSamplerWrapper(
            sampler=sampler,
            transform=transform,
            target_transform=target_transform
        )
        
        # Iterable datasets ignore the shuffle parameter
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            **dataloader_kwargs
        )
    else:
        dataset = LegacySamplerWrapper(
            sampler=sampler,
            transform=transform,
            target_transform=target_transform
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **dataloader_kwargs
        )


class H5Dataset(Dataset):
    """
    Dataset for loading genomic data from HDF5 files.
    """
    
    def __init__(
        self,
        h5_path: str,
        sequence_dataset: str = 'sequences',
        target_dataset: str = 'targets',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        memory_map: bool = False
    ):
        """
        Initialize the HDF5 dataset.
        
        Args:
            h5_path: Path to the HDF5 file
            sequence_dataset: Name of the dataset containing sequences
            target_dataset: Name of the dataset containing targets
            transform: Optional transform to apply to sequences
            target_transform: Optional transform to apply to targets
            memory_map: Whether to load the entire dataset into memory
        """
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
        self.h5_path = h5_path
        self.sequence_dataset = sequence_dataset
        self.target_dataset = target_dataset
        self.transform = transform
        self.target_transform = target_transform
        
        # Open the file and get dataset shapes
        with h5py.File(self.h5_path, 'r') as f:
            if self.sequence_dataset not in f:
                raise KeyError(f"Sequence dataset '{self.sequence_dataset}' not found in HDF5 file")
            if self.target_dataset not in f:
                raise KeyError(f"Target dataset '{self.target_dataset}' not found in HDF5 file")
            
            self.num_samples = f[self.sequence_dataset].shape[0]
            
            # Optional: Memory mapping
            if memory_map:
                self.sequences = f[self.sequence_dataset][:]
                self.targets = f[self.target_dataset][:]
                self.memory_mapped = True
            else:
                self.sequences = None
                self.targets = None
                self.memory_mapped = False
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (sequence, target)
        """
        if self.memory_mapped:
            sequence = self.sequences[idx]
            target = self.targets[idx]
        else:
            with h5py.File(self.h5_path, 'r') as f:
                sequence = f[self.sequence_dataset][idx]
                target = f[self.target_dataset][idx]
        
        # Convert to torch tensors
        sequence = torch.tensor(sequence, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform is not None:
            sequence = self.transform(sequence)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sequence, target


def create_sharded_dataset(
    input_files: List[str],
    output_dir: str,
    shard_size: int = 10000,
    compression: Optional[str] = 'gzip',
    sequence_dataset: str = 'sequences',
    target_dataset: str = 'targets',
    shuffle: bool = True
) -> List[str]:
    """
    Create sharded datasets from large genomic data files.
    
    Args:
        input_files: List of HDF5 files to shard
        output_dir: Directory to save the shards
        shard_size: Number of samples per shard
        compression: Compression to use for output files
        sequence_dataset: Name of the dataset containing sequences
        target_dataset: Name of the dataset containing targets
        shuffle: Whether to shuffle the data before sharding
        
    Returns:
        List of paths to the created shards
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total samples across all input files
    total_samples = 0
    sample_counts = []
    
    for input_file in input_files:
        with h5py.File(input_file, 'r') as f:
            if sequence_dataset not in f:
                raise KeyError(f"Sequence dataset '{sequence_dataset}' not found in {input_file}")
            if target_dataset not in f:
                raise KeyError(f"Target dataset '{target_dataset}' not found in {input_file}")
            
            count = f[sequence_dataset].shape[0]
            total_samples += count
            sample_counts.append(count)
    
    logger.info(f"Total samples across all input files: {total_samples}")
    
    # Create shard files
    num_shards = (total_samples + shard_size - 1) // shard_size
    shard_paths = []
    
    # Create random indices for shuffling
    if shuffle:
        all_indices = np.random.permutation(total_samples)
    else:
        all_indices = np.arange(total_samples)
    
    # Map global indices to file and local indices
    file_indices = []
    local_indices = []
    
    start_idx = 0
    for file_idx, count in enumerate(sample_counts):
        for i in range(count):
            file_indices.append(file_idx)
            local_indices.append(i)
        start_idx += count
    
    file_indices = np.array(file_indices)
    local_indices = np.array(local_indices)
    
    # Create shards
    for shard_idx in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.h5")
        shard_paths.append(shard_path)
        
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, total_samples)
        
        with h5py.File(shard_path, 'w') as out_file:
            # Get indices for this shard
            shard_global_indices = all_indices[start:end]
            shard_file_indices = file_indices[shard_global_indices]
            shard_local_indices = local_indices[shard_global_indices]
            
            # Create datasets - determine shapes first
            with h5py.File(input_files[0], 'r') as f:
                seq_shape = f[sequence_dataset].shape[1:]
                target_shape = f[target_dataset].shape[1:]
                seq_dtype = f[sequence_dataset].dtype
                target_dtype = f[target_dataset].dtype
            
            # Create output datasets
            seq_dataset = out_file.create_dataset(
                sequence_dataset, 
                shape=(end-start,) + seq_shape,
                dtype=seq_dtype,
                compression=compression
            )
            
            target_dataset = out_file.create_dataset(
                target_dataset,
                shape=(end-start,) + target_shape,
                dtype=target_dtype,
                compression=compression
            )
            
            # Copy data to shard
            unique_files = np.unique(shard_file_indices)
            
            for file_idx in unique_files:
                # Get indices for this file
                mask = (shard_file_indices == file_idx)
                shard_indices = np.arange(start, end)[mask]
                file_local_indices = shard_local_indices[mask]
                
                # Copy data from this file
                with h5py.File(input_files[file_idx], 'r') as f:
                    for i, (shard_idx, file_idx) in enumerate(zip(np.where(mask)[0], file_local_indices)):
                        seq_dataset[shard_idx] = f[sequence_dataset][file_idx]
                        target_dataset[shard_idx] = f[target_dataset][file_idx]
            
            # Add metadata
            out_file.attrs['shard_index'] = shard_idx
            out_file.attrs['num_shards'] = num_shards
            out_file.attrs['shard_size'] = shard_size
            out_file.attrs['actual_size'] = end - start
    
    logger.info(f"Created {num_shards} shards in {output_dir}")
    return shard_paths