"""Adapter for UAVarPrior-style samplers."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, Optional, Union, List


class SamplerDatasetAdapter(Dataset):
    """Adapter to convert UAVarPrior samplers into PyTorch datasets.
    
    This class wraps a UAVarPrior-style sampler into a PyTorch Dataset,
    making it compatible with PyTorch DataLoader and Lightning DataModule.
    """
    
    def __init__(
        self,
        sampler,
        mode: str = 'train',
        max_samples: Optional[int] = None,
        cache_samples: bool = False
    ):
        """Initialize the sampler adapter.
        
        Args:
            sampler: UAVarPrior-style sampler instance
            mode: 'train', 'validate', 'test', or 'predict'
            max_samples: Maximum number of samples to use (None for all)
            cache_samples: If True, samples are cached in memory
        """
        self.sampler = sampler
        self.mode = mode
        self.max_samples = max_samples
        self.cache_samples = cache_samples
        
        # Set sampler mode
        if hasattr(self.sampler, 'set_mode'):
            self.sampler.set_mode(mode)
        
        # Cache for samples if enabled
        self.cache = [] if cache_samples else None
        
        # Estimate dataset size
        self._size = self._estimate_size()
    
    def _estimate_size(self) -> int:
        """Estimate the dataset size based on sampler information.
        
        Returns:
            Estimated dataset size
        """
        # If max_samples is set, use that as the size
        if self.max_samples is not None:
            return self.max_samples
        
        # Try to get size from sampler
        if hasattr(self.sampler, 'get_dataset_size'):
            size = self.sampler.get_dataset_size()
            if size is not None and size > 0:
                return size
        
        # Default size for infinite samplers
        return 100000
    
    def __len__(self) -> int:
        """Return the dataset size.
        
        Returns:
            Dataset size
        """
        return self._size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index (may be ignored by stateful samplers)
            
        Returns:
            Dictionary containing tensors for sequence and targets
        """
        # Return cached sample if available
        if self.cache_samples and idx < len(self.cache):
            return self.cache[idx]
        
        # Sample from the sampler
        # Note: stateful samplers may ignore idx and return samples sequentially
        sample = self.sampler.sample()
        
        # Convert numpy arrays to tensors
        tensor_sample = {
            'sequence': torch.tensor(sample['sequence'], dtype=torch.float32),
            'targets': torch.tensor(sample['targets'], dtype=torch.float32)
        }
        
        # Add metadata if available in the original sample
        if 'metadata' in sample:
            tensor_sample['metadata'] = sample['metadata']
        
        # Cache the sample if enabled
        if self.cache_samples:
            self.cache.append(tensor_sample)
        
        return tensor_sample
