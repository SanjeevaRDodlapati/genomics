"""Data module for genomic data using UAVarPrior samplers."""

import pytorch_lightning as pl
import torch
from typing import Optional, Dict, Any

from genomic_lightning.data.base import GenomicDataModule
from genomic_lightning.data.sampler_adapter import SamplerDatasetAdapter


class SamplerDataModule(GenomicDataModule):
    """DataModule that wraps UAVarPrior samplers for use with Lightning.
    
    This allows for easy integration with existing UAVarPrior samplers.
    """
    
    def __init__(
        self,
        sampler,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None, 
        max_test_samples: Optional[int] = None,
        cache_validation: bool = True,
        cache_test: bool = True,
    ):
        """Initialize the sampler-based data module.
        
        Args:
            sampler: UAVarPrior sampler instance
            batch_size: Batch size for training and evaluation
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for data loading
            max_train_samples: Maximum number of training samples to use
            max_val_samples: Maximum number of validation samples to use
            max_test_samples: Maximum number of test samples to use
            cache_validation: Whether to cache validation samples in memory
            cache_test: Whether to cache test samples in memory
        """
        super().__init__(batch_size, num_workers, pin_memory)
        self.sampler = sampler
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.cache_validation = cache_validation
        self.cache_test = cache_test
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Set up training dataset
        if stage == 'fit' or stage is None:
            self.train_dataset = SamplerDatasetAdapter(
                self.sampler,
                mode='train',
                max_samples=self.max_train_samples,
                cache_samples=False  # Don't cache training samples to save memory
            )
            
            self.val_dataset = SamplerDatasetAdapter(
                self.sampler,
                mode='validate',
                max_samples=self.max_val_samples,
                cache_samples=self.cache_validation
            )
        
        # Set up test dataset
        if stage == 'test' or stage is None:
            self.test_dataset = SamplerDatasetAdapter(
                self.sampler,
                mode='test',
                max_samples=self.max_test_samples,
                cache_samples=self.cache_test
            )
        
        # Set up prediction dataset
        if stage == 'predict' or stage is None:
            # For prediction, use test mode but with a different name
            self.predict_dataset = SamplerDatasetAdapter(
                self.sampler,
                mode='test',
                max_samples=self.max_test_samples,
                cache_samples=self.cache_test
            )
