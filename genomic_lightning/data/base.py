"""Base data module for genomic datasets."""

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List, Tuple, Union


class GenomicDataModule(pl.LightningDataModule):
    """Base Lightning DataModule for genomic data."""
    
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """Initialize the base genomic data module.
        
        Args:
            batch_size: Batch size for training and evaluation
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory during data loading
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # These will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for each stage.
        
        This method should be overridden by subclasses to implement
        dataset-specific loading logic.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        raise NotImplementedError("Subclasses must implement setup()")
    
    def train_dataloader(self):
        """Get train dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset not set. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )
    
    def val_dataloader(self):
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset not set. Call setup() first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        """Get test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset not set. Call setup() first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def predict_dataloader(self):
        """Get prediction dataloader."""
        if self.predict_dataset is None:
            raise ValueError("Predict dataset not set. Call setup() first.")
        
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
