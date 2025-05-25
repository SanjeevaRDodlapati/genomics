"""DataModules for genomic datasets."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple

from genomic_lightning.data.base import GenomicDataModule
from genomic_lightning.data.genomic_datasets import (
    GenomicSequenceDataset,
    FastaSequenceDataset
)


class H5DataModule(GenomicDataModule):
    """DataModule for working with HDF5-formatted genomic data."""
    
    def __init__(
        self,
        sequence_file: str,
        target_file: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        sequence_key: str = 'sequences',
        target_key: str = 'targets',
        metadata_key: Optional[str] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ):
        """Initialize the H5 data module.
        
        Args:
            sequence_file: Path to H5 file containing sequences
            target_file: Path to H5 file containing targets
            batch_size: Batch size for training and evaluation
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for data loading
            split_ratio: Train/validation/test split ratio
            sequence_key: Key for sequences in H5 file
            target_key: Key for targets in H5 file
            metadata_key: Optional key for metadata in H5 file
            transform: Function to transform sequences
            target_transform: Function to transform targets
        """
        super().__init__(batch_size, num_workers, pin_memory)
        self.sequence_file = sequence_file
        self.target_file = target_file
        self.split_ratio = split_ratio
        self.sequence_key = sequence_key
        self.target_key = target_key
        self.metadata_key = metadata_key
        self.transform = transform
        self.target_transform = target_transform
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Set up training and validation datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = GenomicSequenceDataset(
                sequence_file=self.sequence_file,
                target_file=self.target_file,
                mode='train',
                transform=self.transform,
                target_transform=self.target_transform,
                split_ratio=self.split_ratio,
                sequence_key=self.sequence_key,
                target_key=self.target_key,
                metadata_key=self.metadata_key,
            )
            
            self.val_dataset = GenomicSequenceDataset(
                sequence_file=self.sequence_file,
                target_file=self.target_file,
                mode='validate',
                transform=self.transform,
                target_transform=self.target_transform,
                split_ratio=self.split_ratio,
                sequence_key=self.sequence_key,
                target_key=self.target_key,
                metadata_key=self.metadata_key,
            )
        
        # Set up test dataset
        if stage == 'test' or stage is None:
            self.test_dataset = GenomicSequenceDataset(
                sequence_file=self.sequence_file,
                target_file=self.target_file,
                mode='test',
                transform=self.transform,
                target_transform=self.target_transform,
                split_ratio=self.split_ratio,
                sequence_key=self.sequence_key,
                target_key=self.target_key,
                metadata_key=self.metadata_key,
            )
        
        # Set up prediction dataset (same as test dataset)
        if stage == 'predict' or stage is None:
            self.predict_dataset = GenomicSequenceDataset(
                sequence_file=self.sequence_file,
                target_file=self.target_file,
                mode='test',
                transform=self.transform,
                target_transform=self.target_transform,
                split_ratio=self.split_ratio,
                sequence_key=self.sequence_key,
                target_key=self.target_key,
                metadata_key=self.metadata_key,
            )


class FastaDataModule(GenomicDataModule):
    """DataModule for working with FASTA-formatted genomic data."""
    
    def __init__(
        self,
        fasta_file: str,
        target_file: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        sequence_length: int = 1000,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ):
        """Initialize the FASTA data module.
        
        Args:
            fasta_file: Path to FASTA file containing sequences
            target_file: Path to file containing targets (CSV, TSV, or H5)
            batch_size: Batch size for training and evaluation
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for data loading
            sequence_length: Length of sequences to extract
            split_ratio: Train/validation/test split ratio
            transform: Function to transform sequences
            target_transform: Function to transform targets
        """
        super().__init__(batch_size, num_workers, pin_memory)
        self.fasta_file = fasta_file
        self.target_file = target_file
        self.sequence_length = sequence_length
        self.split_ratio = split_ratio
        self.transform = transform
        self.target_transform = target_transform
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Set up training and validation datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = FastaSequenceDataset(
                fasta_file=self.fasta_file,
                target_file=self.target_file,
                mode='train',
                sequence_length=self.sequence_length,
                transform=self.transform,
                target_transform=self.target_transform,
                split_ratio=self.split_ratio,
            )
            
            self.val_dataset = FastaSequenceDataset(
                fasta_file=self.fasta_file,
                target_file=self.target_file,
                mode='validate',
                sequence_length=self.sequence_length,
                transform=self.transform,
                target_transform=self.target_transform,
                split_ratio=self.split_ratio,
            )
        
        # Set up test dataset
        if stage == 'test' or stage is None:
            self.test_dataset = FastaSequenceDataset(
                fasta_file=self.fasta_file,
                target_file=self.target_file,
                mode='test',
                sequence_length=self.sequence_length,
                transform=self.transform,
                target_transform=self.target_transform,
                split_ratio=self.split_ratio,
            )
        
        # Set up prediction dataset (same as test dataset)
        if stage == 'predict' or stage is None:
            self.predict_dataset = FastaSequenceDataset(
                fasta_file=self.fasta_file,
                target_file=self.target_file,
                mode='test',
                sequence_length=self.sequence_length,
                transform=self.transform,
                target_transform=self.target_transform,
                split_ratio=self.split_ratio,
            )
