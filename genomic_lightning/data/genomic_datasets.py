"""Direct implementations of genomic datasets without relying on legacy samplers."""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from typing import Dict, Any, Optional, Union, List, Tuple


class GenomicSequenceDataset(Dataset):
    """Dataset for genomic sequence data stored in HDF5 format.
    
    This dataset directly reads sequence and target data from HDF5 files
    without requiring legacy sampler code.
    """
    
    def __init__(
        self,
        sequence_file: str,
        target_file: str,
        mode: str = 'train',
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        sequence_key: str = 'sequences',
        target_key: str = 'targets',
        metadata_key: Optional[str] = 'metadata',
    ):
        """Initialize the genomic sequence dataset.
        
        Args:
            sequence_file: Path to H5 file containing sequences
            target_file: Path to H5 file containing targets
            mode: 'train', 'validate', or 'test'
            transform: Function to transform sequences
            target_transform: Function to transform targets
            split_ratio: Train/validation/test split ratio
            sequence_key: Key for sequences in H5 file
            target_key: Key for targets in H5 file
            metadata_key: Optional key for metadata in H5 file
        """
        self.sequence_file = sequence_file
        self.target_file = target_file
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.split_ratio = split_ratio
        self.sequence_key = sequence_key
        self.target_key = target_key
        self.metadata_key = metadata_key
        
        # Check if files exist
        if not os.path.exists(sequence_file):
            raise FileNotFoundError(f"Sequence file not found: {sequence_file}")
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"Target file not found: {target_file}")
        
        # Get dataset size and calculate split indices
        with h5py.File(sequence_file, 'r') as f:
            self.dataset_size = f[sequence_key].shape[0]
        
        self.train_size = int(self.dataset_size * split_ratio[0])
        self.val_size = int(self.dataset_size * split_ratio[1])
        self.test_size = self.dataset_size - self.train_size - self.val_size
        
        # Set up indices based on mode
        if mode == 'train':
            self.start_idx = 0
            self.end_idx = self.train_size
        elif mode == 'validate':
            self.start_idx = self.train_size
            self.end_idx = self.train_size + self.val_size
        elif mode == 'test':
            self.start_idx = self.train_size + self.val_size
            self.end_idx = self.dataset_size
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train', 'validate', or 'test'.")
    
    def __len__(self):
        """Return the dataset size for the current mode."""
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        """Get a single sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'sequence', 'targets', and optionally 'metadata'
        """
        # Convert to absolute index in the dataset
        abs_idx = idx + self.start_idx
        
        # Ensure index is in range
        if abs_idx < self.start_idx or abs_idx >= self.end_idx:
            raise IndexError(f"Index {idx} out of range for {self.mode} split")
        
        # Read sequence from H5 file
        with h5py.File(self.sequence_file, 'r') as f:
            sequence = f[self.sequence_key][abs_idx]
        
        # Read target from H5 file
        with h5py.File(self.target_file, 'r') as f:
            target = f[self.target_key][abs_idx]
            
            # Read metadata if available
            metadata = None
            if self.metadata_key and self.metadata_key in f:
                try:
                    metadata = f[self.metadata_key][abs_idx]
                except Exception:
                    pass
        
        # Apply transformations
        if self.transform:
            sequence = self.transform(sequence)
        
        if self.target_transform:
            target = self.target_transform(target)
            
        # Convert to tensors
        result = {
            'sequence': torch.tensor(sequence, dtype=torch.float32),
            'targets': torch.tensor(target, dtype=torch.float32)
        }
        
        # Add metadata if available
        if metadata is not None:
            result['metadata'] = metadata
            
        return result


class FastaSequenceDataset(Dataset):
    """Dataset for genomic sequence data stored in FASTA format.
    
    This dataset reads sequences from FASTA files and targets from a separate file.
    """
    
    def __init__(
        self,
        fasta_file: str,
        target_file: str,
        mode: str = 'train',
        sequence_length: int = 1000,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ):
        """Initialize the FASTA sequence dataset.
        
        Args:
            fasta_file: Path to FASTA file containing sequences
            target_file: Path to file containing targets (CSV, TSV, or H5)
            mode: 'train', 'validate', or 'test'
            sequence_length: Length of sequences to extract
            transform: Function to transform sequences
            target_transform: Function to transform targets
            split_ratio: Train/validation/test split ratio
        """
        self.fasta_file = fasta_file
        self.target_file = target_file
        self.mode = mode
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_transform = target_transform
        self.split_ratio = split_ratio
        
        # Check if files exist
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"Target file not found: {target_file}")
        
        # Load sequences and targets
        self._load_data()
        
        # Set up split indices
        self._setup_split_indices()
    
    def _load_data(self):
        """Load sequences and targets from files."""
        # Load sequences from FASTA
        try:
            from Bio import SeqIO
            self.sequences = list(SeqIO.parse(self.fasta_file, "fasta"))
            self.dataset_size = len(self.sequences)
        except ImportError:
            raise ImportError("BioPython is required to read FASTA files. Install with 'pip install biopython'.")
        
        # Load targets based on file extension
        _, ext = os.path.splitext(self.target_file)
        
        if ext.lower() == '.h5':
            with h5py.File(self.target_file, 'r') as f:
                # Assume 'targets' is the key for target data
                self.targets = f['targets'][:]
        
        elif ext.lower() in ['.csv', '.tsv']:
            try:
                import pandas as pd
                delimiter = ',' if ext.lower() == '.csv' else '\t'
                df = pd.read_csv(self.target_file, delimiter=delimiter)
                
                # Assume all columns except the first one are targets
                self.targets = df.iloc[:, 1:].values
            except ImportError:
                raise ImportError("Pandas is required to read CSV/TSV files. Install with 'pip install pandas'.")
        
        else:
            raise ValueError(f"Unsupported target file format: {ext}")
        
        # Ensure sequences and targets have the same length
        if len(self.sequences) != len(self.targets):
            raise ValueError(
                f"Mismatch between sequences ({len(self.sequences)}) "
                f"and targets ({len(self.targets)}) count"
            )
    
    def _setup_split_indices(self):
        """Set up indices for train/validation/test splits."""
        self.train_size = int(self.dataset_size * self.split_ratio[0])
        self.val_size = int(self.dataset_size * self.split_ratio[1])
        self.test_size = self.dataset_size - self.train_size - self.val_size
        
        # Set up indices based on mode
        if self.mode == 'train':
            self.start_idx = 0
            self.end_idx = self.train_size
        elif self.mode == 'validate':
            self.start_idx = self.train_size
            self.end_idx = self.train_size + self.val_size
        elif self.mode == 'test':
            self.start_idx = self.train_size + self.val_size
            self.end_idx = self.dataset_size
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'train', 'validate', or 'test'.")
    
    def one_hot_encode(self, seq):
        """One-hot encode a DNA sequence.
        
        Args:
            seq: DNA sequence string
            
        Returns:
            One-hot encoded sequence as numpy array (4 x sequence_length)
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        seq = seq.upper()
        
        # Initialize one-hot encoded array
        one_hot = np.zeros((4, len(seq)), dtype=np.float32)
        
        for i, nucleotide in enumerate(seq):
            if nucleotide in mapping:
                if mapping[nucleotide] < 4:  # Skip 'N'
                    one_hot[mapping[nucleotide], i] = 1.0
            
        return one_hot
    
    def __len__(self):
        """Return the dataset size for the current mode."""
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        """Get a single sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'sequence' and 'targets'
        """
        # Convert to absolute index in the dataset
        abs_idx = idx + self.start_idx
        
        # Get sequence and target
        sequence_record = self.sequences[abs_idx]
        target = self.targets[abs_idx]
        
        # Extract sequence string
        sequence_str = str(sequence_record.seq)
        
        # One-hot encode the sequence
        sequence = self.one_hot_encode(sequence_str)
        
        # Apply transformations
        if self.transform:
            sequence = self.transform(sequence)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        # Convert to tensors
        return {
            'sequence': torch.tensor(sequence, dtype=torch.float32),
            'targets': torch.tensor(target, dtype=torch.float32),
            'metadata': {'id': sequence_record.id, 'description': sequence_record.description}
        }
