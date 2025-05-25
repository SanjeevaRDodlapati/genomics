#!/usr/bin/env python
"""
Generate synthetic genomic data for model training.

This script creates synthetic genomic data in HDF5 format for training 
small deep learning models like DeepSEA or DanQ.
"""

import os
import numpy as np
import h5py
from typing import Tuple, List

def generate_one_hot_seq(length: int, num_sequences: int) -> np.ndarray:
    """
    Generate random one-hot encoded DNA sequences.
    
    Args:
        length: Length of each sequence
        num_sequences: Number of sequences to generate
        
    Returns:
        Array of shape (num_sequences, 4, length) with one-hot encoded sequences
    """
    # Generate random sequences with A,C,G,T represented as 0,1,2,3
    sequences = np.random.randint(0, 4, (num_sequences, length))
    
    # Convert to one-hot encoding
    one_hot = np.zeros((num_sequences, 4, length), dtype=np.float32)
    
    for i in range(num_sequences):
        for j in range(length):
            one_hot[i, sequences[i, j], j] = 1.0
            
    return one_hot

def generate_binary_targets(num_sequences: int, num_targets: int, sparsity: float = 0.95) -> np.ndarray:
    """
    Generate binary targets for sequences.
    
    Args:
        num_sequences: Number of sequences
        num_targets: Number of target features
        sparsity: Probability of a target being 0 (controls sparsity)
        
    Returns:
        Binary target array of shape (num_sequences, num_targets)
    """
    # Generate targets with specified sparsity
    targets = np.random.binomial(1, 1-sparsity, (num_sequences, num_targets)).astype(np.float32)
    return targets

def create_data_shard(
    output_path: str, 
    num_sequences: int,
    seq_length: int = 1000,
    num_targets: int = 919,
    compression: str = "gzip"
) -> None:
    """
    Create a data shard file with synthetic genomic data.
    
    Args:
        output_path: Path to save the HDF5 file
        num_sequences: Number of sequences to generate
        seq_length: Length of each sequence
        num_targets: Number of target features
        compression: Compression to use for HDF5 datasets
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate data
    print(f"Generating {num_sequences} synthetic sequences...")
    sequences = generate_one_hot_seq(seq_length, num_sequences)
    targets = generate_binary_targets(num_sequences, num_targets)
    
    # Save to HDF5
    print(f"Saving data to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset("sequences", data=sequences, compression=compression)
        f.create_dataset("targets", data=targets, compression=compression)
    
    print("Done!")

def create_train_val_test_data(
    output_dir: str,
    num_train: int = 10000,
    num_val: int = 2000,
    num_test: int = 2000,
    seq_length: int = 1000,
    num_targets: int = 919
) -> None:
    """
    Create train, validation, and test datasets.
    
    Args:
        output_dir: Directory to save the data files
        num_train: Number of training sequences
        num_val: Number of validation sequences
        num_test: Number of test sequences
        seq_length: Length of each sequence
        num_targets: Number of target features
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train shards (we'll make 2 shards for demonstration)
    train_size_per_shard = num_train // 2
    create_data_shard(
        os.path.join(output_dir, "train_shard1.h5"),
        train_size_per_shard,
        seq_length,
        num_targets
    )
    create_data_shard(
        os.path.join(output_dir, "train_shard2.h5"), 
        num_train - train_size_per_shard,
        seq_length,
        num_targets
    )
    
    # Create validation data
    create_data_shard(
        os.path.join(output_dir, "val.h5"),
        num_val,
        seq_length,
        num_targets
    )
    
    # Create test data
    create_data_shard(
        os.path.join(output_dir, "test.h5"),
        num_test,
        seq_length,
        num_targets
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic genomic data for training")
    parser.add_argument("--output-dir", type=str, default="/home/sdodl001/genomic_data",
                        help="Directory to save the generated data")
    parser.add_argument("--num-train", type=int, default=10000,
                        help="Number of training examples")
    parser.add_argument("--num-val", type=int, default=2000,
                        help="Number of validation examples")
    parser.add_argument("--num-test", type=int, default=2000,
                        help="Number of test examples")
    parser.add_argument("--seq-length", type=int, default=1000,
                        help="Length of each sequence")
    parser.add_argument("--num-targets", type=int, default=919,
                        help="Number of target features (919 for DeepSEA)")
    
    args = parser.parse_args()
    
    create_train_val_test_data(
        args.output_dir,
        args.num_train,
        args.num_val,
        args.num_test,
        args.seq_length,
        args.num_targets
    )
