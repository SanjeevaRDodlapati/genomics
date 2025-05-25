#!/usr/bin/env python
"""
Example script for training a simple genomic model with GenomicLightning.

This example demonstrates how to:
1. Create a genomic model
2. Set up a Lightning module
3. Prepare data for training
4. Train the model using PyTorch Lightning
"""

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add parent directory to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic_lightning.models.danq import DanQModel
from genomic_lightning.lightning_modules.danq import DanQLightningModule


def generate_synthetic_data(num_samples=1000, seq_length=1000, num_targets=5):
    """Generate synthetic genomic data for demonstration."""
    # Create random one-hot encoded sequences
    sequences = np.zeros((num_samples, 4, seq_length), dtype=np.float32)
    
    for i in range(num_samples):
        # Generate a random sequence of ACGT
        seq = np.random.randint(0, 4, seq_length)
        # Convert to one-hot encoding
        for j in range(seq_length):
            sequences[i, seq[j], j] = 1.0
    
    # Create random binary targets
    targets = np.random.randint(0, 2, (num_samples, num_targets)).astype(np.float32)
    
    return torch.tensor(sequences), torch.tensor(targets)


def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Parameters
    seq_length = 1000  # Length of input DNA sequences
    num_targets = 5    # Number of binary prediction targets
    batch_size = 32    # Batch size for training
    max_epochs = 10    # Maximum number of training epochs
    
    # Generate synthetic data
    print("Generating synthetic data...")
    sequences, targets = generate_synthetic_data(
        num_samples=1000, 
        seq_length=seq_length, 
        num_targets=num_targets
    )
    
    # Create dataset and split into train/validation/test
    dataset = TensorDataset(sequences, targets)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    print("Creating model...")
    model = DanQModel(
        num_targets=num_targets,
        num_filters=32,      # Reduced for demo
        filter_size=26,
        pool_size=13,
        lstm_hidden=32,      # Reduced for demo
        lstm_layers=1,
        dropout_rate=0.2
    )
    
    # Create Lightning module
    lightning_module = DanQLightningModule(
        model=model,
        learning_rate=1e-3,
        optimizer="adam",
        loss_function="binary_cross_entropy",
        metrics=["auroc", "auprc"]
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='danq-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',  # Use GPU if available
        precision=32,
        default_root_dir='outputs'
    )
    
    # Train model
    print("Training model...")
    trainer.fit(lightning_module, train_loader, val_loader)
    
    # Test model
    print("Testing model...")
    test_results = trainer.test(lightning_module, test_loader)
    print(f"Test results: {test_results}")
    
    # Save model
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in PyTorch Lightning format
    trainer.save_checkpoint(os.path.join(output_dir, 'danq_model.ckpt'))
    
    # Optionally, save in PyTorch format
    torch.save(model.state_dict(), os.path.join(output_dir, 'danq_model.pt'))
    
    print(f"Model saved in {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()