#!/usr/bin/env python
"""
Example script demonstrating how to use GenomicLightning.

This script shows how to train a DeepSEA model using GenomicLightning,
with synthetic data for demonstration purposes.
"""

import os
import numpy as np
import h5py
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from genomic_lightning.models.deepsea import DeepSEA
from genomic_lightning.lightning_modules.deepsea import DeepSEAModule
from genomic_lightning.data.data_modules import H5DataModule


def generate_synthetic_data(output_dir, n_samples=1000, seq_length=1000, n_targets=919):
    """Generate synthetic data for demonstration.
    
    Args:
        output_dir: Directory to save synthetic data
        n_samples: Number of samples to generate
        seq_length: Sequence length
        n_targets: Number of targets
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random one-hot encoded sequences
    sequences = np.zeros((n_samples, 4, seq_length), dtype=np.float32)
    for i in range(n_samples):
        # Generate random sequence (one-hot encoded)
        hot_indices = np.random.randint(0, 4, size=seq_length)
        for j, h in enumerate(hot_indices):
            sequences[i, h, j] = 1.0
    
    # Generate random targets (binary)
    targets = np.random.randint(0, 2, size=(n_samples, n_targets)).astype(np.float32)
    
    # Save sequences and targets as H5 files
    sequence_file = os.path.join(output_dir, 'synthetic_sequences.h5')
    target_file = os.path.join(output_dir, 'synthetic_targets.h5')
    
    with h5py.File(sequence_file, 'w') as f:
        f.create_dataset('sequences', data=sequences, compression='gzip')
    
    with h5py.File(target_file, 'w') as f:
        f.create_dataset('targets', data=targets, compression='gzip')
    
    return sequence_file, target_file


def main():
    """Run the example."""
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Generate synthetic data
    output_dir = os.path.join(os.path.dirname(__file__), 'synthetic_data')
    sequence_file, target_file = generate_synthetic_data(output_dir)
    
    # Create model
    model = DeepSEA(
        sequence_length=1000,
        n_targets=919,
        conv_kernel_sizes=[8, 8, 8],
        conv_channels=[320, 480, 960],
        pool_kernel_sizes=[4, 4, 4],
        dropout_rate=0.2
    )
    
    # Create Lightning module
    lightning_module = DeepSEAModule(
        model=model,
        learning_rate=0.0002,
        weight_decay=1e-6,
        optimizer='adam',
        scheduler='reduce_on_plateau'
    )
    
    # Create data module
    data_module = H5DataModule(
        sequence_file=sequence_file,
        target_file=target_file,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        split_ratio=(0.8, 0.1, 0.1)
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            filename='{epoch:02d}-{val_loss:.4f}'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='deepsea_example'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',  # Use GPU if available
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision if GPU available
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(lightning_module, data_module)
    
    # Test the model
    trainer.test(lightning_module, data_module)
    
    print(f"Training complete! Logs saved to {logger.log_dir}")


if __name__ == '__main__':
    main()
