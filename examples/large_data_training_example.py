#!/usr/bin/env python
"""
Example script for training a genomic model with large-scale data using GenomicLightning.

This example demonstrates how to:
1. Use sharded data for efficient large-scale data handling
2. Train a genomic model with large datasets
3. Use advanced PyTorch Lightning features for distributed training
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import h5py
import numpy as np
from pathlib import Path

# Add parent directory to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic_lightning.models.chromdragonn import ChromDragoNNModel
from genomic_lightning.lightning_modules.chromdragonn import ChromDragoNNLightningModule
from genomic_lightning.data.sharded_data_module import ShardedGenomicDataModule


def generate_synthetic_sharded_data(
    output_dir,
    num_shards=10,
    samples_per_shard=1000,
    seq_length=1000,
    num_targets=919
):
    """Generate synthetic sharded genomic data for demonstration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sharded datasets
    for shard_idx in range(num_shards):
        output_file = os.path.join(output_dir, f"shard_{shard_idx:05d}.h5")
        
        # Create random one-hot encoded sequences
        sequences = np.zeros((samples_per_shard, 4, seq_length), dtype=np.float32)
        
        for i in range(samples_per_shard):
            # Generate a random sequence of ACGT
            seq = np.random.randint(0, 4, seq_length)
            # Convert to one-hot encoding
            for j in range(seq_length):
                sequences[i, seq[j], j] = 1.0
        
        # Create random binary targets
        targets = np.random.randint(0, 2, (samples_per_shard, num_targets)).astype(np.float32)
        
        # Save to HDF5 file
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('sequences', data=sequences)
            f.create_dataset('targets', data=targets)
            
            # Add metadata
            f.attrs['shard_idx'] = shard_idx
            f.attrs['num_samples'] = samples_per_shard
            f.attrs['seq_length'] = seq_length
            f.attrs['num_targets'] = num_targets
    
    # Create a split file
    splits = {
        'train': list(range(0, int(num_shards * 0.7))),
        'val': list(range(int(num_shards * 0.7), int(num_shards * 0.85))),
        'test': list(range(int(num_shards * 0.85), num_shards))
    }
    
    split_file = os.path.join(output_dir, 'splits.txt')
    with open(split_file, 'w') as f:
        for split, indices in splits.items():
            shard_files = [f"shard_{idx:05d}.h5" for idx in indices]
            f.write(f"{split}:{','.join(shard_files)}\n")
    
    return output_dir, split_file


def main(args):
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate synthetic data if needed
    if args.generate_data:
        print("Generating synthetic sharded data...")
        data_dir, split_file = generate_synthetic_sharded_data(
            os.path.join(args.output_dir, 'data'),
            num_shards=args.num_shards,
            samples_per_shard=args.samples_per_shard,
            seq_length=args.seq_length,
            num_targets=args.num_targets
        )
    else:
        data_dir = args.data_dir
        split_file = args.split_file
    
    # Create data module
    data_module = ShardedGenomicDataModule(
        data_dir=data_dir,
        split_file=split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_dataset='sequences',
        target_dataset='targets',
        shuffle=True
    )
    
    # Create model
    print("Creating model...")
    model = ChromDragoNNModel(
        num_targets=args.num_targets,
        num_filters=args.num_filters,
        filter_sizes=args.filter_sizes,
        residual_blocks=args.residual_blocks
    )
    
    # Create Lightning module
    lightning_module = ChromDragoNNLightningModule(
        model=model,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        loss_function=args.loss_function,
        metrics=["auroc", "auprc"]
    )
    
    # Set up callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='chromdragonn-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='chromdragonn'
    )
    
    # Create trainer with distributed training options
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',  # Use GPU if available
        devices=args.devices,
        precision=args.precision,
        strategy=args.strategy if args.devices > 1 else None,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        deterministic=args.deterministic
    )
    
    # Train model
    print("Training model...")
    trainer.fit(lightning_module, data_module)
    
    # Test model
    print("Testing model...")
    test_results = trainer.test(lightning_module, datamodule=data_module)
    print(f"Test results: {test_results}")
    
    # Save model
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save in PyTorch Lightning format
    trainer.save_checkpoint(os.path.join(model_dir, 'chromdragonn_model.ckpt'))
    
    # Optionally, save in PyTorch format
    torch.save(model.state_dict(), os.path.join(model_dir, 'chromdragonn_model.pt'))
    
    print(f"Model saved in {model_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChromDragoNN model on large genomic datasets")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with sharded data files")
    parser.add_argument("--split-file", type=str, default=None,
                        help="File defining train/val/test splits")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate synthetic data for demonstration")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Number of shards to generate")
    parser.add_argument("--samples-per-shard", type=int, default=1000,
                        help="Number of samples per shard")
    parser.add_argument("--seq-length", type=int, default=1000,
                        help="Length of input sequences")
    parser.add_argument("--num-targets", type=int, default=919,
                        help="Number of prediction targets")
    
    # Model parameters
    parser.add_argument("--num-filters", type=int, default=300,
                        help="Number of convolutional filters")
    parser.add_argument("--filter-sizes", type=str, default="10,15,20",
                        help="Comma-separated list of filter sizes")
    parser.add_argument("--residual-blocks", type=int, default=3,
                        help="Number of residual blocks")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for optimization")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for regularization")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"],
                        help="Optimizer for training")
    parser.add_argument("--loss-function", type=str, default="binary_cross_entropy",
                        choices=["binary_cross_entropy", "bce_with_logits", "mse"],
                        help="Loss function for training")
    
    # Hardware/performance parameters
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices (GPUs) to use")
    parser.add_argument("--precision", type=int, default=32,
                        choices=[16, 32],
                        help="Precision for training (16 or 32)")
    parser.add_argument("--strategy", type=str, default="ddp",
                        choices=["ddp", "ddp_spawn", "dp", "horovod"],
                        help="Distributed training strategy")
    parser.add_argument("--gradient-clip-val", type=float, default=0.0,
                        help="Gradient clipping value (0 for no clipping)")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1,
                        help="Number of batches to accumulate gradients")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic algorithms for reproducibility")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for models and logs")
    
    args = parser.parse_args()
    
    # Parse filter sizes from string to list
    try:
        args.filter_sizes = [int(x) for x in args.filter_sizes.split(",")]
    except ValueError:
        parser.error("Filter sizes must be comma-separated integers")
    
    # Validate arguments
    if not args.generate_data and (args.data_dir is None or args.split_file is None):
        parser.error("Either --generate-data or both --data-dir and --split-file must be provided")
    
    main(args)