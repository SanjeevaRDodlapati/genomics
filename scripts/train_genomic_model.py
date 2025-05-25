#!/usr/bin/env python
"""
Train a small genomic deep learning model using GenomicLightning.

This script trains either a DeepSEA or DanQ model on synthetic or real
genomic data using PyTorch Lightning.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

from genomic_lightning.models.deepsea import DeepSEAModel
from genomic_lightning.models.danq import DanQModel
from genomic_lightning.lightning_modules.base import BaseLightningModule
from genomic_lightning.data.sharded_data_module import ShardedGenomicDataModule

def train_model(args):
    """
    Train a genomic deep learning model.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    if args.model_type.lower() == "deepsea":
        model = DeepSEAModel(
            num_targets=args.num_targets,
            num_filters=[320, 480, 960],
            filter_sizes=[8, 8, 8],
            pool_sizes=[4, 4, 4],
            dropout_rates=[0.2, 0.2, 0.5]
        )
        print(f"Created DeepSEA model with {args.num_targets} targets")
    elif args.model_type.lower() == "danq":
        model = DanQModel(
            num_targets=args.num_targets,
            num_filters=320,
            filter_size=26,
            pool_size=13,
            lstm_hidden=320,
            lstm_layers=1
        )
        print(f"Created DanQ model with {args.num_targets} targets")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Wrap model with Lightning module
    lightning_module = BaseLightningModule(
        model=model,
        learning_rate=args.learning_rate,
        optimizer="adam",
        loss_function="bce_with_logits",
        metrics=["auroc", "auprc", "accuracy"]
    )
    
    # Get data file paths
    train_shards = [
        os.path.join(args.data_dir, "train_shard1.h5"),
        os.path.join(args.data_dir, "train_shard2.h5")
    ]
    val_shard = [os.path.join(args.data_dir, "val.h5")]
    test_shard = [os.path.join(args.data_dir, "test.h5")]
    
    # Create data module
    data_module = ShardedGenomicDataModule(
        train_shards=train_shards,
        val_shards=val_shard,
        test_shards=test_shard,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_size=args.cache_size
    )
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename=f"{args.model_type}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min"
    )
    callbacks.append(early_stopping)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="logs"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",  # Use GPU if available
        devices=args.devices if args.devices > 0 else 'auto',
        precision=args.precision,
        default_root_dir=args.output_dir,
        log_every_n_steps=10
    )
    
    # Train model
    print(f"Starting {args.model_type} model training...")
    trainer.fit(lightning_module, data_module)
    
    # Test model
    print("Evaluating model on test data...")
    test_results = trainer.test(lightning_module, datamodule=data_module)
    print(f"Test results: {test_results}")
    
    # Save model
    output_path = os.path.join(args.output_dir, f"{args.model_type}_model.pt")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")
    
    # Save Lightning checkpoint
    lightning_path = os.path.join(args.output_dir, f"{args.model_type}_lightning.ckpt")
    trainer.save_checkpoint(lightning_path)
    print(f"Lightning checkpoint saved to {lightning_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a genomic deep learning model")
    
    # Model parameters
    parser.add_argument("--model-type", type=str, default="deepsea",
                        choices=["deepsea", "danq"],
                        help="Type of model to train")
    parser.add_argument("--num-targets", type=int, default=919,
                        help="Number of output targets")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default="/home/sdodl001/genomic_data",
                        help="Directory with training data")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--cache-size", type=int, default=1000,
                        help="Number of samples to cache in memory")
    
    # Training parameters
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="Learning rate for optimizer")
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping")
    
    # Hardware parameters
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs to use (0 for auto)")
    parser.add_argument("--precision", type=int, default=32,
                        choices=[16, 32],
                        help="Floating point precision")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="/home/sdodl001/genomic_results",
                        help="Directory to save model and logs")
    
    args = parser.parse_args()
    
    train_model(args)
