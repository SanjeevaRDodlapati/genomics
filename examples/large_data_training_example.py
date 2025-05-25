"""
Example script for training models on large genomic data with interpretability.

This script demonstrates:
1. Using sharded genomic datasets for efficient memory management
2. Training models on large datasets with PyTorch Lightning
3. Applying interpretability techniques to analyze model predictions
4. Extracting and visualizing sequence motifs
"""

import os
import torch
import numpy as np
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from genomic_lightning.models.danq import DanQModel
from genomic_lightning.lightning_modules.danq import DanQLightningModule
from genomic_lightning.data.sharded_data_module import ShardedGenomicDataModule
from genomic_lightning.metrics.genomic_metrics import GenomicAUPRC, TopKAccuracy
from genomic_lightning.visualization.motif_visualization import MotifVisualizer


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and interpret a genomic model on large data")
    
    # Data arguments
    parser.add_argument("--train_shards", type=str, nargs="+", required=True,
                       help="Paths to training data shard files (HDF5)")
    parser.add_argument("--val_shards", type=str, nargs="+", required=True,
                       help="Paths to validation data shard files (HDF5)")
    parser.add_argument("--test_shards", type=str, nargs="+", default=None,
                       help="Paths to test data shard files (HDF5)")
    parser.add_argument("--x_dataset", type=str, default="sequences",
                       help="Name of input dataset in HDF5 files")
    parser.add_argument("--y_dataset", type=str, default="targets",
                       help="Name of target dataset in HDF5 files")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="danq", choices=["danq", "deepsea", "chromdragonn"],
                       help="Model architecture to use")
    parser.add_argument("--sequence_length", type=int, default=1000,
                       help="Length of input DNA sequences")
    parser.add_argument("--n_outputs", type=int, default=919,
                       help="Number of output predictions")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--cache_size", type=int, default=1000,
                       help="Number of samples to cache in memory")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory to save outputs")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Directory to save/load model checkpoints")
    
    # Visualization arguments
    parser.add_argument("--interpret", action="store_true",
                       help="Run interpretability analysis after training")
    parser.add_argument("--top_k_filters", type=int, default=20,
                       help="Number of top filters to visualize")
    
    return parser.parse_args()


def create_model(args):
    """Create model based on arguments."""
    if args.model_type == "danq":
        model = DanQModel(
            sequence_length=args.sequence_length,
            n_genomic_features=4,
            n_outputs=args.n_outputs,
        )
    elif args.model_type == "deepsea":
        from genomic_lightning.models.deepsea import DeepSEA
        model = DeepSEA(
            sequence_length=args.sequence_length,
            n_genomic_features=4,
            n_outputs=args.n_outputs,
        )
    elif args.model_type == "chromdragonn":
        from genomic_lightning.models.chromdragonn import ChromDragoNNModel
        model = ChromDragoNNModel(
            sequence_length=args.sequence_length,
            n_genomic_features=4,
            n_outputs=args.n_outputs,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    return model


def create_lightning_module(args, model=None):
    """Create Lightning module based on arguments."""
    if model is None:
        model = create_model(args)
        
    # Define metrics
    metrics = [
        "auroc",  # Use default AUROC from torchmetrics
        GenomicAUPRC(num_classes=args.n_outputs),
        TopKAccuracy(k=5)
    ]
    
    # Create module based on model type
    if args.model_type == "danq":
        return DanQLightningModule(
            model=model,
            learning_rate=args.learning_rate,
            metrics=metrics,
            prediction_output_dir=args.output_dir
        )
    elif args.model_type == "deepsea":
        from genomic_lightning.lightning_modules.deepsea import DeepSEAModule
        return DeepSEAModule(
            model=model,
            learning_rate=args.learning_rate,
            metrics=metrics,
            prediction_output_dir=args.output_dir
        )
    elif args.model_type == "chromdragonn":
        from genomic_lightning.lightning_modules.chromdragonn import ChromDragoNNLightningModule
        return ChromDragoNNLightningModule(
            model=model,
            learning_rate=args.learning_rate,
            metrics=metrics,
            prediction_output_dir=args.output_dir
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def run_training(args):
    """Train the model using PyTorch Lightning."""
    pl.seed_everything(42)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create data module
    data_module = ShardedGenomicDataModule(
        train_shards=args.train_shards,
        val_shards=args.val_shards,
        test_shards=args.test_shards,
        x_dset=args.x_dataset,
        y_dset=args.y_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
    )
    
    # Create Lightning module
    module = create_lightning_module(args)
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{args.model_type}-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        )
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        accelerator="auto",  # Automatically choose GPU if available
        devices="auto",      # Use all available GPUs
    )
    
    # Train model
    trainer.fit(module, datamodule=data_module)
    
    # Test if test data is available
    if args.test_shards:
        trainer.test(datamodule=data_module)
    
    return module, trainer


def run_interpretation(args, module):
    """Run interpretability analysis on trained model."""
    print("\nRunning interpretability analysis...")
    
    # Create visualization directory
    viz_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create motif visualizer
    model = module.model
    device = next(model.parameters()).device
    visualizer = MotifVisualizer(model, device=device)
    
    # Extract and save filter motifs
    motif_dir = os.path.join(viz_dir, "motifs")
    os.makedirs(motif_dir, exist_ok=True)
    
    print(f"Extracting motifs from model filters...")
    visualizer.save_filter_logos(
        model,
        layer_idx=0,  # First convolutional layer
        output_dir=motif_dir,
        top_k_filters=args.top_k_filters
    )
    print(f"Saved motif visualizations to {motif_dir}")


def main():
    """Main function."""
    args = parse_arguments()
    print(f"Training {args.model_type} model on sharded genomic data")
    print(f"Training shards: {len(args.train_shards)}, Validation shards: {len(args.val_shards)}")
    
    # Run training
    module, trainer = run_training(args)
    
    # Run interpretability analysis if requested
    if args.interpret:
        run_interpretation(args, module)
        
    print("\nDone!")


if __name__ == "__main__":
    main()
