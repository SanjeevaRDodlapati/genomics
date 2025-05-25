#!/usr/bin/env python
"""
Example script demonstrating integration between GenomicLightning and UAVarPrior.

This example shows how to:
1. Import models from UAVarPrior
2. Convert models to GenomicLightning format
3. Use UAVarPrior samplers with GenomicLightning
4. Train and evaluate imported models
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Add parent directory to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add UAVarPrior to path if needed
uavarprior_path = "/home/sdodl001/UAVarPrior"
if os.path.exists(uavarprior_path) and uavarprior_path not in sys.path:
    sys.path.insert(0, uavarprior_path)

from genomic_lightning.utils.legacy_import import import_uavarprior_model
from genomic_lightning.utils.wrapper_conversion import wrap_model_with_lightning
from genomic_lightning.utils.sampler_utils import create_dataloader_from_legacy_sampler


def import_uavarprior_sampler(config_file):
    """
    Import a sampler from UAVarPrior.
    
    Args:
        config_file: Path to UAVarPrior configuration file
        
    Returns:
        UAVarPrior sampler object
    """
    try:
        # Import UAVarPrior modules
        from uavarprior.config import load_config
        from uavarprior.data.samplers import get_sampler
        
        # Load configuration
        config = load_config(config_file)
        
        # Create samplers
        train_sampler = get_sampler(config, 'train')
        val_sampler = get_sampler(config, 'validation')
        test_sampler = get_sampler(config, 'test')
        
        return train_sampler, val_sampler, test_sampler
        
    except ImportError as e:
        print(f"Error importing UAVarPrior modules: {str(e)}")
        print("Make sure UAVarPrior is installed and accessible.")
        sys.exit(1)


def main(args):
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Import model from UAVarPrior
    print(f"Importing model from {args.model_path}...")
    model = import_uavarprior_model(
        model_path=args.model_path,
        model_type=args.model_type,
        config_path=args.config_path
    )
    
    print(f"Model successfully imported: {model.__class__.__name__}")
    
    # Wrap model with Lightning module
    lightning_module = wrap_model_with_lightning(
        model=model,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        loss_function=args.loss_function,
        metrics=["auroc", "auprc", "accuracy"]
    )
    
    # Import UAVarPrior samplers if config provided
    if args.sampler_config:
        print(f"Importing UAVarPrior samplers from {args.sampler_config}...")
        train_sampler, val_sampler, test_sampler = import_uavarprior_sampler(args.sampler_config)
        
        # Create data loaders from UAVarPrior samplers
        train_loader = create_dataloader_from_legacy_sampler(
            sampler=train_sampler,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        val_loader = create_dataloader_from_legacy_sampler(
            sampler=val_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        test_loader = create_dataloader_from_legacy_sampler(
            sampler=test_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        print(f"Created data loaders from UAVarPrior samplers")
    else:
        print("No sampler config provided. Using synthetic data for demonstration...")
        
        # Create synthetic data
        from genomic_lightning.models.danq import DanQModel  # For getting default shapes
        
        # Try to infer input shape from the model
        if hasattr(model, 'conv_layer'):
            in_channels = model.conv_layer.in_channels
            seq_length = 1000  # Default
        else:
            in_channels = 4  # Default for genomic data
            seq_length = 1000
            
        # Try to infer output shape from the model
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
            num_targets = model.classifier.out_features
        else:
            num_targets = 919  # Default for DeepSEA
        
        # Generate synthetic data
        from torch.utils.data import TensorDataset, DataLoader, random_split
        import numpy as np
        
        # Create random sequences
        sequences = np.zeros((1000, in_channels, seq_length), dtype=np.float32)
        for i in range(1000):
            seq = np.random.randint(0, 4, seq_length)
            for j in range(seq_length):
                sequences[i, seq[j], j] = 1.0
                
        # Create random targets
        targets = np.random.randint(0, 2, (1000, num_targets)).astype(np.float32)
        
        # Convert to tensors
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        # Create dataset and split
        dataset = TensorDataset(sequences_tensor, targets_tensor)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename=f'{args.model_type}-{{epoch:02d}}-{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    callbacks.append(early_stopping)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator='auto',  # Use GPU if available
        devices=args.devices,
        precision=args.precision,
        default_root_dir=args.output_dir,
        log_every_n_steps=10
    )
    
    if args.train:
        # Train model
        print("Training model...")
        trainer.fit(lightning_module, train_loader, val_loader)
        
        # Test model
        print("Testing model...")
        test_results = trainer.test(lightning_module, test_loader)
        print(f"Test results: {test_results}")
        
        # Save model
        model_dir = os.path.join(args.output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save in PyTorch Lightning format
        trainer.save_checkpoint(os.path.join(model_dir, f'{args.model_type}_model.ckpt'))
        
        # Save in PyTorch format
        torch.save(model.state_dict(), os.path.join(model_dir, f'{args.model_type}_model.pt'))
        
        print(f"Model saved in {model_dir}")
    else:
        # Just evaluate the model
        print("Evaluating model...")
        test_results = trainer.test(lightning_module, test_loader)
        print(f"Test results: {test_results}")
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import and use UAVarPrior models with GenomicLightning")
    
    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to UAVarPrior model weights (.pth file)")
    
    # Optional arguments
    parser.add_argument("--model-type", type=str, default="deepsea",
                        choices=["deepsea", "danq", "chromdragonn", "custom"],
                        help="Type of model architecture")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to model configuration file (optional)")
    parser.add_argument("--sampler-config", type=str, default=None,
                        help="Path to UAVarPrior sampler configuration file (optional)")
    parser.add_argument("--train", action="store_true",
                        help="Train the model after importing")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="Maximum number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate for optimization")
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
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="uavarprior_integration",
                        help="Output directory for models and logs")
    
    args = parser.parse_args()
    
    main(args)