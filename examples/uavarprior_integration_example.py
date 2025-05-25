#!/usr/bin/env python
"""
Example script demonstrating how to use an existing UAVarPrior model with GenomicLightning.

This script shows how to take a model from UAVarPrior and use it within
the GenomicLightning framework.
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Ensure UAVarPrior is in the path
sys.path.append('/home/sdodl001/UAVarPrior')

# Import GenomicLightning components
from genomic_lightning.lightning_modules.deepsea import DeepSEAModule
from genomic_lightning.data.sampler_adapter import SamplerDatasetAdapter
from genomic_lightning.data.sampler_data_module import SamplerDataModule
from genomic_lightning.utils.legacy_import import import_uavarprior_model


def main():
    """Run the example of using UAVarPrior model in GenomicLightning."""
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Import UAVarPrior's initialize_model function
    try:
        initialize_model = import_uavarprior_model()
    except ImportError as e:
        print(f"Error importing from UAVarPrior: {e}")
        print("Make sure UAVarPrior is installed and in your PYTHONPATH.")
        sys.exit(1)
    
    # Configure model
    model_config = {
        'name': 'DeepSEA',
        'class': 'DeepSEA',
        'classArgs': {
            'sequence_length': 1000,
            'n_targets': 919
        },
        'built': 'pytorch'
    }
    
    # Initialize model using UAVarPrior's function
    try:
        model = initialize_model(model_config, train=True)
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)
    
    # Create Lightning module
    lightning_module = DeepSEAModule(
        model=model,
        learning_rate=0.0002,
        weight_decay=1e-6,
        optimizer='adam',
        scheduler='reduce_on_plateau'
    )
    
    # Try to import a sampler from UAVarPrior
    try:
        from uavarprior.samplers import IntervalsSampler
        
        # This is just a placeholder - you would need actual paths to your data
        sampler = IntervalsSampler(
            input_path='/path/to/data/intervals.bed',
            genome='hg38',
            target_path='/path/to/data/targets.h5'
        )
    except ImportError:
        print("Could not import IntervalsSampler. Using a dummy sampler for demonstration.")
        # Define a dummy sampler class for demonstration
        class DummySampler:
            def __init__(self):
                pass
                
            def set_mode(self, mode):
                self.mode = mode
                
            def get_dataset_size(self):
                return 1000
                
            def sample(self):
                # Generate a random sample
                import numpy as np
                sequence = np.random.rand(4, 1000).astype(np.float32)
                targets = np.random.randint(0, 2, size=919).astype(np.float32)
                return {'sequence': sequence, 'targets': targets}
        
        sampler = DummySampler()
    
    # Create data module
    data_module = SamplerDataModule(
        sampler=sampler,
        batch_size=32,
        num_workers=4,
        max_train_samples=1000,
        max_val_samples=200,
        max_test_samples=200
    )
    
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'uavarprior_integration')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            filename='{epoch:02d}-{val_loss:.4f}',
            dirpath=os.path.join(output_dir, 'checkpoints')
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='uavarprior_integration'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=5,  # Just a few epochs for demonstration
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',  # Use GPU if available
        devices=1,
        precision=32,  # Use 32-bit precision for stability in this example
        log_every_n_steps=10,
        # For demonstration, limit validation and test batches
        limit_val_batches=10,
        limit_test_batches=10
    )
    
    # Train the model
    trainer.fit(lightning_module, data_module)
    
    # Test the model
    trainer.test(lightning_module, data_module)
    
    print(f"Training complete! Logs saved to {logger.log_dir}")


if __name__ == '__main__':
    main()
