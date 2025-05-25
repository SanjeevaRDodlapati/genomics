"""Command-line interface for GenomicLightning."""

import os
import click
import pytorch_lightning as pl
import torch
from typing import Dict, Any, Optional, List, Tuple

from genomic_lightning.config.loader import load_config
from genomic_lightning.config.factory import create_lightning_module, create_data_module


@click.group()
def cli():
    """GenomicLightning - PyTorch Lightning framework for genomic deep learning models."""
    pass


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--gpus', type=int, default=None, help='Number of GPUs to use')
@click.option('--precision', type=int, default=None, help='Precision for training (16, 32)')
@click.option('--max-epochs', type=int, default=None, help='Maximum number of epochs')
@click.option('--seed', type=int, default=None, help='Random seed')
def train(config_path, gpus, precision, max_epochs, seed):
    """Train a model using the specified configuration."""
    # Set random seed if provided
    if seed is not None:
        pl.seed_everything(seed)
    
    # Load configuration
    config = load_config(config_path)
    
    # Override config with CLI options
    trainer_config = config.get('trainer', {})
    if gpus is not None:
        trainer_config['gpus'] = gpus
    if precision is not None:
        trainer_config['precision'] = precision
    if max_epochs is not None:
        trainer_config['max_epochs'] = max_epochs
    config['trainer'] = trainer_config
    
    # Create Lightning module and data module
    model = create_lightning_module(config)
    data_module = create_data_module(config)
    
    # Configure trainer
    trainer_args = trainer_config.copy()
    callbacks = _get_callbacks(config)
    if callbacks:
        trainer_args['callbacks'] = callbacks
    
    logger = _get_logger(config)
    if logger:
        trainer_args['logger'] = logger
    
    # Create trainer
    trainer = pl.Trainer(**trainer_args)
    
    # Start training
    trainer.fit(model, data_module)
    
    # Run test if requested
    if config.get('test_after_train', False):
        trainer.test(model, data_module)
    
    # Return for potential further use
    return {
        'trainer': trainer,
        'model': model,
        'data_module': data_module,
        'config': config
    }


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--ckpt', type=click.Path(exists=True), help='Path to checkpoint file')
@click.option('--output-dir', type=click.Path(), help='Directory to save outputs')
def evaluate(config_path, ckpt, output_dir):
    """Evaluate a trained model."""
    # Load configuration
    config = load_config(config_path)
    
    # Override output directory if provided
    if output_dir:
        config['output_dir'] = output_dir
    
    # Ensure output directory exists
    os.makedirs(config.get('output_dir', 'outputs'), exist_ok=True)
    
    # Create Lightning module and data module
    model = create_lightning_module(config)
    data_module = create_data_module(config)
    
    # Configure trainer
    trainer_args = config.get('trainer', {}).copy()
    trainer_args['gpus'] = trainer_args.get('gpus', 1)  # Default to 1 GPU
    
    # Create trainer
    trainer = pl.Trainer(**trainer_args)
    
    # Run test
    trainer.test(model=model, datamodule=data_module, ckpt_path=ckpt)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--ckpt', type=click.Path(exists=True), help='Path to checkpoint file')
@click.option('--output-dir', type=click.Path(), help='Directory to save predictions')
def predict(config_path, ckpt, output_dir):
    """Generate predictions using a trained model."""
    # Load configuration
    config = load_config(config_path)
    
    # Override output directory if provided
    if output_dir:
        config['output_dir'] = output_dir
    
    # Ensure output directory exists
    os.makedirs(config.get('output_dir', 'predictions'), exist_ok=True)
    
    # Create Lightning module and data module
    model = create_lightning_module(config)
    data_module = create_data_module(config)
    
    # Configure trainer
    trainer_args = config.get('trainer', {}).copy()
    trainer_args['gpus'] = trainer_args.get('gpus', 1)  # Default to 1 GPU
    
    # Create trainer
    trainer = pl.Trainer(**trainer_args)
    
    # Run predictions
    predictions = trainer.predict(model=model, datamodule=data_module, ckpt_path=ckpt)
    
    # Save predictions (specific saving logic would depend on your needs)
    from genomic_lightning.utils.prediction_utils import save_predictions
    save_predictions(predictions, config.get('output_dir', 'predictions'), config)


def _get_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    """Create callbacks based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping', {}).get('enabled', True):
        es_config = config.get('early_stopping', {})
        es_callback = pl.callbacks.EarlyStopping(
            monitor=es_config.get('monitor', 'val_loss'),
            patience=es_config.get('patience', 10),
            mode=es_config.get('mode', 'min'),
            verbose=es_config.get('verbose', True),
        )
        callbacks.append(es_callback)
    
    # Model checkpoint
    if config.get('model_checkpoint', {}).get('enabled', True):
        ckpt_config = config.get('model_checkpoint', {})
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            monitor=ckpt_config.get('monitor', 'val_loss'),
            mode=ckpt_config.get('mode', 'min'),
            save_top_k=ckpt_config.get('save_top_k', 1),
            save_last=ckpt_config.get('save_last', True),
            filename=ckpt_config.get('filename', '{epoch}-{val_loss:.4f}'),
            dirpath=os.path.join(
                config.get('output_dir', 'outputs'),
                'checkpoints'
            ),
            verbose=ckpt_config.get('verbose', True),
        )
        callbacks.append(ckpt_callback)
    
    # Learning rate monitor
    if config.get('lr_monitor', {}).get('enabled', True):
        lr_config = config.get('lr_monitor', {})
        lr_callback = pl.callbacks.LearningRateMonitor(
            logging_interval=lr_config.get('logging_interval', 'epoch'),
            log_momentum=lr_config.get('log_momentum', False),
        )
        callbacks.append(lr_callback)
    
    # Add any custom callbacks
    # (This would need to be implemented based on your callback registry)
    
    return callbacks


def _get_logger(config: Dict[str, Any]) -> Optional[pl.loggers.Logger]:
    """Create logger based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Logger or None if not configured
    """
    logger_config = config.get('logger', {})
    if not logger_config.get('enabled', True):
        return None
    
    logger_type = logger_config.get('type', 'tensorboard')
    
    if logger_type.lower() == 'tensorboard':
        return pl.loggers.TensorBoardLogger(
            save_dir=logger_config.get('save_dir', os.path.join(
                config.get('output_dir', 'outputs'),
                'logs'
            )),
            name=logger_config.get('name', 'lightning_logs'),
            version=logger_config.get('version', None),
            default_hp_metric=logger_config.get('default_hp_metric', True),
        )
    
    elif logger_type.lower() == 'wandb':
        # Check if wandb is installed
        try:
            import wandb
        except ImportError:
            click.echo(
                "Warning: WandB logger requested but wandb package not installed. "
                "Install with 'pip install wandb'."
            )
            return None
        
        return pl.loggers.WandbLogger(
            project=logger_config.get('project', 'genomic-lightning'),
            name=logger_config.get('name', None),
            save_dir=logger_config.get('save_dir', os.path.join(
                config.get('output_dir', 'outputs'),
                'logs'
            )),
            log_model=logger_config.get('log_model', False),
        )
    
    elif logger_type.lower() == 'csv':
        return pl.loggers.CSVLogger(
            save_dir=logger_config.get('save_dir', os.path.join(
                config.get('output_dir', 'outputs'),
                'logs'
            )),
            name=logger_config.get('name', 'lightning_logs'),
            version=logger_config.get('version', None),
        )
    
    else:
        click.echo(f"Warning: Unknown logger type '{logger_type}'. Using TensorBoard.")
        return pl.loggers.TensorBoardLogger(
            save_dir=os.path.join(config.get('output_dir', 'outputs'), 'logs'),
            name='lightning_logs',
        )
