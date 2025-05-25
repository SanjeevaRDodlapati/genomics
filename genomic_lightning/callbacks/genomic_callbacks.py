"""Specialized callbacks for genomic deep learning models."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch


class GenomicVisualizationCallback(Callback):
    """Callback for visualizing genomic model predictions and patterns.
    
    This callback generates visualizations of predictions and learned patterns
    during training, which can help with model interpretation.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        num_examples: int = 10,
        plot_frequency: int = 10,
    ):
        """Initialize the genomic visualization callback.
        
        Args:
            output_dir: Directory to save visualizations
            num_examples: Number of examples to visualize
            plot_frequency: Plot every N epochs
        """
        super().__init__()
        self.output_dir = output_dir
        self.num_examples = num_examples
        self.plot_frequency = plot_frequency
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Generate visualizations at the end of validation epochs.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        # Only generate visualizations every plot_frequency epochs
        if trainer.current_epoch % self.plot_frequency != 0:
            return
        
        # Ensure output directory exists
        output_dir = self.output_dir or os.path.join(
            trainer.logger.log_dir, "visualizations"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        try:
            self._visualize_filters(trainer, pl_module, output_dir)
        except Exception as e:
            print(f"Error visualizing filters: {str(e)}")
    
    def _visualize_filters(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output_dir: str
    ) -> None:
        """Visualize first layer filters of the model.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            output_dir: Directory to save visualizations
        """
        # Get model from pl_module
        model = pl_module.model
        
        # Try to extract first convolutional layer
        first_conv = None
        if hasattr(model, 'conv_layers') and len(model.conv_layers) > 0:
            first_conv = model.conv_layers[0]
        else:
            # Search for the first convolutional layer
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv1d):
                    first_conv = module
                    break
        
        if first_conv is None:
            return
        
        # Get weights and visualize them
        weights = first_conv.weight.detach().cpu().numpy()
        
        # Plot filters
        num_filters = min(16, weights.shape[0])  # Show at most 16 filters
        fig, axes = plt.subplots(num_filters, 1, figsize=(10, num_filters * 2))
        
        # If there's only one filter, axes won't be an array
        if num_filters == 1:
            axes = [axes]
        
        for i in range(num_filters):
            # Plot motif as a heatmap
            im = axes[i].imshow(weights[i], aspect='auto', cmap='viridis')
            axes[i].set_title(f'Filter {i+1}')
            axes[i].set_ylabel('Channel')
            axes[i].set_xlabel('Position')
            fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(
            output_dir, f'filters_epoch_{trainer.current_epoch:03d}.png'
        )
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()


class GenomicPredictionSamplingCallback(Callback):
    """Callback for sampling and logging predictions during training.
    
    This callback periodically samples predictions from the validation set
    and logs them for analysis.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        sampling_frequency: int = 5,
        save_dir: Optional[str] = None,
    ):
        """Initialize the prediction sampling callback.
        
        Args:
            num_samples: Number of samples to visualize
            sampling_frequency: Sample every N epochs
            save_dir: Directory to save samples
        """
        super().__init__()
        self.num_samples = num_samples
        self.sampling_frequency = sampling_frequency
        self.save_dir = save_dir
        self.samples = None
        self.targets = None
    
    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Initialize sample collection at the start of validation.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        # Only collect samples every sampling_frequency epochs
        if trainer.current_epoch % self.sampling_frequency != 0:
            return
        
        # Reset sample collection
        self.samples = None
        self.targets = None
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect samples during validation.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            outputs: Outputs from the validation step
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index
        """
        # Only collect samples every sampling_frequency epochs
        if trainer.current_epoch % self.sampling_frequency != 0:
            return
            
        # Only collect from the first few batches
        if batch_idx >= self.num_samples // batch['sequence'].size(0) + 1:
            return
            
        # If this is the first batch we're collecting from
        if self.samples is None:
            self.samples = []
            self.targets = []
            
        # Get predictions from outputs
        if 'predictions' in outputs:
            predictions = outputs['predictions']
        else:
            # If predictions not in outputs, use model to predict
            with torch.no_grad():
                predictions = pl_module(batch['sequence']).sigmoid()
                
        # Save predictions and targets
        self.samples.append(predictions.detach().cpu())
        self.targets.append(batch['targets'].detach().cpu())
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Process collected samples at the end of validation.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        # Only process samples every sampling_frequency epochs
        if trainer.current_epoch % self.sampling_frequency != 0:
            return
            
        # If no samples were collected, return
        if self.samples is None or not self.samples:
            return
            
        # Concatenate samples and targets
        all_samples = torch.cat(self.samples)
        all_targets = torch.cat(self.targets)
        
        # Limit to num_samples
        if all_samples.size(0) > self.num_samples:
            all_samples = all_samples[:self.num_samples]
            all_targets = all_targets[:self.num_samples]
            
        # Convert to numpy
        samples_np = all_samples.numpy()
        targets_np = all_targets.numpy()
        
        # Log samples
        self._log_prediction_statistics(trainer, samples_np, targets_np)
        
        # Save samples if save_dir is provided
        if self.save_dir:
            self._save_prediction_samples(
                trainer, samples_np, targets_np
            )
    
    def _log_prediction_statistics(
        self,
        trainer: pl.Trainer,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> None:
        """Log statistics about predictions.
        
        Args:
            trainer: PyTorch Lightning trainer
            predictions: Prediction samples
            targets: Target samples
        """
        # Calculate prediction statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # Log statistics
        if trainer.logger:
            trainer.logger.experiment.add_histogram(
                'val_predictions',
                predictions.flatten(),
                global_step=trainer.global_step
            )
            trainer.logger.experiment.add_histogram(
                'val_targets',
                targets.flatten(),
                global_step=trainer.global_step
            )
    
    def _save_prediction_samples(
        self,
        trainer: pl.Trainer,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> None:
        """Save prediction samples to disk.
        
        Args:
            trainer: PyTorch Lightning trainer
            predictions: Prediction samples
            targets: Target samples
        """
        # Ensure save directory exists
        save_dir = self.save_dir or os.path.join(
            trainer.logger.log_dir, "predictions"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # Save predictions and targets
        np.save(
            os.path.join(save_dir, f'preds_epoch_{trainer.current_epoch:03d}.npy'),
            predictions
        )
        np.save(
            os.path.join(save_dir, f'targets_epoch_{trainer.current_epoch:03d}.npy'),
            targets
        )
