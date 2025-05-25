"""
ChromDragoNN Lightning Module implementation for genomic sequence analysis.
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Union, Tuple

from genomic_lightning.models.chromdragonn import ChromDragoNNModel
from genomic_lightning.lightning_modules.base import BaseLightningModule


class ChromDragoNNLightningModule(BaseLightningModule):
    """
    PyTorch Lightning module for the ChromDragoNN model.
    Extends the BaseLightningModule with ChromDragoNN-specific functionality.
    """
    
    def __init__(
        self,
        n_outputs: int = 919,
        sequence_length: int = 1000,
        n_genomic_features: int = 4,
        n_filters: int = 300,
        n_residual_blocks: int = 5,
        first_kernel_size: int = 19,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        loss_function: str = "binary_cross_entropy",
        metrics: Optional[List[str]] = None,
        prediction_output_dir: Optional[str] = None,
        output_format: str = "npy",
    ):
        """
        Initialize the ChromDragoNN Lightning module.
        
        Args:
            n_outputs: Number of output predictions
            sequence_length: Length of input DNA sequence
            n_genomic_features: Number of features per position (4 for DNA)
            n_filters: Number of convolutional filters
            n_residual_blocks: Number of residual blocks
            first_kernel_size: Size of first convolutional filter
            dropout_rate: Dropout rate
            learning_rate: Learning rate for the optimizer
            weight_decay: L2 regularization weight
            loss_function: Loss function name
            metrics: List of metrics to track
            prediction_output_dir: Directory to save predictions
            output_format: Format to save predictions
        """
        model = ChromDragoNNModel(
            sequence_length=sequence_length,
            n_genomic_features=n_genomic_features,
            n_outputs=n_outputs,
            n_filters=n_filters,
            n_residual_blocks=n_residual_blocks,
            first_kernel_size=first_kernel_size,
            dropout_rate=dropout_rate
        )
        
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_function=loss_function,
            metrics=metrics,
            prediction_output_dir=prediction_output_dir,
            output_format=output_format
        )
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        """
        Configure optimizers with cosine annealing scheduler.
        
        Returns:
            Optimizer configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
        
    def on_epoch_start(self):
        """
        Custom logic to run at the start of each epoch.
        """
        # Track gradient norms for residual blocks
        if self.current_epoch % 5 == 0:  # Every 5 epochs
            for i, block in enumerate(self.model.residual_blocks):
                for name, param in block.named_parameters():
                    if param.requires_grad:
                        self.logger.experiment.add_histogram(
                            f"gradients/block_{i}_{name}", 
                            param.grad, 
                            self.current_epoch
                        )