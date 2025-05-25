"""
DanQ Lightning Module implementation for genomic sequence analysis.
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Union, Tuple

from genomic_lightning.models.danq import DanQModel
from genomic_lightning.lightning_modules.base import BaseLightningModule


class DanQLightningModule(BaseLightningModule):
    """
    PyTorch Lightning module for the DanQ model.
    Extends the BaseLightningModule with DanQ-specific functionality.
    """
    
    def __init__(
        self,
        n_outputs: int = 919,
        sequence_length: int = 1000,
        n_genomic_features: int = 4,
        conv_kernel_size: int = 26,
        conv_filters: int = 320,
        pool_size: int = 13,
        pool_stride: int = 13,
        lstm_hidden: int = 320,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        loss_function: str = "binary_cross_entropy",
        metrics: Optional[List[str]] = None,
        prediction_output_dir: Optional[str] = None,
        output_format: str = "npy",
    ):
        """
        Initialize the DanQ Lightning module.
        
        Args:
            n_outputs: Number of output predictions
            sequence_length: Length of input DNA sequence
            n_genomic_features: Number of features per position (4 for DNA)
            conv_kernel_size: Size of convolutional filters
            conv_filters: Number of convolutional filters
            pool_size: Max pooling size
            pool_stride: Max pooling stride
            lstm_hidden: Number of LSTM hidden units
            dropout_rate: Dropout rate
            learning_rate: Learning rate for the optimizer
            weight_decay: L2 regularization weight
            loss_function: Loss function name
            metrics: List of metrics to track
            prediction_output_dir: Directory to save predictions
            output_format: Format to save predictions
        """
        model = DanQModel(
            sequence_length=sequence_length,
            n_genomic_features=n_genomic_features,
            n_outputs=n_outputs,
            conv_kernel_size=conv_kernel_size,
            conv_filters=conv_filters,
            pool_size=pool_size,
            pool_stride=pool_stride,
            lstm_hidden=lstm_hidden,
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
        Configure optimizers with layer-specific learning rates.
        
        Returns:
            Optimizer configuration
        """
        # Group parameters by layer type for custom learning rates
        conv_params = []
        lstm_params = []
        dense_params = []
        
        for name, param in self.model.named_parameters():
            if 'conv' in name:
                conv_params.append(param)
            elif 'lstm' in name:
                lstm_params.append(param)
            else:
                dense_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': conv_params, 'lr': self.hparams.learning_rate},
            {'params': lstm_params, 'lr': self.hparams.learning_rate * 0.5},  # Lower LR for LSTM
            {'params': dense_params, 'lr': self.hparams.learning_rate * 1.5}   # Higher LR for dense
        ]
        
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
