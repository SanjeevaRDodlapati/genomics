"""
Utilities for wrapping models with PyTorch Lightning.

This module provides functions to wrap PyTorch models with PyTorch Lightning
for easy training and evaluation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
import pytorch_lightning as pl
import torchmetrics
import logging

logger = logging.getLogger(__name__)

class GenericLightningWrapper(pl.LightningModule):
    """
    Generic PyTorch Lightning wrapper for genomic deep learning models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        loss_function: str = "binary_cross_entropy",
        metrics: Optional[List[str]] = None,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize the lightning wrapper.
        
        Args:
            model: PyTorch model to wrap
            learning_rate: Learning rate for optimization
            optimizer: Optimizer to use ('adam', 'sgd', etc.)
            loss_function: Loss function to use ('binary_cross_entropy', 'mse', etc.)
            metrics: List of metrics to track ('auroc', 'auprc', etc.)
            optimizer_kwargs: Additional arguments for optimizer
            scheduler: Learning rate scheduler ('cosine', 'step', etc.)
            scheduler_kwargs: Additional arguments for scheduler
            class_weights: Optional weights for handling class imbalance
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.loss_function_name = loss_function
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.class_weights = class_weights
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model'])
        
        # Set up metrics
        self.setup_metrics(metrics or ['auroc'])
        
        # Set up loss function
        self.loss_function = self._get_loss_function(loss_function)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)
    
    def _get_loss_function(self, loss_name: str) -> Callable:
        """
        Get the loss function by name.
        
        Args:
            loss_name: Name of the loss function
            
        Returns:
            Loss function
        """
        if loss_name.lower() == "binary_cross_entropy":
            if self.class_weights is not None:
                return lambda y_pred, y_true: torch.nn.functional.binary_cross_entropy(
                    y_pred, y_true, weight=self.class_weights
                )
            else:
                return torch.nn.functional.binary_cross_entropy
                
        elif loss_name.lower() == "bce_with_logits":
            if self.class_weights is not None:
                return lambda y_pred, y_true: torch.nn.functional.binary_cross_entropy_with_logits(
                    y_pred, y_true, pos_weight=self.class_weights
                )
            else:
                return torch.nn.functional.binary_cross_entropy_with_logits
                
        elif loss_name.lower() == "mse":
            return torch.nn.functional.mse_loss
            
        elif loss_name.lower() == "l1":
            return torch.nn.functional.l1_loss
            
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def setup_metrics(self, metric_names: List[str]):
        """
        Set up metrics for training and evaluation.
        
        Args:
            metric_names: Names of the metrics to track
        """
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        
        # Try to determine output size of the model
        try:
            # Use small input for genomic data (4 channels, short sequence)
            dummy_input = torch.randn(1, 4, 100)
            with torch.no_grad():
                dummy_output = self.model(dummy_input)
            
            if isinstance(dummy_output, tuple):
                num_classes = dummy_output[0].shape[1]
            else:
                num_classes = dummy_output.shape[1]
        except Exception as e:
            logger.warning(f"Could not determine output size: {str(e)}")
            num_classes = 1
        
        # Create metrics
        for name in metric_names:
            if name.lower() == "auroc":
                self.train_metrics[name] = torchmetrics.AUROC(task="binary", num_classes=num_classes)
                self.val_metrics[name] = torchmetrics.AUROC(task="binary", num_classes=num_classes)
                self.test_metrics[name] = torchmetrics.AUROC(task="binary", num_classes=num_classes)
                
            elif name.lower() == "auprc" or name.lower() == "auc":
                self.train_metrics[name] = torchmetrics.AveragePrecision(task="binary", num_classes=num_classes)
                self.val_metrics[name] = torchmetrics.AveragePrecision(task="binary", num_classes=num_classes)
                self.test_metrics[name] = torchmetrics.AveragePrecision(task="binary", num_classes=num_classes)
                
            elif name.lower() == "accuracy":
                self.train_metrics[name] = torchmetrics.Accuracy(task="binary", num_classes=num_classes)
                self.val_metrics[name] = torchmetrics.Accuracy(task="binary", num_classes=num_classes)
                self.test_metrics[name] = torchmetrics.Accuracy(task="binary", num_classes=num_classes)
                
            elif name.lower() == "f1":
                self.train_metrics[name] = torchmetrics.F1Score(task="binary", num_classes=num_classes)
                self.val_metrics[name] = torchmetrics.F1Score(task="binary", num_classes=num_classes)
                self.test_metrics[name] = torchmetrics.F1Score(task="binary", num_classes=num_classes)
                
            elif name.lower() == "precision":
                self.train_metrics[name] = torchmetrics.Precision(task="binary", num_classes=num_classes)
                self.val_metrics[name] = torchmetrics.Precision(task="binary", num_classes=num_classes)
                self.test_metrics[name] = torchmetrics.Precision(task="binary", num_classes=num_classes)
                
            elif name.lower() == "recall":
                self.train_metrics[name] = torchmetrics.Recall(task="binary", num_classes=num_classes)
                self.val_metrics[name] = torchmetrics.Recall(task="binary", num_classes=num_classes)
                self.test_metrics[name] = torchmetrics.Recall(task="binary", num_classes=num_classes)
                
            elif name.lower() == "mse":
                self.train_metrics[name] = torchmetrics.MeanSquaredError()
                self.val_metrics[name] = torchmetrics.MeanSquaredError()
                self.test_metrics[name] = torchmetrics.MeanSquaredError()
                
            elif name.lower() == "mae":
                self.train_metrics[name] = torchmetrics.MeanAbsoluteError()
                self.val_metrics[name] = torchmetrics.MeanAbsoluteError()
                self.test_metrics[name] = torchmetrics.MeanAbsoluteError()
                
            else:
                logger.warning(f"Unknown metric: {name}")
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Tuple of (sequence, target)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        sequences, targets = batch
        predictions = self.model(sequences)
        
        # Calculate loss
        loss = self.loss_function(predictions, targets)
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate and log metrics
        for name, metric in self.train_metrics.items():
            value = metric(predictions, targets)
            self.log(f'train_{name}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Tuple of (sequence, target)
            batch_idx: Batch index
            
        Returns:
            Dictionary of validation values
        """
        sequences, targets = batch
        predictions = self.model(sequences)
        
        # Calculate loss
        loss = self.loss_function(predictions, targets)
        
        # Log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate and log metrics
        for name, metric in self.val_metrics.items():
            value = metric(predictions, targets)
            self.log(f'val_{name}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss}
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Tuple of (sequence, target)
            batch_idx: Batch index
            
        Returns:
            Dictionary of test values
        """
        sequences, targets = batch
        predictions = self.model(sequences)
        
        # Calculate loss
        loss = self.loss_function(predictions, targets)
        
        # Log loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Calculate and log metrics
        for name, metric in self.test_metrics.items():
            value = metric(predictions, targets)
            self.log(f'test_{name}', value, on_step=False, on_epoch=True)
        
        return {'test_loss': loss}
    
    def configure_optimizers(self) -> Dict:
        """
        Configure optimizers and schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Create optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate,
                **self.optimizer_kwargs
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.learning_rate,
                **self.optimizer_kwargs
            )
        elif self.optimizer_name.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(), 
                lr=self.learning_rate,
                **self.optimizer_kwargs
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.learning_rate,
                **self.optimizer_kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        # Create scheduler if specified
        if self.scheduler_name is None:
            return optimizer
        
        if self.scheduler_name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **self.scheduler_kwargs
            )
        elif self.scheduler_name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                **self.scheduler_kwargs
            )
        elif self.scheduler_name.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **self.scheduler_kwargs
            )
        elif self.scheduler_name.lower() == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                **self.scheduler_kwargs
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
        
        # Return optimizer and scheduler configuration
        if self.scheduler_name.lower() == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }


def wrap_model_with_lightning(
    model: nn.Module,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    loss_function: str = "binary_cross_entropy",
    metrics: Optional[List[str]] = None,
    optimizer_kwargs: Optional[Dict] = None,
    scheduler: Optional[str] = None,
    scheduler_kwargs: Optional[Dict] = None,
    class_weights: Optional[torch.Tensor] = None
) -> pl.LightningModule:
    """
    Wrap a PyTorch model with a PyTorch Lightning module for easy training.
    
    Args:
        model: PyTorch model to wrap
        learning_rate: Learning rate for optimization
        optimizer: Optimizer to use ('adam', 'sgd', etc.)
        loss_function: Loss function to use ('binary_cross_entropy', 'mse', etc.)
        metrics: List of metrics to track ('auroc', 'auprc', etc.)
        optimizer_kwargs: Additional arguments for optimizer
        scheduler: Learning rate scheduler ('cosine', 'step', etc.)
        scheduler_kwargs: Additional arguments for scheduler
        class_weights: Optional weights for handling class imbalance
        
    Returns:
        Lightning module wrapping the model
    """
    return GenericLightningWrapper(
        model=model,
        learning_rate=learning_rate,
        optimizer=optimizer,
        loss_function=loss_function,
        metrics=metrics,
        optimizer_kwargs=optimizer_kwargs,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        class_weights=class_weights
    )