"""Base Lightning module for genomic models."""

import pytorch_lightning as pl
import torch
import torchmetrics
from typing import Dict, Any, Optional, Union, List, Tuple


class GenomicBaseModule(pl.LightningModule):
    """Base Lightning module for genomic models.
    
    This serves as the foundation for all genomic model implementations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: Optional[str] = "reduce_on_plateau",
        criterion: Optional[torch.nn.Module] = None,
    ):
        """Initialize the base genomic module.

        Args:
            model: PyTorch model
            learning_rate: Initial learning rate
            weight_decay: L2 regularization factor
            optimizer: Optimizer type ('adam', 'sgd', etc.)
            scheduler: Learning rate scheduler ('reduce_on_plateau', 'cosine', etc.)
            criterion: Loss function, defaults to BCEWithLogitsLoss
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler
        self.criterion = criterion or torch.nn.BCEWithLogitsLoss()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=["model", "criterion"])
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metrics for evaluation."""
        metrics = {
            "auroc": torchmetrics.AUROC(task="binary"),
            "average_precision": torchmetrics.AveragePrecision(task="binary"),
        }
        
        self.train_metrics = torchmetrics.MetricCollection(metrics, prefix="train_")
        self.val_metrics = torchmetrics.MetricCollection(metrics, prefix="val_")
        self.test_metrics = torchmetrics.MetricCollection(metrics, prefix="test_")
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step logic.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        inputs = batch["sequence"]
        targets = batch["targets"]
        
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Update metrics
        metrics = self.train_metrics(outputs.sigmoid(), targets)
        self.log_dict(metrics)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step logic.
        
        Args:
            batch: Validation batch
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        inputs = batch["sequence"]
        targets = batch["targets"]
        
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Update metrics
        metrics = self.val_metrics(outputs.sigmoid(), targets)
        self.log_dict(metrics)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step logic.
        
        Args:
            batch: Test batch
            batch_idx: Batch index
            
        Returns:
            Test loss
        """
        inputs = batch["sequence"]
        targets = batch["targets"]
        
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Update metrics
        metrics = self.test_metrics(outputs.sigmoid(), targets)
        self.log_dict(metrics)
        self.log("test_loss", loss, prog_bar=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step logic.
        
        Args:
            batch: Prediction batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index
            
        Returns:
            Model predictions
        """
        inputs = batch["sequence"]
        outputs = self(inputs)
        
        return {
            "predictions": outputs.sigmoid(),
            "inputs": inputs,
            "metadata": batch.get("metadata", None)
        }
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Optimizer configuration
        """
        # Configure optimizer
        if self.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # Configure scheduler if specified
        if self.scheduler_type is None:
            return optimizer
        
        if self.scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        
        # Default case
        return optimizer
