"""
Base Lightning Module for genomic models.
"""

import os
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Union, Tuple

from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score


class BaseLightningModule(pl.LightningModule):
    """
    Base PyTorch Lightning module for genomic models.
    
    This class provides common functionality for genomic deep learning models,
    including training, validation, testing, and prediction steps.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        loss_function: str = "binary_cross_entropy",
        metrics: Optional[List[str]] = None,
        prediction_output_dir: Optional[str] = None,
        output_format: str = "npy",
    ):
        """
        Initialize the base Lightning module.
        
        Args:
            model: PyTorch model
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            loss_function: Loss function name
            metrics: List of metric names to track
            prediction_output_dir: Directory to save predictions
            output_format: Format to save predictions (npy, csv, h5)
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.prediction_output_dir = prediction_output_dir
        self.output_format = output_format
        
        # Set up metrics
        metrics = metrics or ["auroc"]
        self.setup_metrics(metrics)
        
        # Save predictions for later use
        self.test_predictions = []
        self.test_targets = []
        self.test_metadata = []
        
        # Save hyperparameters for loading
        self.save_hyperparameters(ignore=["model"])
        
    def setup_metrics(self, metrics: List[str]):
        """
        Set up tracking metrics for the module.
        
        Args:
            metrics: List of metric names to track
        """
        self.train_metrics = torch.nn.ModuleDict()
        self.val_metrics = torch.nn.ModuleDict()
        self.test_metrics = torch.nn.ModuleDict()
        
        for metric in metrics:
            if metric == "auroc":
                self.train_metrics["auroc"] = AUROC(task="binary")
                self.val_metrics["auroc"] = AUROC(task="binary")
                self.test_metrics["auroc"] = AUROC(task="binary")
            elif metric == "accuracy":
                self.train_metrics["accuracy"] = Accuracy(task="binary")
                self.val_metrics["accuracy"] = Accuracy(task="binary")
                self.test_metrics["accuracy"] = Accuracy(task="binary")
            elif metric == "precision":
                self.train_metrics["precision"] = Precision(task="binary")
                self.val_metrics["precision"] = Precision(task="binary")
                self.test_metrics["precision"] = Precision(task="binary")
            elif metric == "recall":
                self.train_metrics["recall"] = Recall(task="binary")
                self.val_metrics["recall"] = Recall(task="binary")
                self.test_metrics["recall"] = Recall(task="binary")
            elif metric == "f1":
                self.train_metrics["f1"] = F1Score(task="binary")
                self.val_metrics["f1"] = F1Score(task="binary")
                self.test_metrics["f1"] = F1Score(task="binary")
            # Custom metrics are handled in subclasses
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)
    
    def compute_loss(self, y_hat, y):
        """
        Compute loss between predictions and targets.
        
        Args:
            y_hat: Predictions
            y: Targets
            
        Returns:
            Loss value
        """
        if self.loss_function == "binary_cross_entropy":
            return torch.nn.functional.binary_cross_entropy(y_hat, y)
        elif self.loss_function == "mse":
            return torch.nn.functional.mse_loss(y_hat, y)
        elif self.loss_function == "cross_entropy":
            return torch.nn.functional.cross_entropy(y_hat, y)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.compute_loss(y_hat, y)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for name, metric in self.train_metrics.items():
            self.log(f"train_{name}", metric(y_hat, y), on_step=False, on_epoch=True)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Dictionary with predictions, targets, and loss
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.compute_loss(y_hat, y)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric in self.val_metrics.items():
            self.log(f"val_{name}", metric(y_hat, y), on_step=False, on_epoch=True)
        
        return {"loss": loss, "preds": y_hat, "targets": y}
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Dictionary with predictions, targets, and metadata
        """
        if len(batch) == 2:
            x, y = batch
            metadata = None
        else:
            x, y, metadata = batch
        
        y_hat = self.model(x)
        loss = self.compute_loss(y_hat, y)
        
        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for name, metric in self.test_metrics.items():
            self.log(f"test_{name}", metric(y_hat, y), on_step=False, on_epoch=True)
        
        # Save predictions for later analysis
        self.test_predictions.append(y_hat.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        if metadata is not None:
            self.test_metadata.append(metadata)
        
        return {"loss": loss, "preds": y_hat, "targets": y, "metadata": metadata}
    
    def on_test_end(self):
        """
        Called at the end of testing to process all predictions.
        """
        if not self.test_predictions:
            return
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.test_predictions, dim=0).numpy()
        all_targets = torch.cat(self.test_targets, dim=0).numpy()
        
        # Process metadata if available
        all_metadata = None
        if self.test_metadata:
            try:
                all_metadata = np.concatenate(self.test_metadata, axis=0)
            except:
                # If metadata cannot be concatenated, keep as list
                all_metadata = self.test_metadata
        
        # Save predictions if output directory is specified
        if self.prediction_output_dir:
            os.makedirs(self.prediction_output_dir, exist_ok=True)
            pred_path = os.path.join(self.prediction_output_dir, "predictions")
            
            if self.output_format == "npy":
                np.save(f"{pred_path}.npy", all_preds)
                np.save(f"{pred_path}_targets.npy", all_targets)
                if all_metadata is not None:
                    np.save(f"{pred_path}_metadata.npy", all_metadata)
            elif self.output_format == "csv":
                np.savetxt(f"{pred_path}.csv", all_preds, delimiter=",")
                np.savetxt(f"{pred_path}_targets.csv", all_targets, delimiter=",")
            elif self.output_format == "h5":
                import h5py
                with h5py.File(f"{pred_path}.h5", "w") as f:
                    f.create_dataset("predictions", data=all_preds)
                    f.create_dataset("targets", data=all_targets)
                    if all_metadata is not None:
                        f.create_dataset("metadata", data=all_metadata)
        
        # Reset predictions for next test run
        self.test_predictions = []
        self.test_targets = []
        self.test_metadata = []
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            dataloader_idx: Dataloader index
            
        Returns:
            Predictions
        """
        if isinstance(batch, tuple) and len(batch) >= 1:
            x = batch[0]
        else:
            x = batch
        
        return self.model(x)
    
    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        
        Returns:
            Optimizer configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
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
            },
        }


class GenomicBaseModule(BaseLightningModule):
    """
    Alias for BaseLightningModule for backward compatibility.
    """
    pass