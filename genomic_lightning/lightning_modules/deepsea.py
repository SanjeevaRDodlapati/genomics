"""Lightning module for DeepSEA-like genomic models."""

import torch
import torchmetrics
from typing import Dict, Any, Optional, Union, List, Tuple

from genomic_lightning.lightning_modules.base import GenomicBaseModule


class DeepSEAModule(GenomicBaseModule):
    """Lightning module specialized for DeepSEA and similar model architectures.
    
    Adds DeepSEA-specific functionality on top of the base genomic module.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: Optional[str] = "reduce_on_plateau",
        criterion: Optional[torch.nn.Module] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Initialize the DeepSEA module.
        
        Args:
            model: PyTorch model
            learning_rate: Initial learning rate
            weight_decay: L2 regularization factor
            optimizer: Optimizer type ('adam', 'sgd', etc.)
            scheduler: Learning rate scheduler ('reduce_on_plateau', 'cosine', etc.)
            criterion: Loss function, defaults to BCEWithLogitsLoss
            pos_weight: Optional positive class weight for imbalanced datasets
        """
        # Initialize criterion with positive weights if provided
        if criterion is None and pos_weight is not None:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
        )
    
    def _init_metrics(self):
        """Initialize DeepSEA-specific metrics for evaluation."""
        super()._init_metrics()
        
        # Add DeepSEA-specific metrics
        self.train_metrics.add_metrics({
            "accuracy": torchmetrics.Accuracy(task="binary"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
            "f1": torchmetrics.F1Score(task="binary"),
        })
        
        self.val_metrics.add_metrics({
            "accuracy": torchmetrics.Accuracy(task="binary"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
            "f1": torchmetrics.F1Score(task="binary"),
        })
        
        self.test_metrics.add_metrics({
            "accuracy": torchmetrics.Accuracy(task="binary"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
            "f1": torchmetrics.F1Score(task="binary"),
        })
    
    def training_step(self, batch, batch_idx):
        """DeepSEA-specific training step logic.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        # Use the base implementation for standard steps
        return super().training_step(batch, batch_idx)
        
    def on_validation_epoch_end(self):
        """Actions to perform at the end of a validation epoch.
        
        For DeepSEA models, this could involve specialized visualization.
        """
        # DeepSEA-specific visualization or analysis could go here
        pass
