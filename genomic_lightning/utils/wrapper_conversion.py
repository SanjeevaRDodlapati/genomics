"""Utilities for converting existing models and wrappers to Lightning."""

import torch
import inspect
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

from genomic_lightning.lightning_modules.base import GenomicBaseModule


def convert_wrapper_to_lightning(
    wrapper,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    scheduler: Optional[str] = "reduce_on_plateau"
) -> GenomicBaseModule:
    """Convert a UAVarPrior/FuGEP wrapper to a Lightning module.
    
    Args:
        wrapper: Wrapper instance containing model, optimizer, and criterion
        learning_rate: Learning rate to use (if optimizers need to be recreated)
        weight_decay: Weight decay to use (if optimizers need to be recreated)
        scheduler: Learning rate scheduler (if optimizers need to be recreated)
        
    Returns:
        A GenomicBaseModule wrapping the model from the wrapper
    """
    # Extract model from wrapper
    if hasattr(wrapper, 'model'):
        model = wrapper.model
    else:
        raise ValueError("Wrapper does not have a 'model' attribute")
    
    # Extract criterion from wrapper
    criterion = None
    if hasattr(wrapper, 'criterion'):
        criterion = wrapper.criterion
    
    # Create a Lightning module
    lightning_module = GenomicBaseModule(
        model=model,
        criterion=criterion,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler
    )
    
    # If wrapper has special methods for forward pass, try to preserve them
    if hasattr(wrapper, 'forward') and callable(wrapper.forward):
        # Create a forward method that mimics the wrapper's forward method
        def wrapped_forward(self, *args, **kwargs):
            return wrapper.forward(*args, **kwargs)
        
        # Bind the method to the Lightning module
        setattr(lightning_module.__class__, 'forward', wrapped_forward)
    
    return lightning_module


def analyze_wrapper(wrapper) -> Dict[str, Any]:
    """Analyze a wrapper to extract useful information for conversion.
    
    Args:
        wrapper: Wrapper instance to analyze
        
    Returns:
        Dictionary with information about the wrapper
    """
    info = {
        'has_model': hasattr(wrapper, 'model'),
        'has_optimizer': hasattr(wrapper, 'optimizer'),
        'has_criterion': hasattr(wrapper, 'criterion'),
        'methods': [],
        'properties': [],
        'training_methods': [],
        'validation_methods': [],
        'prediction_methods': []
    }
    
    # Get all methods and properties
    for name in dir(wrapper):
        # Skip private attributes
        if name.startswith('_'):
            continue
        
        # Get attribute
        attr = getattr(wrapper, name)
        
        # Check if it's a method
        if callable(attr):
            info['methods'].append(name)
            
            # Categorize methods
            if 'train' in name.lower():
                info['training_methods'].append(name)
            elif any(word in name.lower() for word in ['valid', 'val', 'eval']):
                info['validation_methods'].append(name)
            elif any(word in name.lower() for word in ['predict', 'inference']):
                info['prediction_methods'].append(name)
        
        # Check if it's a property
        elif not name.startswith('__'):
            info['properties'].append(name)
    
    # Check for specific method signatures
    if hasattr(wrapper, 'train_step') and callable(wrapper.train_step):
        # Inspect train_step method
        sig = inspect.signature(wrapper.train_step)
        info['train_step_params'] = list(sig.parameters.keys())
    
    if hasattr(wrapper, 'validate_step') and callable(wrapper.validate_step):
        # Inspect validate_step method
        sig = inspect.signature(wrapper.validate_step)
        info['validate_step_params'] = list(sig.parameters.keys())
    
    return info


def create_wrapper_adapter_module(wrapper_class: type) -> type:
    """Create a Lightning module class that adapts a specific wrapper class.
    
    Args:
        wrapper_class: Wrapper class to adapt
        
    Returns:
        A new Lightning module class
    """
    class WrapperAdapterModule(GenomicBaseModule):
        """Lightning module that adapts a specific wrapper class."""
        
        def __init__(
            self,
            wrapper_instance,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0,
            scheduler: Optional[str] = "reduce_on_plateau"
        ):
            """Initialize with an instance of the wrapper.
            
            Args:
                wrapper_instance: Instance of the wrapper
                learning_rate: Learning rate
                weight_decay: Weight decay
                scheduler: Learning rate scheduler
            """
            # Extract model from wrapper
            if not hasattr(wrapper_instance, 'model'):
                raise ValueError("Wrapper does not have a 'model' attribute")
            
            model = wrapper_instance.model
            
            # Extract criterion from wrapper
            criterion = None
            if hasattr(wrapper_instance, 'criterion'):
                criterion = wrapper_instance.criterion
            
            super().__init__(
                model=model,
                criterion=criterion,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                scheduler=scheduler
            )
            
            # Store wrapper instance
            self.wrapper = wrapper_instance
        
        def training_step(self, batch, batch_idx):
            """Use wrapper's train_step if available, otherwise use default."""
            if hasattr(self.wrapper, 'train_step') and callable(self.wrapper.train_step):
                # Call wrapper's train_step
                result = self.wrapper.train_step(batch)
                
                # Extract loss
                if isinstance(result, dict) and 'loss' in result:
                    loss = result['loss']
                    # If loss is a tensor, use it directly
                    if isinstance(loss, torch.Tensor):
                        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
                        return loss
                    # If loss is a scalar, convert to tensor
                    else:
                        loss_tensor = torch.tensor(loss, device=self.device)
                        self.log('train_loss', loss_tensor, prog_bar=True, on_step=True, on_epoch=True)
                        return loss_tensor
            
            # Fall back to default implementation
            return super().training_step(batch, batch_idx)
        
        def validation_step(self, batch, batch_idx):
            """Use wrapper's validate_step if available, otherwise use default."""
            if hasattr(self.wrapper, 'validate_step') and callable(self.wrapper.validate_step):
                # Call wrapper's validate_step
                with torch.no_grad():
                    result = self.wrapper.validate_step(batch)
                
                # Extract loss
                if isinstance(result, dict) and 'loss' in result:
                    loss = result['loss']
                    # If loss is a tensor, use it directly
                    if isinstance(loss, torch.Tensor):
                        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
                        return loss
                    # If loss is a scalar, convert to tensor
                    else:
                        loss_tensor = torch.tensor(loss, device=self.device)
                        self.log('val_loss', loss_tensor, prog_bar=True, on_epoch=True)
                        return loss_tensor
            
            # Fall back to default implementation
            return super().validation_step(batch, batch_idx)
    
    # Set class name based on wrapper class
    WrapperAdapterModule.__name__ = f"{wrapper_class.__name__}LightningAdapter"
    WrapperAdapterModule.__qualname__ = f"{wrapper_class.__name__}LightningAdapter"
    
    return WrapperAdapterModule


def create_adapter_for_wrapper(
    wrapper_instance,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    scheduler: Optional[str] = "reduce_on_plateau"
) -> GenomicBaseModule:
    """Create a Lightning module instance that adapts a wrapper instance.
    
    Args:
        wrapper_instance: Instance of a wrapper
        learning_rate: Learning rate
        weight_decay: Weight decay
        scheduler: Learning rate scheduler
        
    Returns:
        A Lightning module instance that adapts the wrapper
    """
    wrapper_class = wrapper_instance.__class__
    
    # Create adapter class
    adapter_class = create_wrapper_adapter_module(wrapper_class)
    
    # Create instance of adapter
    return adapter_class(
        wrapper_instance=wrapper_instance,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler
    )
