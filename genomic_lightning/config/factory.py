"""Factory functions for creating model and data components from configuration."""

import torch
from typing import Dict, Any, Optional, Union, Type

from genomic_lightning.models.deepsea import DeepSEA
from genomic_lightning.models.danq import DanQModel
from genomic_lightning.models.chromdragonn import ChromDragoNNModel

from genomic_lightning.lightning_modules.base import GenomicBaseModule
from genomic_lightning.lightning_modules.deepsea import DeepSEAModule
from genomic_lightning.lightning_modules.danq import DanQLightningModule
from genomic_lightning.lightning_modules.chromdragonn import ChromDragoNNLightningModule
from genomic_lightning.data.sampler_data_module import SamplerDataModule


# Dictionary mapping model class names to their implementations
MODEL_REGISTRY = {
    "DeepSEA": DeepSEA,
    "DanQModel": DanQModel,
    "ChromDragoNNModel": ChromDragoNNModel,
}

# Dictionary mapping Lightning module class names to their implementations
MODULE_REGISTRY = {
    "GenomicBaseModule": GenomicBaseModule,
    "DeepSEAModule": DeepSEAModule,
    "DanQLightningModule": DanQLightningModule,
    "ChromDragoNNLightningModule": ChromDragoNNLightningModule,
}


def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Create a model instance from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model class is not found in registry
    """
    # Get model class from registry
    model_class = config.get("class", None)
    if model_class not in MODEL_REGISTRY:
        raise ValueError(
            f"Model class '{model_class}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    # Extract model arguments
    model_args = config.get("args", {})
    
    # Create model instance
    model = MODEL_REGISTRY[model_class](**model_args)
    
    return model


def create_lightning_module(
    config: Dict[str, Any],
    model: Optional[torch.nn.Module] = None
) -> GenomicBaseModule:
    """Create a Lightning module from configuration.
    
    Args:
        config: Module configuration dictionary
        model: Optional pre-created model instance
        
    Returns:
        Instantiated Lightning module
        
    Raises:
        ValueError: If module class is not found in registry
    """
    # Create model if not provided
    if model is None:
        model = create_model(config.get("model", {}))
    
    # Get module class from registry (default to appropriate module based on model)
    module_class_name = config.get("lightning_module", None)
    if module_class_name is None:
        # Default to appropriate module based on model class
        model_class = config.get("model", {}).get("class", None)
        if model_class == "DeepSEA":
            module_class_name = "DeepSEAModule"
        elif model_class == "DanQModel":
            module_class_name = "DanQLightningModule"
        elif model_class == "ChromDragoNNModel":
            module_class_name = "ChromDragoNNLightningModule"
        else:
            module_class_name = "GenomicBaseModule"
    
    if module_class_name not in MODULE_REGISTRY:
        raise ValueError(
            f"Lightning module class '{module_class_name}' not found in registry. "
            f"Available modules: {list(MODULE_REGISTRY.keys())}"
        )
    
    # Extract module arguments
    module_args = config.get("lightning_args", {})
    
    # Create module instance
    module = MODULE_REGISTRY[module_class_name](model=model, **module_args)
    
    return module


def create_data_module(config: Dict[str, Any], sampler=None) -> SamplerDataModule:
    """Create a data module from configuration.
    
    Args:
        config: Data configuration dictionary
        sampler: Optional pre-created sampler instance
        
    Returns:
        Instantiated data module
    """
    # Extract data module arguments
    data_args = config.get("data_args", {})
    
    # Use provided sampler or get from config
    if sampler is None:
        # Note: This assumes you have a function to create a sampler from config
        # which would need to be implemented based on your sampler framework
        from genomic_lightning.utils.sampler_utils import create_sampler
        sampler = create_sampler(config.get("sampler", {}))
    
    # Create data module
    data_module = SamplerDataModule(sampler=sampler, **data_args)
    
    return data_module
