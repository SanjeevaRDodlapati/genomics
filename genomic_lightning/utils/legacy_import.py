"""
Utilities for importing models from legacy UAVarPrior and FuGEP frameworks.

This module provides functions to load pre-trained models from the legacy
UAVarPrior and FuGEP frameworks and convert them to be compatible with
GenomicLightning.
"""

import os
import sys
import torch
import yaml
import importlib.util
from typing import Dict, Any, Optional, Union, Tuple, Type
import logging

# Set up logger
logger = logging.getLogger(__name__)

def import_uavarprior_model(
    model_path: str,
    model_type: str = "deepsea",
    class_definition: Optional[Type] = None,
    config_path: Optional[str] = None
) -> torch.nn.Module:
    """
    Import a model from the UAVarPrior framework.
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        model_type: Type of model ('deepsea', 'danq', 'chromdragonn', or 'custom')
        class_definition: If model_type is 'custom', provide the model class
        config_path: Optional path to configuration file
        
    Returns:
        PyTorch model loaded with weights from the UAVarPrior model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # Load model weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Determine model architecture
    if model_type.lower() == "deepsea":
        from genomic_lightning.models.deepsea import DeepSEAModel
        
        # Extract architecture details from config if available
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract relevant parameters
            num_filters = config.get("num_filters", [320, 480, 960])
            filter_sizes = config.get("filter_sizes", [8, 8, 8])
            pool_sizes = config.get("pool_sizes", [4, 4, 4])
            dropout_rates = config.get("dropout_rates", [0.2, 0.2, 0.5])
            num_targets = config.get("num_targets", 919)
            
            model = DeepSEAModel(
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                pool_sizes=pool_sizes,
                dropout_rates=dropout_rates,
                num_targets=num_targets
            )
        else:
            # Use default DeepSEA architecture (919 targets)
            model = DeepSEAModel()
            
    elif model_type.lower() == "danq":
        from genomic_lightning.models.danq import DanQModel
        
        # Extract architecture details from config if available
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract relevant parameters
            num_targets = config.get("num_targets", 919)
            num_filters = config.get("num_filters", 320)
            filter_size = config.get("filter_size", 26)
            pool_size = config.get("pool_size", 13)
            lstm_hidden = config.get("lstm_hidden", 320)
            lstm_layers = config.get("lstm_layers", 1)
            
            model = DanQModel(
                num_targets=num_targets,
                num_filters=num_filters,
                filter_size=filter_size,
                pool_size=pool_size,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers
            )
        else:
            # Use default DanQ architecture
            model = DanQModel()
            
    elif model_type.lower() == "chromdragonn":
        from genomic_lightning.models.chromdragonn import ChromDragoNNModel
        
        # Extract architecture details from config if available
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract relevant parameters
            num_targets = config.get("num_targets", 919)
            num_filters = config.get("num_filters", 300)
            filter_sizes = config.get("filter_sizes", [10, 15, 20])
            residual_blocks = config.get("residual_blocks", 3)
            
            model = ChromDragoNNModel(
                num_targets=num_targets,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                residual_blocks=residual_blocks
            )
        else:
            # Use default ChromDragoNN architecture
            model = ChromDragoNNModel()
            
    elif model_type.lower() == "custom":
        if class_definition is None:
            raise ValueError("For 'custom' model_type, you must provide the class_definition")
        
        # Initialize the custom model
        model = class_definition()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Check if state_dict structure matches the model
    # UAVarPrior models may have different key naming conventions
    try:
        # Try direct loading
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.warning(f"Direct loading failed: {str(e)}. Attempting to adapt state dict...")
        
        # Attempt to adapt the state dict to match the model's structure
        adapted_state_dict = {}
        
        # Common prefix differences
        uavarprior_prefixes = ["module.", "model.", "network."]
        
        for key, value in state_dict.items():
            # Try removing common prefixes
            adapted_key = key
            for prefix in uavarprior_prefixes:
                if key.startswith(prefix):
                    adapted_key = key[len(prefix):]
                    break
            
            # Try adding a common prefix if needed
            if adapted_key not in model.state_dict() and not any(adapted_key.startswith(p) for p in uavarprior_prefixes):
                for prefix in ["", "model.", "network."]:
                    if f"{prefix}{adapted_key}" in model.state_dict():
                        adapted_key = f"{prefix}{adapted_key}"
                        break
            
            # Handle common layer name differences
            if adapted_key not in model.state_dict():
                # Conv layers might be named differently
                if "conv" in adapted_key:
                    for potential_key in model.state_dict().keys():
                        if "conv" in potential_key and adapted_key.split(".")[-1] == potential_key.split(".")[-1]:
                            adapted_key = potential_key
                            break
                            
                # FC layers might be named differently
                elif "fc" in adapted_key or "linear" in adapted_key:
                    for potential_key in model.state_dict().keys():
                        if ("fc" in potential_key or "linear" in potential_key) and adapted_key.split(".")[-1] == potential_key.split(".")[-1]:
                            adapted_key = potential_key
                            break
            
            # Add the adapted key if it exists in the model
            if adapted_key in model.state_dict():
                # Check if shapes match
                if value.shape == model.state_dict()[adapted_key].shape:
                    adapted_state_dict[adapted_key] = value
                else:
                    logger.warning(f"Shape mismatch for {adapted_key}: {value.shape} vs {model.state_dict()[adapted_key].shape}")
            else:
                logger.warning(f"Could not find matching key for {key}")
        
        # Try loading the adapted state dict
        model.load_state_dict(adapted_state_dict, strict=False)
        
        # Check how much of the model was successfully loaded
        missing_keys = set(model.state_dict().keys()) - set(adapted_state_dict.keys())
        if missing_keys:
            logger.warning(f"Warning: {len(missing_keys)}/{len(model.state_dict())} keys were not loaded: {list(missing_keys)[:5]}...")
    
    return model


def import_fugep_model(
    model_path: str,
    model_config: str
) -> torch.nn.Module:
    """
    Import a model from the FuGEP framework.
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        model_config: Path to the model configuration file (.yml)
        
    Returns:
        PyTorch model loaded with weights from the FuGEP model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if not os.path.exists(model_config):
        raise FileNotFoundError(f"Config file not found: {model_config}")
    
    # Load the configuration
    with open(model_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model architecture details
    model_type = config.get("model_type", "deepsea")
    model_params = config.get("model_params", {})
    
    # Map FuGEP model types to GenomicLightning models
    if model_type.lower() == "deepsea":
        from genomic_lightning.models.deepsea import DeepSEAModel
        
        num_targets = model_params.get("num_targets", 919)
        num_filters = model_params.get("num_filters", [320, 480, 960])
        filter_sizes = model_params.get("filter_sizes", [8, 8, 8])
        pool_sizes = model_params.get("pool_sizes", [4, 4, 4])
        dropout_rates = model_params.get("dropout_rates", [0.2, 0.2, 0.5])
        
        model = DeepSEAModel(
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            pool_sizes=pool_sizes,
            dropout_rates=dropout_rates,
            num_targets=num_targets
        )
    
    elif model_type.lower() == "danq":
        from genomic_lightning.models.danq import DanQModel
        
        num_targets = model_params.get("num_targets", 919)
        num_filters = model_params.get("num_filters", 320)
        filter_size = model_params.get("filter_size", 26)
        pool_size = model_params.get("pool_size", 13)
        lstm_hidden = model_params.get("lstm_hidden", 320)
        lstm_layers = model_params.get("lstm_layers", 1)
        
        model = DanQModel(
            num_targets=num_targets,
            num_filters=num_filters,
            filter_size=filter_size,
            pool_size=pool_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers
        )
    
    elif model_type.lower() == "chromdragonn":
        from genomic_lightning.models.chromdragonn import ChromDragoNNModel
        
        num_targets = model_params.get("num_targets", 919)
        num_filters = model_params.get("num_filters", 300)
        filter_sizes = model_params.get("filter_sizes", [10, 15, 20])
        residual_blocks = model_params.get("residual_blocks", 3)
        
        model = ChromDragoNNModel(
            num_targets=num_targets,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            residual_blocks=residual_blocks
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # FuGEP models often have a specific structure in the state dict
    # Try to load directly first
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.warning(f"Direct loading failed: {str(e)}. Attempting to adapt state dict...")
        
        # Check if the state dict is inside a 'model' or 'state_dict' key
        if isinstance(state_dict, dict):
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
        # Adapt the state dict keys
        adapted_state_dict = {}
        
        # Common prefix differences
        fugep_prefixes = ["module.", "model.", "network."]
        
        for key, value in state_dict.items():
            # Try removing common prefixes
            adapted_key = key
            for prefix in fugep_prefixes:
                if key.startswith(prefix):
                    adapted_key = key[len(prefix):]
                    break
            
            # Try adding a common prefix if needed
            if adapted_key not in model.state_dict() and not any(adapted_key.startswith(p) for p in fugep_prefixes):
                for prefix in ["", "model.", "network."]:
                    if f"{prefix}{adapted_key}" in model.state_dict():
                        adapted_key = f"{prefix}{adapted_key}"
                        break
            
            # Add the adapted key if it exists in the model
            if adapted_key in model.state_dict():
                # Check if shapes match
                if value.shape == model.state_dict()[adapted_key].shape:
                    adapted_state_dict[adapted_key] = value
                else:
                    logger.warning(f"Shape mismatch for {adapted_key}: {value.shape} vs {model.state_dict()[adapted_key].shape}")
            else:
                logger.warning(f"Could not find matching key for {key}")
        
        # Try loading the adapted state dict
        model.load_state_dict(adapted_state_dict, strict=False)
        
        # Check how much of the model was successfully loaded
        missing_keys = set(model.state_dict().keys()) - set(adapted_state_dict.keys())
        if missing_keys:
            logger.warning(f"Warning: {len(missing_keys)}/{len(model.state_dict())} keys were not loaded: {list(missing_keys)[:5]}...")
    
    return model


def import_model_from_path(model_path: str, model_type: str = None, config_path: str = None) -> torch.nn.Module:
    """
    Unified function to import models from various sources based on file extension.
    
    Args:
        model_path: Path to the saved model
        model_type: Optional model type hint
        config_path: Optional path to configuration file
        
    Returns:
        PyTorch model loaded with weights
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine the format based on file extension
    if model_path.endswith('.ckpt'):
        # This is likely a PyTorch Lightning checkpoint
        from pytorch_lightning import LightningModule
        
        # Try to load as Lightning module
        try:
            if model_type and model_type.lower() == "deepsea":
                from genomic_lightning.lightning_modules.deepsea import DeepSEALightningModule
                model = DeepSEALightningModule.load_from_checkpoint(model_path)
            elif model_type and model_type.lower() == "danq":
                from genomic_lightning.lightning_modules.danq import DanQLightningModule
                model = DanQLightningModule.load_from_checkpoint(model_path)
            elif model_type and model_type.lower() == "chromdragonn":
                from genomic_lightning.lightning_modules.chromdragonn import ChromDragoNNLightningModule
                model = ChromDragoNNLightningModule.load_from_checkpoint(model_path)
            else:
                # Try to infer the model type from the checkpoint
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # Check if there's a clear class path
                if 'hyper_parameters' in checkpoint and 'model_class' in checkpoint['hyper_parameters']:
                    model_class_path = checkpoint['hyper_parameters']['model_class']
                    module_path, class_name = model_class_path.rsplit('.', 1)
                    
                    # Dynamically import the class
                    module = __import__(module_path, fromlist=[class_name])
                    model_class = getattr(module, class_name)
                    model = model_class.load_from_checkpoint(model_path)
                else:
                    # Default to a base lightning module
                    from genomic_lightning.lightning_modules.base import BaseLightningModule
                    model = BaseLightningModule.load_from_checkpoint(model_path)
        except Exception as e:
            logger.error(f"Error loading Lightning checkpoint: {str(e)}")
            raise
            
    elif model_path.endswith('.pth') or model_path.endswith('.pt'):
        # This could be a raw PyTorch model or a legacy model
        
        # Try to determine the source framework
        if config_path and config_path.endswith('.yml'):
            # This is likely a FuGEP model
            model = import_fugep_model(model_path, config_path)
        else:
            # Assume UAVarPrior model or raw PyTorch model
            model_type = model_type if model_type else "deepsea"
            model = import_uavarprior_model(model_path, model_type, config_path=config_path)
    else:
        raise ValueError(f"Unsupported model file format: {model_path}")
    
    return model