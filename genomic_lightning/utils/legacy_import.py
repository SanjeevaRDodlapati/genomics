"""Utilities for importing models from UAVarPrior or FuGEP."""

import importlib
import sys
from typing import Dict, Any, Optional, Union, Callable


def import_uavarprior_model(model_name: str = "DeepSEA"):
    """Import a model from UAVarPrior.
    
    Args:
        model_name: Name of the model to import
        
    Returns:
        Imported model class
        
    Raises:
        ImportError: If the model cannot be imported
    """
    try:
        # Try to import from UAVarPrior
        module = importlib.import_module("uavarprior.model")
        
        # Get model initialization function
        initialize_model = getattr(module, "initialize_model", None)
        if initialize_model is not None:
            return initialize_model
        
        # If initialize_model is not found, try to get the model class directly
        model_class = getattr(module, model_name, None)
        if model_class is not None:
            return model_class
        
        # Last resort: try to find the model in uavarprior.model.models
        models_module = importlib.import_module("uavarprior.model.models")
        model_class = getattr(models_module, model_name, None)
        if model_class is not None:
            return model_class
            
        raise ImportError(f"Could not find model {model_name} in UAVarPrior")
    
    except ImportError as e:
        raise ImportError(f"Failed to import model {model_name} from UAVarPrior: {str(e)}")


def import_fugep_model(model_name: str = "DeepSEA"):
    """Import a model from FuGEP.
    
    Args:
        model_name: Name of the model to import
        
    Returns:
        Imported model class
        
    Raises:
        ImportError: If the model cannot be imported
    """
    try:
        # Try to import from FuGEP
        module = importlib.import_module("fugep.model")
        
        # Get model initialization function
        initialize_model = getattr(module, "initialize_model", None)
        if initialize_model is not None:
            return initialize_model
        
        # If initialize_model is not found, try to get the model class directly
        model_class = getattr(module, model_name, None)
        if model_class is not None:
            return model_class
        
        # Last resort: try to find the model in fugep.model.models
        models_module = importlib.import_module("fugep.model.models")
        model_class = getattr(models_module, model_name, None)
        if model_class is not None:
            return model_class
            
        raise ImportError(f"Could not find model {model_name} in FuGEP")
    
    except ImportError as e:
        raise ImportError(f"Failed to import model {model_name} from FuGEP: {str(e)}")


def initialize_model_from_legacy(
    config: Dict[str, Any],
    framework: str = "uavarprior"
) -> Any:
    """Initialize a model from UAVarPrior or FuGEP based on configuration.
    
    Args:
        config: Model configuration dictionary
        framework: Which framework to import from ('uavarprior' or 'fugep')
        
    Returns:
        Initialized model
        
    Raises:
        ImportError: If the model cannot be imported
        ValueError: If the framework is not recognized
    """
    if framework.lower() == "uavarprior":
        initialize_fn = import_uavarprior_model()
    elif framework.lower() == "fugep":
        initialize_fn = import_fugep_model()
    else:
        raise ValueError(f"Unknown framework: {framework}. Use 'uavarprior' or 'fugep'.")
    
    # Initialize the model
    if callable(initialize_fn):
        return initialize_fn(config)
    
    # If initialize_fn is a class, not a function
    return initialize_fn(**config.get("args", {}))
