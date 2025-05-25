"""Utilities for working with UAVarPrior or FuGEP samplers."""

import importlib
import sys
from typing import Dict, Any, Optional, Union, Callable


def import_sampler_class(path: str) -> type:
    """Import a sampler class from a module path.
    
    Args:
        path: Full import path to the sampler class (e.g., 'uavarprior.samplers.IntervalsSampler')
        
    Returns:
        Sampler class
        
    Raises:
        ImportError: If the sampler class cannot be imported
    """
    try:
        # Split module path and class name
        module_path, class_name = path.rsplit('.', 1)
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the class
        sampler_class = getattr(module, class_name)
        
        return sampler_class
    
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import sampler class '{path}': {str(e)}")


def create_sampler(config: Dict[str, Any]) -> Any:
    """Create a sampler instance from configuration.
    
    This function supports several ways to specify a sampler:
    1. Using a 'class' key with the import path of the sampler class
    2. Using a '!obj' YAML tag in the config file
    
    Args:
        config: Sampler configuration dictionary
        
    Returns:
        Sampler instance
        
    Raises:
        ValueError: If the sampler cannot be created from the config
    """
    # Check if the config has a 'class' key
    if 'class' in config:
        # Get the sampler class
        sampler_class = import_sampler_class(config['class'])
        
        # Extract arguments
        args = config.get('args', {})
        
        # Create sampler instance
        return sampler_class(**args)
    
    # Check if this is a direct instance reference (UAVarPrior/FuGEP style)
    if isinstance(config, dict) and len(config) == 1 and list(config.keys())[0].startswith('!obj:'):
        # Extract class path and arguments
        obj_spec = list(config.keys())[0]
        class_path = obj_spec.replace('!obj:', '')
        args = list(config.values())[0]
        
        # Import the class
        sampler_class = import_sampler_class(class_path)
        
        # Create sampler instance
        return sampler_class(**args)
    
    # If we have a direct sampler instance, use it
    if not isinstance(config, dict):
        return config
    
    # Otherwise, we don't know how to create the sampler
    raise ValueError(
        f"Cannot create sampler from config: {config}. "
        f"Config should have a 'class' key or use YAML '!obj:' tag."
    )
