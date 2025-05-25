"""
Utility functions to convert models between UAVarPrior/FuGEP and GenomicLightning.
"""

import torch
import numpy as np
import sys
import os
import importlib.util
from typing import Dict, Any, Optional, Union, Tuple, List

from genomic_lightning.models.deepsea import DeepSEA
from genomic_lightning.models.danq import DanQModel
from genomic_lightning.models.chromdragonn import ChromDragoNNModel


def import_module_from_path(module_path: str, module_name: str = None):
    """
    Import a Python module from filesystem path.
    
    Args:
        module_path: Path to Python module file
        module_name: Name to give the imported module (defaults to filename)
        
    Returns:
        Imported module object
    """
    if module_name is None:
        module_name = os.path.basename(module_path).replace(".py", "")
        
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {module_path}")
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_fugep_model_class(fugep_path: str) -> Tuple[Any, str]:
    """
    Find the model class in a FuGEP installation.
    
    Args:
        fugep_path: Path to FuGEP installation
        
    Returns:
        Tuple of (model_class, model_name)
    """
    models_path = os.path.join(fugep_path, "fugep", "models")
    if not os.path.isdir(models_path):
        raise FileNotFoundError(f"FuGEP models directory not found at {models_path}")
        
    # Try to find DeepSEA model
    deepsea_path = os.path.join(models_path, "deepsea.py")
    if os.path.isfile(deepsea_path):
        deepsea_module = import_module_from_path(deepsea_path)
        model_classes = [obj for name, obj in deepsea_module.__dict__.items() 
                        if isinstance(obj, type) and "deepsea" in name.lower()]
        if model_classes:
            return model_classes[0], "DeepSEA"
            
    # Try to find other models
    model_files = [f for f in os.listdir(models_path) if f.endswith(".py")]
    for model_file in model_files:
        model_path = os.path.join(models_path, model_file)
        model_module = import_module_from_path(model_path)
        model_classes = [obj for name, obj in model_module.__dict__.items() 
                        if isinstance(obj, type) and hasattr(obj, "forward")]
                        
        if model_classes:
            model_name = model_file.replace(".py", "").title()
            return model_classes[0], model_name
            
    raise ValueError(f"No model classes found in {models_path}")


def find_uavarprior_model_class(uavarprior_path: str) -> Tuple[Any, str]:
    """
    Find the model class in a UAVarPrior installation.
    
    Args:
        uavarprior_path: Path to UAVarPrior installation
        
    Returns:
        Tuple of (model_class, model_name)
    """
    models_path = os.path.join(uavarprior_path, "uavarprior", "models")
    if not os.path.isdir(models_path):
        raise FileNotFoundError(f"UAVarPrior models directory not found at {models_path}")
        
    # Try to find DeepSEA model first
    deepsea_path = os.path.join(models_path, "deepsea.py")
    if os.path.isfile(deepsea_path):
        deepsea_module = import_module_from_path(deepsea_path)
        model_classes = [obj for name, obj in deepsea_module.__dict__.items() 
                        if isinstance(obj, type) and "deepsea" in name.lower()]
        if model_classes:
            return model_classes[0], "DeepSEA"
    
    # Try to find other models
    model_files = [f for f in os.listdir(models_path) if f.endswith(".py")]
    for model_file in model_files:
        model_path = os.path.join(models_path, model_file)
        model_module = import_module_from_path(model_path)
        model_classes = [obj for name, obj in model_module.__dict__.items() 
                        if isinstance(obj, type) and hasattr(obj, "forward")]
                        
        if model_classes:
            model_name = model_file.replace(".py", "").title()
            return model_classes[0], model_name
            
    raise ValueError(f"No model classes found in {models_path}")


def get_corresponding_genomic_lightning_model(legacy_model_name: str):
    """
    Get the corresponding GenomicLightning model class based on a legacy model name.
    
    Args:
        legacy_model_name: Name of legacy model
        
    Returns:
        GenomicLightning model class
    """
    # Map legacy model names to GenomicLightning models
    model_map = {
        "DeepSEA": DeepSEA,
        "DanQ": DanQModel,
        "ChromDragonn": ChromDragoNNModel,
    }
    
    for model_key in model_map:
        if model_key.lower() in legacy_model_name.lower():
            return model_map[model_key]
            
    # Default to DeepSEA if no match found
    print(f"WARNING: No direct GenomicLightning model match for {legacy_model_name}. Using DeepSEA as default.")
    return DeepSEA


def load_legacy_model_weights(
    legacy_model_path: str,
    target_model: torch.nn.Module,
    mapping_dict: Optional[Dict[str, str]] = None
) -> torch.nn.Module:
    """
    Load weights from a legacy model checkpoint into a GenomicLightning model.
    
    Args:
        legacy_model_path: Path to legacy model checkpoint
        target_model: GenomicLightning model to load weights into
        mapping_dict: Dictionary mapping legacy parameter names to target names
        
    Returns:
        Updated target model with loaded weights
    """
    # Load legacy checkpoint
    checkpoint = torch.load(legacy_model_path, map_location="cpu")
    
    # Extract state dict (handle different formats)
    if "state_dict" in checkpoint:
        legacy_state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        legacy_state_dict = checkpoint["model_state_dict"]
    else:
        # Assume checkpoint is the state dict itself
        legacy_state_dict = checkpoint
    
    # Create mapping between legacy and target model parameters
    if mapping_dict is None:
        mapping_dict = {}
        for legacy_name, _ in legacy_state_dict.items():
            # Try to find a matching parameter in target model
            target_name = legacy_name
            if "module." in target_name:
                target_name = target_name.replace("module.", "")
                
            if target_name in target_model.state_dict():
                mapping_dict[legacy_name] = target_name
    
    # Create new state dict for target model
    target_state_dict = target_model.state_dict()
    loaded_params = 0
    
    for legacy_name, target_name in mapping_dict.items():
        if legacy_name in legacy_state_dict and target_name in target_state_dict:
            legacy_param = legacy_state_dict[legacy_name]
            target_param_shape = target_state_dict[target_name].shape
            
            # Check if shapes match
            if legacy_param.shape == target_param_shape:
                target_state_dict[target_name] = legacy_param
                loaded_params += 1
            else:
                print(f"WARNING: Shape mismatch for {legacy_name} -> {target_name}: "
                     f"{legacy_param.shape} vs {target_param_shape}")
    
    # Load the updated state dict
    target_model.load_state_dict(target_state_dict, strict=False)
    
    print(f"Loaded {loaded_params}/{len(target_state_dict)} parameters from legacy model")
    return target_model


def convert_model_to_genomic_lightning(
    legacy_model_path: str,
    legacy_code_path: str,
    output_path: str,
    sequence_length: int = 1000,
    n_genomic_features: int = 4,
    n_outputs: int = 919,
    model_type: Optional[str] = None
) -> str:
    """
    Convert a legacy model from UAVarPrior/FuGEP to GenomicLightning format.
    
    Args:
        legacy_model_path: Path to legacy model checkpoint
        legacy_code_path: Path to legacy code installation
        output_path: Path to save converted model
        sequence_length: Length of input sequences
        n_genomic_features: Number of genomic features
        n_outputs: Number of output features
        model_type: Override detected model type
        
    Returns:
        Path to saved converted model
    """
    # Find the legacy model class
    try:
        if "fugep" in legacy_code_path.lower():
            LegacyModelClass, detected_model_name = find_fugep_model_class(legacy_code_path)
        else:
            LegacyModelClass, detected_model_name = find_uavarprior_model_class(legacy_code_path)
    except Exception as e:
        print(f"ERROR finding legacy model class: {e}")
        if model_type is None:
            raise ValueError(f"Could not detect legacy model type and no model_type provided")
        detected_model_name = model_type
    
    # Use provided model type or detected one
    model_type = model_type or detected_model_name
    
    # Get corresponding GenomicLightning model
    ModelClass = get_corresponding_genomic_lightning_model(model_type)
    
    # Create the target model
    target_model = ModelClass(
        sequence_length=sequence_length,
        n_genomic_features=n_genomic_features,
        n_outputs=n_outputs
    )
    
    # Load weights from legacy model
    target_model = load_legacy_model_weights(
        legacy_model_path=legacy_model_path,
        target_model=target_model
    )
    
    # Save the converted model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "state_dict": target_model.state_dict(),
        "model_type": model_type,
        "original_path": legacy_model_path,
        "sequence_length": sequence_length,
        "n_genomic_features": n_genomic_features,
        "n_outputs": n_outputs
    }, output_path)
    
    return output_path
