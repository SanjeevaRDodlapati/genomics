"""
Utilities for converting between different model formats.

This module provides functions to convert models between different formats,
such as PyTorch models to ONNX, TorchScript, or TensorFlow.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
import logging

logger = logging.getLogger(__name__)

def convert_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 4, 1000),
    dynamic_axes: Optional[Dict[str, List[int]]] = None,
    opset_version: int = 12,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None
) -> str:
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to convert
        output_path: Path to save the ONNX model
        input_shape: Shape of the input tensor (batch_size, channels, seq_length)
        dynamic_axes: Dictionary specifying dynamic axes for inputs/outputs
        opset_version: ONNX opset version
        input_names: Names for model inputs
        output_names: Names for model outputs
        
    Returns:
        Path to the saved ONNX model
    """
    if not output_path.endswith(".onnx"):
        output_path = output_path + ".onnx"
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, requires_grad=True)
    
    # Set default input/output names if not provided
    if input_names is None:
        input_names = ["input"]
    
    if output_names is None:
        output_names = ["output"]
    
    # Set default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "seq_length"},
            "output": {0: "batch_size"}
        }
    
    # Export the model
    torch.onnx.export(
        model,                     # model being run
        dummy_input,               # model input
        output_path,               # where to save the model
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=input_names,   # the model's input names
        output_names=output_names, # the model's output names
        dynamic_axes=dynamic_axes  # variable length axes
    )
    
    logger.info(f"Model exported to ONNX format at: {output_path}")
    return output_path


def convert_to_torchscript(
    model: nn.Module,
    output_path: str,
    method: str = "trace",
    example_input: Optional[torch.Tensor] = None,
    input_shape: Tuple[int, ...] = (1, 4, 1000)
) -> str:
    """
    Convert a PyTorch model to TorchScript format.
    
    Args:
        model: PyTorch model to convert
        output_path: Path to save the TorchScript model
        method: Conversion method ('trace' or 'script')
        example_input: Example input tensor for tracing
        input_shape: Shape of the input if example_input is not provided
        
    Returns:
        Path to the saved TorchScript model
    """
    if not output_path.endswith(".pt") and not output_path.endswith(".pth"):
        output_path = output_path + ".pt"
    
    # Set model to evaluation mode
    model.eval()
    
    # Create scripted/traced model
    if method.lower() == "trace":
        # Create dummy input if not provided
        if example_input is None:
            example_input = torch.randn(input_shape)
        
        # Trace the model
        ts_model = torch.jit.trace(model, example_input)
    
    elif method.lower() == "script":
        # Script the model
        ts_model = torch.jit.script(model)
    
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'trace' or 'script'")
    
    # Save the model
    ts_model.save(output_path)
    
    logger.info(f"Model exported to TorchScript format at: {output_path}")
    return output_path


def optimize_for_mobile(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 4, 1000)
) -> str:
    """
    Optimize a PyTorch model for mobile deployment.
    
    Args:
        model: PyTorch model to convert
        output_path: Path to save the optimized model
        input_shape: Shape of the input tensor
        
    Returns:
        Path to the saved optimized model
    """
    if not output_path.endswith(".pt") and not output_path.endswith(".pth"):
        output_path = output_path + ".pt"
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    example_input = torch.randn(input_shape)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_model = optimize_for_mobile(traced_model)
    
    # Save the model
    optimized_model.save(output_path)
    
    logger.info(f"Model optimized for mobile at: {output_path}")
    return output_path


def quantize_model(
    model: nn.Module,
    output_path: str,
    quantization_type: str = "static",
    input_shape: Tuple[int, ...] = (1, 4, 1000),
    calibration_data: Optional[torch.Tensor] = None
) -> str:
    """
    Quantize a PyTorch model for improved performance and reduced size.
    
    Args:
        model: PyTorch model to quantize
        output_path: Path to save the quantized model
        quantization_type: Type of quantization ('static', 'dynamic', 'qat')
        input_shape: Shape of the input tensor
        calibration_data: Data for static quantization calibration
        
    Returns:
        Path to the saved quantized model
    """
    if not output_path.endswith(".pt") and not output_path.endswith(".pth"):
        output_path = output_path + ".pt"
    
    # Set model to evaluation mode
    model.eval()
    
    if quantization_type.lower() == "dynamic":
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.Conv1d}, 
            dtype=torch.qint8
        )
    
    elif quantization_type.lower() == "static":
        # Prepare model for static quantization
        model_fp32 = model
        
        # Fuse modules where applicable
        # Example: fuse Conv+BN+ReLU
        for m in model_fp32.modules():
            if hasattr(m, "fuse_model"):
                m.fuse_model()
        
        # Prepare for static quantization
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_fp32_prepared = torch.quantization.prepare(model_fp32)
        
        # Calibrate with sample data
        if calibration_data is not None:
            with torch.no_grad():
                for data in calibration_data:
                    model_fp32_prepared(data)
        else:
            # Use random data if no calibration data is provided
            dummy_input = torch.randn(input_shape)
            with torch.no_grad():
                model_fp32_prepared(dummy_input)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_fp32_prepared)
    
    elif quantization_type.lower() == "qat":
        # Quantization-aware training
        # This requires training the model with quantization awareness
        raise NotImplementedError("Quantization-aware training requires model training and is not implemented in this function")
    
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    # Create TorchScript representation
    quantized_script = torch.jit.script(quantized_model)
    
    # Save the model
    quantized_script.save(output_path)
    
    logger.info(f"Quantized model saved at: {output_path}")
    return output_path


def convert_lightning_to_pytorch(
    lightning_model_path: str,
    output_path: str,
    model_only: bool = True
) -> str:
    """
    Extract a PyTorch model from a PyTorch Lightning checkpoint.
    
    Args:
        lightning_model_path: Path to the Lightning checkpoint
        output_path: Path to save the PyTorch model
        model_only: Whether to extract only the model or the full Lightning module
        
    Returns:
        Path to the saved PyTorch model
    """
    from pytorch_lightning import LightningModule
    
    if not os.path.exists(lightning_model_path):
        raise FileNotFoundError(f"Lightning model not found: {lightning_model_path}")
        
    if not output_path.endswith(".pt") and not output_path.endswith(".pth"):
        output_path = output_path + ".pt"
    
    # Load the Lightning model
    checkpoint = torch.load(lightning_model_path, map_location=torch.device('cpu'))
    
    if model_only:
        # Extract only the model state dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            
            # Remove Lightning module prefixes
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    # Remove 'model.' prefix
                    cleaned_key = key[6:]
                    cleaned_state_dict[cleaned_key] = value
            
            # Save just the model state dict
            torch.save(cleaned_state_dict, output_path)
            
        else:
            raise KeyError("Could not find 'state_dict' in the Lightning checkpoint")
    else:
        # Save the entire checkpoint
        torch.save(checkpoint, output_path)
    
    logger.info(f"Model extracted from Lightning checkpoint and saved at: {output_path}")
    return output_path