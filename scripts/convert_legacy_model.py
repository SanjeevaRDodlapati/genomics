#!/usr/bin/env python
"""
Script to convert models from legacy UAVarPrior/FuGEP frameworks to GenomicLightning.

This script:
1. Detects model architecture from the legacy codebase
2. Creates a corresponding GenomicLightning model
3. Transfers weights from the legacy model to the new model
4. Validates the converted model by running test predictions
5. Saves the converted model in GenomicLightning format
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add GenomicLightning to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomic_lightning.utils.model_conversion import convert_model_to_genomic_lightning
from genomic_lightning.models.deepsea import DeepSEA
from genomic_lightning.models.danq import DanQModel
from genomic_lightning.models.chromdragonn import ChromDragoNNModel


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert UAVarPrior/FuGEP models to GenomicLightning format"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the legacy model checkpoint"
    )
    
    parser.add_argument(
        "--legacy_code_path",
        type=str,
        required=True,
        help="Path to the legacy code installation (UAVarPrior or FuGEP)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the converted model (default: 'converted_models/{model_name}.pt')"
    )
    
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Input sequence length (default: 1000)"
    )
    
    parser.add_argument(
        "--n_genomic_features",
        type=int,
        default=4,
        help="Number of genomic features (default: 4 for A,C,G,T)"
    )
    
    parser.add_argument(
        "--n_outputs",
        type=int,
        default=919,
        help="Number of output features (default: 919 for DeepSEA)"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["DeepSEA", "DanQ", "ChromDragoNN"],
        help="Override detected model type"
    )
    
    parser.add_argument(
        "--test_conversion",
        action="store_true",
        help="Run test prediction to validate conversion"
    )
    
    return parser.parse_args()


def validate_conversion(
    legacy_model_path,
    legacy_code_path,
    converted_model_path,
    sequence_length=1000,
    n_genomic_features=4
):
    """
    Validate model conversion by comparing predictions between original and converted models.
    
    Args:
        legacy_model_path: Path to legacy model checkpoint
        legacy_code_path: Path to legacy code installation
        converted_model_path: Path to converted model
        sequence_length: Input sequence length
        n_genomic_features: Number of genomic features
        
    Returns:
        Tuple of (correlation, max_difference)
    """
    print("\n--- Validating model conversion ---")
    
    # Create random test input
    test_batch_size = 5
    test_input = torch.rand(test_batch_size, n_genomic_features, sequence_length)
    
    # Load GenomicLightning model
    checkpoint = torch.load(converted_model_path, map_location="cpu")
    model_type = checkpoint.get("model_type", "DeepSEA")
    
    if model_type == "DeepSEA":
        gl_model = DeepSEA(
            sequence_length=sequence_length,
            n_genomic_features=n_genomic_features,
            n_outputs=checkpoint.get("n_outputs", 919)
        )
    elif model_type == "DanQ":
        gl_model = DanQModel(
            sequence_length=sequence_length,
            n_genomic_features=n_genomic_features,
            n_outputs=checkpoint.get("n_outputs", 919)
        )
    elif model_type == "ChromDragoNN":
        gl_model = ChromDragoNNModel(
            sequence_length=sequence_length,
            n_genomic_features=n_genomic_features,
            n_outputs=checkpoint.get("n_outputs", 919)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    gl_model.load_state_dict(checkpoint["state_dict"])
    gl_model.eval()
    
    # Get predictions from the GenomicLightning model
    with torch.no_grad():
        gl_preds = gl_model(test_input).numpy()
    
    # Now try to load and run the legacy model
    try:
        # Add legacy code path to sys.path
        if legacy_code_path not in sys.path:
            sys.path.insert(0, legacy_code_path)
        
        # Try different approaches to load the legacy model
        legacy_preds = None
        
        # Approach 1: Try to load as a standard PyTorch checkpoint
        try:
            checkpoint = torch.load(legacy_model_path, map_location="cpu")
            if "fugep" in legacy_code_path.lower():
                from fugep.models.deepsea import DeepSEA as LegacyModel
            else:
                from uavarprior.models.deepsea import DeepSEA as LegacyModel
                
            legacy_model = LegacyModel(
                sequence_length=sequence_length,
                n_genomic_features=n_genomic_features
            )
            
            if "state_dict" in checkpoint:
                legacy_model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                legacy_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                legacy_model.load_state_dict(checkpoint)
                
            legacy_model.eval()
            with torch.no_grad():
                legacy_preds = legacy_model(test_input).numpy()
                
        except Exception as e:
            print(f"Approach 1 failed: {e}")
            
        # If that failed, try a different approach
        if legacy_preds is None:
            print("Trying alternative approach...")
            try:
                # Try to import the module containing the model class
                if "fugep" in legacy_code_path.lower():
                    from fugep.models import load_model
                else:
                    from uavarprior.models import load_model
                    
                legacy_model = load_model(legacy_model_path)
                legacy_model.eval()
                
                with torch.no_grad():
                    legacy_preds = legacy_model(test_input).numpy()
                    
            except Exception as e:
                print(f"Approach 2 failed: {e}")
                
        # If we have legacy predictions, compare them
        if legacy_preds is not None:
            # Calculate correlation and max difference
            correlations = []
            for i in range(test_batch_size):
                corr = np.corrcoef(
                    gl_preds[i].flatten(),
                    legacy_preds[i].flatten()
                )[0, 1]
                correlations.append(corr)
                
            mean_correlation = np.mean(correlations)
            max_difference = np.max(np.abs(gl_preds - legacy_preds))
            
            print(f"Mean correlation: {mean_correlation:.4f}")
            print(f"Max difference: {max_difference:.6f}")
            
            if mean_correlation > 0.99:
                print("✅ Conversion successful: High correlation between models")
            else:
                print("⚠️ Conversion may have issues: Low correlation between models")
                
            return mean_correlation, max_difference
            
    except Exception as e:
        print(f"Failed to run legacy model for validation: {e}")
        
    print("⚠️ Could not validate conversion with legacy model")
    return None, None


def main():
    """Main function."""
    args = parse_arguments()
    
    print("=== GenomicLightning Model Converter ===")
    print(f"Converting model: {args.model_path}")
    print(f"Legacy code path: {args.legacy_code_path}")
    
    # Set default output path if not provided
    if args.output_path is None:
        model_name = os.path.basename(args.model_path).split(".")[0]
        output_dir = Path("converted_models")
        output_dir.mkdir(exist_ok=True)
        args.output_path = str(output_dir / f"{model_name}_genomic_lightning.pt")
        
    print(f"Output path: {args.output_path}")
    
    # Convert the model
    converted_path = convert_model_to_genomic_lightning(
        legacy_model_path=args.model_path,
        legacy_code_path=args.legacy_code_path,
        output_path=args.output_path,
        sequence_length=args.sequence_length,
        n_genomic_features=args.n_genomic_features,
        n_outputs=args.n_outputs,
        model_type=args.model_type
    )
    
    print(f"\n✅ Model successfully converted and saved to {converted_path}")
    
    # Validate the conversion if requested
    if args.test_conversion:
        validate_conversion(
            legacy_model_path=args.model_path,
            legacy_code_path=args.legacy_code_path,
            converted_model_path=converted_path,
            sequence_length=args.sequence_length,
            n_genomic_features=args.n_genomic_features
        )


if __name__ == "__main__":
    main()
