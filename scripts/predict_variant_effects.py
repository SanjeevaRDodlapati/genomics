#!/usr/bin/env python
"""
Example script for genomic variant effect prediction using GenomicLightning models.

This script demonstrates:
1. Loading a GenomicLightning model
2. Predicting the effect of variants from a VCF file
3. Analyzing and visualizing variant effects
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from genomic_lightning.models.deepsea import DeepSEA
from genomic_lightning.models.danq import DanQModel
from genomic_lightning.models.chromdragonn import ChromDragoNNModel
from genomic_lightning.variant_analysis.variant_effect import (
    VariantSequenceExtractor,
    VariantEffectPredictor,
    VariantAnalyzer
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict and analyze variant effects using GenomicLightning models"
    )
    
    # Input files
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the GenomicLightning model checkpoint"
    )
    
    parser.add_argument(
        "--vcf_path",
        type=str,
        required=True,
        help="Path to the VCF file containing variants"
    )
    
    parser.add_argument(
        "--reference_genome",
        type=str,
        required=True,
        help="Path to the reference genome FASTA file"
    )
    
    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="DeepSEA",
        choices=["DeepSEA", "DanQ", "ChromDragoNN"],
        help="Model architecture type"
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
    
    # Analysis parameters
    parser.add_argument(
        "--max_variants",
        type=int,
        default=None,
        help="Maximum number of variants to process (default: all)"
    )
    
    parser.add_argument(
        "--feature_names_file",
        type=str,
        default=None,
        help="Path to a text file with feature names (one per line)"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="variant_analysis",
        help="Directory to save results (default: 'variant_analysis')"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run prediction on"
    )
    
    return parser.parse_args()


def load_model(args):
    """
    Load model from checkpoint.
    
    Args:
        args: Command line arguments
        
    Returns:
        Loaded model
    """
    # Try to load as a GenomicLightning checkpoint first
    try:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        
        # Create model based on type information in checkpoint
        if "model_type" in checkpoint:
            args.model_type = checkpoint["model_type"]
            
        if "n_outputs" in checkpoint:
            args.n_outputs = checkpoint["n_outputs"]
    except:
        print("Could not load checkpoint metadata, using provided arguments.")
        
    # Create model based on type
    if args.model_type == "DeepSEA":
        model = DeepSEA(
            sequence_length=args.sequence_length,
            n_genomic_features=args.n_genomic_features,
            n_outputs=args.n_outputs
        )
    elif args.model_type == "DanQ":
        model = DanQModel(
            sequence_length=args.sequence_length,
            n_genomic_features=args.n_genomic_features,
            n_outputs=args.n_outputs
        )
    elif args.model_type == "ChromDragoNN":
        model = ChromDragoNNModel(
            sequence_length=args.sequence_length,
            n_genomic_features=args.n_genomic_features,
            n_outputs=args.n_outputs
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    # Load weights
    try:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        raise ValueError(f"Failed to load model weights: {e}")
        
    return model


def load_feature_names(path):
    """
    Load feature names from a text file.
    
    Args:
        path: Path to text file with feature names
        
    Returns:
        List of feature names
    """
    if not path or not os.path.exists(path):
        return None
        
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    """Main function."""
    args = parse_arguments()
    
    print("=== GenomicLightning Variant Effect Prediction ===")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args)
    model.to(args.device)
    model.eval()
    
    # Load feature names if provided
    feature_names = load_feature_names(args.feature_names_file)
    
    # Create sequence extractor
    print(f"Initializing sequence extractor with reference genome: {args.reference_genome}")
    extractor = VariantSequenceExtractor(args.reference_genome)
    
    # Create variant effect predictor
    predictor = VariantEffectPredictor(
        model=model,
        sequence_extractor=extractor,
        window_size=args.sequence_length,
        device=args.device
    )
    
    # Run prediction
    print(f"Predicting variant effects from {args.vcf_path}")
    prediction_file = output_dir / "variant_predictions.csv"
    df = predictor.predict_from_vcf(
        vcf_path=args.vcf_path,
        output_path=str(prediction_file),
        max_variants=args.max_variants
    )
    
    print(f"Saved predictions to {prediction_file}")
    
    # Create analyzer
    analyzer = VariantAnalyzer(df, feature_names=feature_names)
    
    # Get and save top affected features
    top_features = analyzer.get_top_affected_features(n=20)
    top_features_file = output_dir / "top_affected_features.csv"
    top_features.to_csv(top_features_file, index=False)
    print(f"Saved top affected features to {top_features_file}")
    
    # Plot feature effects
    fig = analyzer.plot_feature_effects(n_features=15)
    fig_file = output_dir / "feature_effects.png"
    fig.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"Saved feature effects plot to {fig_file}")
    plt.close(fig)
    
    # Find top variants
    if feature_names:
        # If we have feature names, use the top feature name
        top_feature = top_features["feature"].iloc[0]
    else:
        # Otherwise use the feature index
        top_feature = 0
        
    top_variants = analyzer.get_variants_by_feature_effect(
        feature=top_feature,
        top_n=20,
        abs_effect=True
    )
    top_variants_file = output_dir / f"top_variants_for_{top_feature}.csv"
    top_variants.to_csv(top_variants_file, index=False)
    print(f"Saved top variants for feature {top_feature} to {top_variants_file}")
    
    # Plot effect for top variant
    if not top_variants.empty:
        top_variant_id = top_variants["variant_id"].iloc[0]
        fig = analyzer.plot_variant_effect(variant_id=top_variant_id)
        variant_fig_file = output_dir / f"variant_effect_{top_variant_id.replace(':', '_')}.png"
        fig.savefig(variant_fig_file, dpi=300, bbox_inches='tight')
        print(f"Saved effect plot for variant {top_variant_id} to {variant_fig_file}")
        plt.close(fig)
    
    print("\nAnalysis complete!")
    print(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
