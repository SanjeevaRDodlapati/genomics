"""
Utilities for genomic variant analysis with VCF files.

This module provides functionality to:
1. Load genomic variants from VCF files
2. Extract sequences around variants
3. Predict effects of variants using genomic models
4. Analyze and visualize variant impact scores
"""

import os
import pysam
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
from pyfaidx import Fasta
import matplotlib.pyplot as plt
import seaborn as sns


class VariantSequenceExtractor:
    """
    Extract genomic sequences around variants from reference genome.
    """
    
    def __init__(self, reference_genome_path: str):
        """
        Initialize the sequence extractor.
        
        Args:
            reference_genome_path: Path to reference genome FASTA file
        """
        self.genome = Fasta(reference_genome_path)
        
    def extract_sequence(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        window_size: int = 1000,
        center: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract sequence around a variant.
        
        Args:
            chrom: Chromosome name
            pos: 1-based position of the variant
            ref: Reference allele
            alt: Alternate allele
            window_size: Size of sequence window to extract
            center: Whether to center the variant in the window
            
        Returns:
            Dict with "ref_seq" and "alt_seq" as one-hot encoded sequences
        """
        # Adjust chromosome name if needed (remove "chr" prefix if not in reference)
        if chrom not in self.genome and chrom.startswith("chr"):
            chrom = chrom[3:]
        elif chrom not in self.genome and not chrom.startswith("chr"):
            chrom = "chr" + chrom
            
        if chrom not in self.genome:
            raise ValueError(f"Chromosome {chrom} not found in reference genome")
            
        # Calculate window coordinates
        pos = int(pos)  # Convert to 0-based
        if center:
            start = pos - window_size // 2
            end = start + window_size
        else:
            start = pos - window_size + len(ref)
            end = pos + window_size
            
        # Ensure coordinates are valid
        start = max(0, start)
        
        # Extract reference sequence
        ref_sequence = str(self.genome[chrom][start:end]).upper()
        
        # Create alternate sequence by replacing the reference allele with the alternate
        variant_position = pos - start
        
        if variant_position < 0 or variant_position >= len(ref_sequence):
            raise ValueError(f"Variant position {pos} is outside extracted sequence window")
            
        # Validate that ref allele matches the reference genome
        genome_ref = ref_sequence[variant_position:variant_position + len(ref)]
        if genome_ref != ref.upper():
            raise ValueError(
                f"Reference allele mismatch: variant ref={ref}, "
                f"genome ref={genome_ref} at {chrom}:{pos}"
            )
            
        # Create alt sequence by replacing ref with alt
        alt_sequence = (
            ref_sequence[:variant_position] +
            alt.upper() +
            ref_sequence[variant_position + len(ref):]
        )
        
        # Ensure both sequences are the same length by trimming or padding
        target_len = window_size
        if len(ref_sequence) < target_len:
            ref_sequence = ref_sequence + "N" * (target_len - len(ref_sequence))
        else:
            ref_sequence = ref_sequence[:target_len]
            
        if len(alt_sequence) < target_len:
            alt_sequence = alt_sequence + "N" * (target_len - len(alt_sequence))
        else:
            alt_sequence = alt_sequence[:target_len]
            
        # One-hot encode sequences
        ref_one_hot = self.one_hot_encode(ref_sequence)
        alt_one_hot = self.one_hot_encode(alt_sequence)
        
        return {
            "ref_seq": ref_one_hot,
            "alt_seq": alt_one_hot
        }
    
    def one_hot_encode(self, sequence: str) -> np.ndarray:
        """
        One-hot encode a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            One-hot encoded sequence as numpy array
        """
        sequence = sequence.upper()
        mapping = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
            "N": [0.25, 0.25, 0.25, 0.25]  # Uniform distribution for N
        }
        
        encoded = []
        for nucleotide in sequence:
            encoded.append(mapping.get(nucleotide, mapping["N"]))
            
        # Shape: (len(sequence), 4) - transpose to get (4, len(sequence))
        return np.array(encoded).T


class VariantEffectPredictor:
    """
    Predict the effect of genomic variants using a deep learning model.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        sequence_extractor: VariantSequenceExtractor,
        window_size: int = 1000,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize the variant effect predictor.
        
        Args:
            model: PyTorch model for prediction
            sequence_extractor: VariantSequenceExtractor instance
            window_size: Size of sequence window
            batch_size: Batch size for predictions
            device: Device to run predictions on
        """
        self.model = model.to(device)
        self.model.eval()
        self.extractor = sequence_extractor
        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device
        
    def predict_variant_effect(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str
    ) -> Dict[str, Any]:
        """
        Predict the effect of a single variant.
        
        Args:
            chrom: Chromosome name
            pos: Position of the variant
            ref: Reference allele
            alt: Alternate allele
            
        Returns:
            Dictionary with prediction results
        """
        # Extract sequences
        sequences = self.extractor.extract_sequence(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            window_size=self.window_size
        )
        
        # Convert to torch tensors
        ref_seq = torch.tensor(sequences["ref_seq"]).float().unsqueeze(0)  # Add batch dimension
        alt_seq = torch.tensor(sequences["alt_seq"]).float().unsqueeze(0)
        
        # Make predictions
        with torch.no_grad():
            ref_pred = self.model(ref_seq.to(self.device)).cpu().numpy()[0]
            alt_pred = self.model(alt_seq.to(self.device)).cpu().numpy()[0]
            
        # Calculate difference
        diff = alt_pred - ref_pred
        
        return {
            "variant_id": f"{chrom}:{pos}_{ref}>{alt}",
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "ref_predictions": ref_pred,
            "alt_predictions": alt_pred,
            "diff": diff,
            "abs_diff": np.abs(diff),
            "max_effect_index": np.argmax(np.abs(diff)),
            "max_effect": diff[np.argmax(np.abs(diff))]
        }
    
    def predict_from_vcf(
        self,
        vcf_path: str,
        output_path: Optional[str] = None,
        max_variants: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Predict effects for variants in a VCF file.
        
        Args:
            vcf_path: Path to VCF file
            output_path: Path to save results (optional)
            max_variants: Maximum number of variants to process
            
        Returns:
            DataFrame with prediction results
        """
        # Open VCF file
        vcf = pysam.VariantFile(vcf_path)
        
        # Create empty lists to store results
        results = []
        counter = 0
        
        # Process variants
        for variant in vcf:
            chrom = variant.chrom
            pos = variant.pos
            ref = variant.ref
            
            # Process each alternate allele
            for alt in variant.alts:
                try:
                    result = self.predict_variant_effect(chrom, pos, ref, alt)
                    results.append(result)
                except Exception as e:
                    print(f"Failed to process variant {chrom}:{pos} {ref}>{alt}: {e}")
                    
                counter += 1
                if counter % 100 == 0:
                    print(f"Processed {counter} variants")
                    
                if max_variants is not None and counter >= max_variants:
                    break
                    
            if max_variants is not None and counter >= max_variants:
                break
                
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save to file if requested
        if output_path:
            df.to_csv(output_path, index=False)
            
        return df


class VariantAnalyzer:
    """
    Analyze and visualize variant effect predictions.
    """
    
    def __init__(self, prediction_df: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Initialize the variant analyzer.
        
        Args:
            prediction_df: DataFrame with variant predictions
            feature_names: Names of predicted features
        """
        self.df = prediction_df
        self.feature_names = feature_names
        
        # Generate feature indices if names not provided
        if feature_names is None:
            if "ref_predictions" in self.df.columns:
                n_features = len(self.df["ref_predictions"].iloc[0])
                self.feature_names = [f"Feature_{i}" for i in range(n_features)]
                
    def get_top_affected_features(self, n: int = 10) -> pd.DataFrame:
        """
        Get top affected features across all variants.
        
        Args:
            n: Number of top features to return
            
        Returns:
            DataFrame with top affected features
        """
        # Expand diff column into separate columns for each feature
        diff_df = pd.DataFrame(
            np.vstack(self.df["diff"].values),
            columns=self.feature_names
        )
        
        # Calculate mean absolute effect for each feature
        mean_abs_effect = diff_df.abs().mean()
        
        # Get top features
        top_features = mean_abs_effect.sort_values(ascending=False).head(n)
        
        return pd.DataFrame({
            "feature": top_features.index,
            "mean_abs_effect": top_features.values
        })
    
    def get_variants_by_feature_effect(
        self,
        feature: Union[str, int],
        top_n: int = 20,
        abs_effect: bool = True
    ) -> pd.DataFrame:
        """
        Get variants with highest effect on a specific feature.
        
        Args:
            feature: Feature name or index
            top_n: Number of top variants to return
            abs_effect: Whether to sort by absolute effect
            
        Returns:
            DataFrame with top variants for the feature
        """
        # Convert feature name to index if needed
        if isinstance(feature, str) and feature in self.feature_names:
            feature_idx = self.feature_names.index(feature)
        elif isinstance(feature, int):
            feature_idx = feature
        else:
            raise ValueError(f"Feature {feature} not found")
            
        # Create a copy of the dataframe with the specific feature effect
        variant_df = self.df.copy()
        variant_df["feature_effect"] = variant_df["diff"].apply(lambda x: x[feature_idx])
        
        # Sort by effect
        if abs_effect:
            variant_df["abs_feature_effect"] = variant_df["feature_effect"].abs()
            sorted_df = variant_df.sort_values("abs_feature_effect", ascending=False).head(top_n)
        else:
            sorted_df = variant_df.sort_values("feature_effect", ascending=False).head(top_n)
            
        return sorted_df[["variant_id", "chrom", "pos", "ref", "alt", "feature_effect"]]
    
    def plot_feature_effects(self, n_features: int = 10) -> plt.Figure:
        """
        Plot mean absolute effect for top features.
        
        Args:
            n_features: Number of top features to plot
            
        Returns:
            Matplotlib figure
        """
        top_features = self.get_top_affected_features(n=n_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot horizontal bar chart
        sns.barplot(
            x="mean_abs_effect",
            y="feature",
            data=top_features,
            ax=ax
        )
        
        # Set labels
        ax.set_title("Top Features Affected by Variants")
        ax.set_xlabel("Mean Absolute Effect")
        ax.set_ylabel("Feature")
        
        plt.tight_layout()
        return fig
    
    def plot_variant_effect(
        self,
        variant_id: str,
        top_n_features: int = 20
    ) -> plt.Figure:
        """
        Plot effect of a variant across features.
        
        Args:
            variant_id: ID of the variant to plot
            top_n_features: Number of top affected features to show
            
        Returns:
            Matplotlib figure
        """
        # Get variant data
        variant = self.df[self.df["variant_id"] == variant_id].iloc[0]
        
        # Get top affected features for this variant
        feature_effects = []
        for i, effect in enumerate(variant["diff"]):
            feature_effects.append({
                "feature": self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}",
                "effect": effect,
                "abs_effect": abs(effect)
            })
            
        # Sort by absolute effect
        feature_effects = sorted(feature_effects, key=lambda x: x["abs_effect"], reverse=True)
        
        # Take top N
        feature_effects = feature_effects[:top_n_features]
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(feature_effects)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        sns.barplot(
            x="effect",
            y="feature",
            data=plot_df,
            ax=ax,
            palette=["red" if x < 0 else "blue" for x in plot_df["effect"]]
        )
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Set labels
        ax.set_title(f"Effect of Variant {variant_id}")
        ax.set_xlabel("Effect (Alt - Ref)")
        ax.set_ylabel("Feature")
        
        plt.tight_layout()
        return fig
