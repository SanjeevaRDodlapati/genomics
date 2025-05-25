"""
Tools for variant effect prediction with genomic deep learning models.

This module provides utilities for predicting the effect of genetic variants
on regulatory elements using trained genomic deep learning models.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import pyfaidx
import pybedtools
from typing import Dict, Any, Optional, Union, Tuple, List, Callable, Iterator
import logging
from pathlib import Path
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VariantEffectPredictor:
    """
    Predicts effects of genetic variants on regulatory elements.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sequence_length: int = 1000,
        reference_genome: Optional[str] = None,
        center_sequence: bool = True,
        batch_size: int = 32,
        use_cuda: bool = torch.cuda.is_available()
    ):
        """
        Initialize the variant effect predictor.
        
        Args:
            model: Trained PyTorch model
            sequence_length: Length of input sequences for the model
            reference_genome: Path to reference genome FASTA file
            center_sequence: Whether to center variants in the input sequence
            batch_size: Batch size for predictions
            use_cuda: Whether to use GPU acceleration
        """
        self.model = model
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.center_sequence = center_sequence
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move model to GPU if available and requested
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Load reference genome if provided
        self.reference_genome = None
        if reference_genome:
            if os.path.exists(reference_genome):
                self.reference_genome = pyfaidx.Fasta(reference_genome, rebuild=False)
            else:
                raise FileNotFoundError(f"Reference genome not found: {reference_genome}")
    
    def predict_variant_effect(
        self,
        variants: Union[str, pd.DataFrame],
        output_file: Optional[str] = None,
        reference_allele_col: str = "ref",
        alternate_allele_col: str = "alt",
        chromosome_col: str = "chrom",
        position_col: str = "pos",
        id_col: Optional[str] = None
    ) -> Union[pd.DataFrame, str]:
        """
        Predict the effect of variants.
        
        Args:
            variants: Path to variant file (VCF/BED) or DataFrame with variant info
            output_file: Optional path to save results
            reference_allele_col: Column name for reference allele
            alternate_allele_col: Column name for alternate allele
            chromosome_col: Column name for chromosome
            position_col: Column name for position
            id_col: Optional column name for variant ID
            
        Returns:
            DataFrame with variant effects or path to output file
        """
        if self.reference_genome is None:
            raise ValueError("Reference genome required for variant effect prediction")
        
        # Load variants if file path provided
        if isinstance(variants, str):
            if variants.endswith('.vcf') or variants.endswith('.vcf.gz'):
                # Load VCF file
                variants_df = self._load_vcf(variants)
            elif variants.endswith('.bed') or variants.endswith('.bed.gz'):
                # Load BED file
                variants_df = self._load_bed(variants)
            else:
                raise ValueError(f"Unsupported variant file format: {variants}")
        else:
            # Use provided DataFrame
            variants_df = variants
        
        # Validate required columns
        required_cols = [chromosome_col, position_col, reference_allele_col, alternate_allele_col]
        missing_cols = [col for col in required_cols if col not in variants_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in variants data: {missing_cols}")
        
        # Generate reference and alternate sequences
        ref_sequences = []
        alt_sequences = []
        variant_ids = []
        
        for idx, row in variants_df.iterrows():
            chrom = str(row[chromosome_col])
            pos = int(row[position_col])
            ref_allele = str(row[reference_allele_col])
            alt_allele = str(row[alternate_allele_col])
            
            # Generate ID if not provided
            if id_col and id_col in row:
                variant_id = row[id_col]
            else:
                variant_id = f"{chrom}:{pos}_{ref_allele}>{alt_allele}"
            
            # Get sequence window around variant
            try:
                ref_seq, alt_seq = self._get_variant_sequences(chrom, pos, ref_allele, alt_allele)
                ref_sequences.append(ref_seq)
                alt_sequences.append(alt_seq)
                variant_ids.append(variant_id)
            except Exception as e:
                logger.warning(f"Error processing variant {variant_id}: {str(e)}")
        
        # Convert sequences to one-hot encoding
        ref_onehot = np.array([self._sequence_to_onehot(seq) for seq in ref_sequences])
        alt_onehot = np.array([self._sequence_to_onehot(seq) for seq in alt_sequences])
        
        # Make predictions
        ref_predictions = self._predict_batch(ref_onehot)
        alt_predictions = self._predict_batch(alt_onehot)
        
        # Calculate variant effects (difference between alt and ref)
        effects = alt_predictions - ref_predictions
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'variant_id': variant_ids,
            'chromosome': variants_df[chromosome_col].values[:len(variant_ids)],
            'position': variants_df[position_col].values[:len(variant_ids)],
            'reference': variants_df[reference_allele_col].values[:len(variant_ids)],
            'alternate': variants_df[alternate_allele_col].values[:len(variant_ids)]
        })
        
        # Add prediction columns
        num_targets = ref_predictions.shape[1]
        for i in range(num_targets):
            result_df[f'ref_pred_{i}'] = ref_predictions[:, i]
            result_df[f'alt_pred_{i}'] = alt_predictions[:, i]
            result_df[f'effect_{i}'] = effects[:, i]
        
        # Add summary statistics
        result_df['max_effect'] = effects.max(axis=1)
        result_df['min_effect'] = effects.min(axis=1)
        result_df['mean_effect'] = effects.mean(axis=1)
        result_df['abs_max_effect'] = np.abs(effects).max(axis=1)
        
        # Save results if output file provided
        if output_file:
            if output_file.endswith('.csv'):
                result_df.to_csv(output_file, index=False)
            elif output_file.endswith('.tsv'):
                result_df.to_csv(output_file, sep='\t', index=False)
            elif output_file.endswith('.h5') or output_file.endswith('.hdf5'):
                with h5py.File(output_file, 'w') as f:
                    # Save DataFrame columns
                    for col in result_df.columns:
                        if col in ['ref_pred', 'alt_pred', 'effect']:
                            # Save arrays
                            f.create_dataset(col, data=result_df[col].values)
                        else:
                            # Save other columns as datasets
                            data = result_df[col].values
                            if data.dtype == np.object:
                                data = np.array(data, dtype=h5py.special_dtype(vlen=str))
                            f.create_dataset(col, data=data)
                    
                    # Save raw predictions and effects
                    f.create_dataset('ref_predictions', data=ref_predictions)
                    f.create_dataset('alt_predictions', data=alt_predictions)
                    f.create_dataset('effects', data=effects)
            else:
                # Default to CSV
                result_df.to_csv(output_file, index=False)
            
            return output_file
        
        return result_df
    
    def _get_variant_sequences(
        self,
        chrom: str,
        pos: int,
        ref_allele: str,
        alt_allele: str
    ) -> Tuple[str, str]:
        """
        Get reference and alternate sequences for a variant.
        
        Args:
            chrom: Chromosome
            pos: Position (1-based)
            ref_allele: Reference allele
            alt_allele: Alternate allele
            
        Returns:
            Tuple of (reference sequence, alternate sequence)
        """
        # Adjust position to 0-based for pyfaidx
        pos_0based = pos - 1
        
        # Calculate window size
        window_size = self.sequence_length
        if len(ref_allele) != len(alt_allele):
            # For indels, adjust window size to maintain sequence length
            size_diff = len(alt_allele) - len(ref_allele)
            window_size -= size_diff
        
        # Calculate start and end positions
        if self.center_sequence:
            # Center the variant in the sequence
            start = pos_0based - (window_size // 2)
            end = start + window_size
        else:
            # Variant at the beginning of the sequence
            start = pos_0based
            end = start + window_size
        
        # Ensure start position is not negative
        if start < 0:
            end -= start  # Extend the end by the amount start is negative
            start = 0
        
        # Get the reference sequence
        try:
            # Get sequence with pyfaidx (converts to uppercase)
            ref_seq = str(self.reference_genome[chrom][start:end]).upper()
        except Exception as e:
            raise ValueError(f"Error retrieving sequence for {chrom}:{pos}: {str(e)}")
        
        # Replace reference allele with alternate allele to create alt sequence
        var_pos = pos_0based - start
        alt_seq = ref_seq[:var_pos] + alt_allele + ref_seq[var_pos + len(ref_allele):]
        
        # Pad or trim sequences to ensure they match the expected length
        ref_seq = self._adjust_sequence_length(ref_seq, self.sequence_length)
        alt_seq = self._adjust_sequence_length(alt_seq, self.sequence_length)
        
        return ref_seq, alt_seq
    
    def _adjust_sequence_length(self, sequence: str, target_length: int) -> str:
        """
        Adjust sequence to the target length by padding or trimming.
        
        Args:
            sequence: DNA sequence
            target_length: Target length
            
        Returns:
            Adjusted sequence
        """
        current_length = len(sequence)
        
        if current_length == target_length:
            return sequence
        elif current_length < target_length:
            # Pad with N's at the end
            padding = 'N' * (target_length - current_length)
            return sequence + padding
        else:
            # Trim from the end
            return sequence[:target_length]
    
    def _sequence_to_onehot(self, sequence: str) -> np.ndarray:
        """
        Convert DNA sequence to one-hot encoding.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            One-hot encoded array [4, length]
        """
        sequence = sequence.upper()
        seq_len = len(sequence)
        onehot = np.zeros((4, seq_len), dtype=np.float32)
        
        # Define nucleotide mapping
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        for i, nuc in enumerate(sequence):
            if nuc in nuc_map:
                onehot[nuc_map[nuc], i] = 1.0
            else:
                # For ambiguous nucleotides (N, etc.), use uniform probability
                onehot[:, i] = 0.25
        
        return onehot
    
    def _predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions on a batch of sequences.
        
        Args:
            sequences: Batch of one-hot encoded sequences [batch, 4, length]
            
        Returns:
            Model predictions [batch, num_targets]
        """
        num_samples = sequences.shape[0]
        all_preds = []
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, num_samples, self.batch_size):
                batch_end = min(i + self.batch_size, num_samples)
                batch = sequences[i:batch_end]
                
                # Convert to torch tensor
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
                
                # Make predictions
                preds = self.model(batch_tensor)
                
                # Handle different output formats
                if isinstance(preds, tuple):
                    preds = preds[0]  # Take first output if model returns multiple outputs
                
                # Convert to numpy and append to results
                all_preds.append(preds.cpu().numpy())
        
        # Combine all predictions
        return np.vstack(all_preds)
    
    def _load_vcf(self, vcf_path: str) -> pd.DataFrame:
        """
        Load variants from a VCF file.
        
        Args:
            vcf_path: Path to VCF file
            
        Returns:
            DataFrame with variant information
        """
        try:
            from cyvcf2 import VCF
            vcf = VCF(vcf_path)
            
            variants = []
            for variant in vcf:
                variants.append({
                    'chrom': variant.CHROM,
                    'pos': variant.POS,
                    'ref': variant.REF,
                    'alt': variant.ALT[0],  # Take first alternate allele
                    'id': variant.ID if variant.ID else f"{variant.CHROM}:{variant.POS}_{variant.REF}>{variant.ALT[0]}"
                })
            
            return pd.DataFrame(variants)
        
        except ImportError:
            logger.warning("cyvcf2 not installed. Using a slower method to parse VCF.")
            
            # Parse VCF manually
            variants = []
            with open(vcf_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    fields = line.strip().split('\t')
                    chrom = fields[0]
                    pos = int(fields[1])
                    var_id = fields[2]
                    ref = fields[3]
                    alts = fields[4].split(',')
                    
                    # Create a record for each alternate allele
                    for alt in alts:
                        variants.append({
                            'chrom': chrom,
                            'pos': pos,
                            'ref': ref,
                            'alt': alt,
                            'id': var_id if var_id != '.' else f"{chrom}:{pos}_{ref}>{alt}"
                        })
            
            return pd.DataFrame(variants)
    
    def _load_bed(self, bed_path: str) -> pd.DataFrame:
        """
        Load variants from a BED file.
        
        Args:
            bed_path: Path to BED file
            
        Returns:
            DataFrame with variant information
        """
        try:
            bed = pybedtools.BedTool(bed_path)
            
            variants = []
            for entry in bed:
                # BED format is 0-based, convert to 1-based for genomic coordinates
                chrom = entry.chrom
                pos = entry.start + 1  # Convert to 1-based
                
                # Parse variant information
                if len(entry.fields) >= 6:
                    name = entry.name
                    
                    # Parse ref/alt from name field if in format "ref>alt"
                    if ">" in name:
                        ref, alt = name.split(">")
                    else:
                        # Default to simple SNV if not specified
                        ref = "N"
                        alt = "N"
                else:
                    # Minimal BED format, no name field
                    name = f"{chrom}:{pos}"
                    ref = "N"
                    alt = "N"
                
                variants.append({
                    'chrom': chrom,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                    'id': name
                })
            
            return pd.DataFrame(variants)
        
        except Exception as e:
            raise ValueError(f"Error loading BED file: {str(e)}")


def score_variants(
    model: nn.Module,
    variants_file: str,
    reference_genome: str,
    output_file: str,
    sequence_length: int = 1000,
    batch_size: int = 32,
    use_cuda: bool = True
) -> str:
    """
    Score genetic variants with a trained genomic deep learning model.
    
    Args:
        model: Trained PyTorch model
        variants_file: Path to VCF or BED file with variants
        reference_genome: Path to reference genome FASTA file
        output_file: Path to save results
        sequence_length: Length of input sequences for the model
        batch_size: Batch size for predictions
        use_cuda: Whether to use GPU acceleration
        
    Returns:
        Path to the output file with variant scores
    """
    predictor = VariantEffectPredictor(
        model=model,
        sequence_length=sequence_length,
        reference_genome=reference_genome,
        batch_size=batch_size,
        use_cuda=use_cuda
    )
    
    return predictor.predict_variant_effect(
        variants=variants_file,
        output_file=output_file
    )


def compute_target_impact_scores(
    effects: np.ndarray,
    target_names: Optional[List[str]] = None,
    percentile_threshold: float = 99.0
) -> pd.DataFrame:
    """
    Compute impact scores for variants across all targets.
    
    Args:
        effects: Variant effect matrix [num_variants, num_targets]
        target_names: Optional list of target names
        percentile_threshold: Percentile threshold for significant effects
        
    Returns:
        DataFrame with impact scores per target
    """
    num_variants, num_targets = effects.shape
    
    # Create target names if not provided
    if target_names is None:
        target_names = [f"Target_{i}" for i in range(num_targets)]
    
    # Calculate threshold values for significant effects
    pos_threshold = np.percentile(effects, percentile_threshold)
    neg_threshold = np.percentile(effects, 100 - percentile_threshold)
    
    # Compute metrics for each target
    target_metrics = []
    for i in range(num_targets):
        target_effects = effects[:, i]
        
        # Count significant effects
        pos_sig = np.sum(target_effects >= pos_threshold)
        neg_sig = np.sum(target_effects <= neg_threshold)
        
        # Calculate effect sizes
        mean_effect = np.mean(target_effects)
        max_effect = np.max(target_effects)
        min_effect = np.min(target_effects)
        std_effect = np.std(target_effects)
        
        target_metrics.append({
            'target_name': target_names[i],
            'mean_effect': mean_effect,
            'max_effect': max_effect,
            'min_effect': min_effect,
            'std_effect': std_effect,
            'positive_significant': pos_sig,
            'negative_significant': neg_sig,
            'total_significant': pos_sig + neg_sig,
            'effect_ratio': (pos_sig + 1) / (neg_sig + 1)  # Add 1 to avoid division by zero
        })
    
    return pd.DataFrame(target_metrics)