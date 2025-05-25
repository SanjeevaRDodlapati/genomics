"""
Visualization and interpretability tools for genomic deep learning models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GradientShap,
    Occlusion,
    NoiseTunnel,
    visualization
)


class MotifVisualizer:
    """
    Visualization tool for extracting and plotting sequence motifs.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize the motif visualizer.
        
        Args:
            model: PyTorch model
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.device = device
        
        # Setup attribution methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.deep_lift = DeepLift(self.model)
        self.gradient_shap = GradientShap(self.model)
        self.occlusion = Occlusion(self.model)
        
    def get_integrated_gradients(
        self, 
        input_seqs: torch.Tensor, 
        target_class: int,
        n_steps: int = 50,
        internal_batch_size: int = 4
    ) -> np.ndarray:
        """
        Calculate integrated gradients attribution for input sequences.
        
        Args:
            input_seqs: Input sequences [batch_size, n_features, seq_length]
            target_class: Target class index
            n_steps: Number of steps for path integral
            internal_batch_size: Batch size for attribution calculation
            
        Returns:
            Attribution scores
        """
        input_seqs = input_seqs.to(self.device)
        
        # Create baseline of all zeros
        baseline = torch.zeros_like(input_seqs).to(self.device)
        
        # Calculate attributions
        attributions = self.integrated_gradients.attribute(
            input_seqs,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size
        )
        
        return attributions.cpu().detach().numpy()
    
    def get_deep_lift_attributions(
        self, 
        input_seqs: torch.Tensor, 
        target_class: int
    ) -> np.ndarray:
        """
        Calculate DeepLIFT attributions for input sequences.
        
        Args:
            input_seqs: Input sequences [batch_size, n_features, seq_length]
            target_class: Target class index
            
        Returns:
            Attribution scores
        """
        input_seqs = input_seqs.to(self.device)
        
        # Create baseline of all zeros
        baseline = torch.zeros_like(input_seqs).to(self.device)
        
        # Calculate attributions
        attributions = self.deep_lift.attribute(
            input_seqs,
            baselines=baseline,
            target=target_class
        )
        
        return attributions.cpu().detach().numpy()
        
    def visualize_sequence_attribution(
        self,
        input_seq: Union[torch.Tensor, np.ndarray],
        attributions: Union[torch.Tensor, np.ndarray],
        method_name: str = "Integrated Gradients",
        figsize: Tuple[int, int] = (20, 4),
        show_nucleotide_names: bool = True
    ) -> plt.Figure:
        """
        Visualize attribution scores for a sequence.
        
        Args:
            input_seq: Input sequence [n_features, seq_length]
            attributions: Attribution scores [n_features, seq_length]
            method_name: Name of the attribution method
            figsize: Figure size
            show_nucleotide_names: Whether to show nucleotide names (A, C, G, T)
            
        Returns:
            Matplotlib figure
        """
        if isinstance(input_seq, torch.Tensor):
            input_seq = input_seq.cpu().detach().numpy()
            
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.cpu().detach().numpy()
            
        # Make sure both are 2D
        if input_seq.ndim > 2:
            input_seq = input_seq[0]  # Take first sequence in batch
            
        if attributions.ndim > 2:
            attributions = attributions[0]  # Take first attribution in batch
            
        # Convert to correct shape if needed
        if input_seq.shape[0] == 4 and input_seq.shape[1] > 4:
            # Shape is already [4, seq_length]
            pass
        elif input_seq.shape[0] > 4 and input_seq.shape[1] == 4:
            # Shape is [seq_length, 4], transpose
            input_seq = input_seq.T
            attributions = attributions.T
            
        seq_length = input_seq.shape[1]
        nucleotides = ['A', 'C', 'G', 'T']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create position indices
        positions = np.arange(seq_length)
        
        # Create heatmap of attributions
        im = ax.imshow(attributions, cmap='RdBu_r', aspect='auto', 
                       vmin=-np.abs(attributions).max(), vmax=np.abs(attributions).max())
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution Score')
        
        # Set labels
        ax.set_title(f"{method_name} Attribution")
        ax.set_xlabel("Sequence Position")
        
        # Set y-tick labels to nucleotides if requested
        if show_nucleotide_names:
            ax.set_yticks(np.arange(4))
            ax.set_yticklabels(nucleotides)
        else:
            ax.set_ylabel("Feature Index")
            
        # Mark the actual nucleotide at each position
        for pos in range(seq_length):
            nuc_idx = np.argmax(input_seq[:, pos])
            ax.text(pos, nuc_idx, nucleotides[nuc_idx], 
                   ha='center', va='center', color='black')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def extract_motifs(
        self, 
        model: nn.Module,
        layer_idx: int,
        filter_indices: Optional[List[int]] = None,
        window_size: int = 15
    ) -> Dict[int, np.ndarray]:
        """
        Extract motif representations from convolutional filters.
        
        Args:
            model: The model to extract motifs from
            layer_idx: Index of convolutional layer
            filter_indices: Specific filter indices to extract (None for all)
            window_size: Size of sequence window for motifs
            
        Returns:
            Dictionary mapping filter indices to motif matrices
        """
        # Get the convolutional layer
        conv_layers = [module for module in model.modules() 
                       if isinstance(module, nn.Conv1d)]
        
        if layer_idx >= len(conv_layers):
            raise ValueError(f"Layer index {layer_idx} out of range. "
                            f"Model has {len(conv_layers)} convolutional layers.")
        
        conv_layer = conv_layers[layer_idx]
        
        # Extract weights
        weights = conv_layer.weight.data.cpu().numpy()
        
        # Determine which filters to process
        if filter_indices is None:
            filter_indices = list(range(weights.shape[0]))
            
        # Extract motifs for each filter
        motifs = {}
        
        for filter_idx in filter_indices:
            if filter_idx >= weights.shape[0]:
                continue
                
            # Get filter weights
            filter_weights = weights[filter_idx]  # shape: [in_channels, kernel_size]
            
            # Convert to position weight matrix
            pwm = filter_weights
            
            # Normalize to create probability matrix
            pwm = pwm - pwm.min(axis=0)
            row_sums = pwm.sum(axis=0)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            pwm = pwm / row_sums
            
            motifs[filter_idx] = pwm
            
        return motifs
    
    def plot_sequence_logo(
        self,
        pwm: np.ndarray,
        figsize: Tuple[int, int] = (10, 3),
        title: str = "Sequence Logo"
    ) -> plt.Figure:
        """
        Plot a sequence logo from a position weight matrix.
        
        Args:
            pwm: Position weight matrix [4, motif_length]
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Ensure correct shape
        if pwm.shape[0] != 4:
            pwm = pwm.T
            
        motif_length = pwm.shape[1]
        nucleotides = ['A', 'C', 'G', 'T']
        
        # Calculate information content
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        pwm_norm = pwm + epsilon
        pwm_norm = pwm_norm / pwm_norm.sum(axis=0)
        
        # Information content = 2 - entropy
        entropy = -np.sum(pwm_norm * np.log2(pwm_norm), axis=0)
        info_content = 2 - entropy
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot stacked logo
        y_base = np.zeros(motif_length)
        max_height = 0
        
        for i, nuc in enumerate(nucleotides):
            # Height is proportional to probability × information content
            heights = pwm_norm[i] * info_content
            
            # Different colors for each nucleotide
            color = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}[nuc]
            
            # Plot bars
            ax.bar(np.arange(motif_length), heights, bottom=y_base, 
                  width=0.8, color=color, label=nuc, align='center')
            
            # Update baseline for stacking
            y_base = y_base + heights
            max_height = max(max_height, y_base.max())
        
        # Set axis labels and title
        ax.set_xlabel("Position")
        ax.set_ylabel("Information Content (bits)")
        ax.set_title(title)
        ax.set_xlim(-0.5, motif_length - 0.5)
        ax.set_ylim(0, max(2, max_height * 1.1))  # Set y-limit to 2 bits or higher
        
        # Add legend
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def save_filter_logos(
        self,
        model: nn.Module,
        layer_idx: int = 0,
        output_dir: str = "motif_logos",
        top_k_filters: Optional[int] = None
    ) -> None:
        """
        Extract and save sequence logos for convolutional filters.
        
        Args:
            model: The model to extract motifs from
            layer_idx: Index of convolutional layer
            output_dir: Directory to save plots
            top_k_filters: Number of top filters to plot (None for all)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract motifs
        motifs = self.extract_motifs(model, layer_idx)
        
        # If top_k is specified, select filters with highest max activation
        if top_k_filters is not None and top_k_filters < len(motifs):
            # Calculate max activation for each filter
            max_activations = {idx: np.max(np.abs(motif)) for idx, motif in motifs.items()}
            
            # Sort by activation and take top-k
            top_filters = sorted(max_activations.keys(), 
                                key=lambda idx: max_activations[idx], 
                                reverse=True)[:top_k_filters]
            
            # Filter motifs to only include top filters
            motifs = {idx: motifs[idx] for idx in top_filters}
        
        # Generate and save plots
        for filter_idx, motif in motifs.items():
            fig = self.plot_sequence_logo(
                motif,
                title=f"Filter {filter_idx} Sequence Logo"
            )
            
            plt.savefig(os.path.join(output_dir, f"filter_{filter_idx}_logo.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
