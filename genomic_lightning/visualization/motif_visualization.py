"""
Visualization tools for genomic deep learning models.

This module provides tools for visualizing motifs learned by genomic deep learning models,
including convolutional filters that may correspond to transcription factor binding sites.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from typing import List, Dict, Union, Tuple, Optional
import logomaker


class MotifVisualizer:
    """
    Class for visualizing motifs learned by genomic deep learning models.
    """
    
    def __init__(self, model: nn.Module, layer_name: Optional[str] = None):
        """
        Initialize the motif visualizer.
        
        Args:
            model: The PyTorch model to visualize
            layer_name: Optional name of the convolutional layer to visualize.
                        If None, will attempt to find the first convolutional layer.
        """
        self.model = model
        self.layer_name = layer_name
        self._conv_layer = self._get_conv_layer()
        
    def _get_conv_layer(self) -> nn.Conv1d:
        """
        Get the convolutional layer to visualize.
        
        Returns:
            The convolutional layer of interest.
        """
        if self.layer_name is not None:
            for name, module in self.model.named_modules():
                if name == self.layer_name and isinstance(module, nn.Conv1d):
                    return module
            
            raise ValueError(f"Could not find convolutional layer named {self.layer_name}")
        
        # If layer_name is None, find the first convolutional layer
        for module in self.model.modules():
            if isinstance(module, nn.Conv1d):
                return module
                
        raise ValueError("Could not find any convolutional layer in the model")

    def get_filters(self) -> torch.Tensor:
        """
        Extract convolutional filters from the model.
        
        Returns:
            Tensor of shape [num_filters, 4, filter_length] for genomic models
        """
        weights = self._conv_layer.weight.data
        
        # Check if in genomic format (num_filters, 4, filter_length)
        if weights.shape[1] != 4:
            raise ValueError(f"Expected 4 input channels for genomic data, got {weights.shape[1]}")
            
        return weights
    
    def plot_filter(self, filter_idx: int, figsize: Tuple[int, int] = (10, 2)) -> plt.Figure:
        """
        Plot a single filter as a sequence logo.
        
        Args:
            filter_idx: Index of the filter to plot
            figsize: Size of the figure (width, height)
            
        Returns:
            The matplotlib figure object
        """
        filters = self.get_filters()
        
        if filter_idx >= filters.shape[0]:
            raise ValueError(f"Filter index {filter_idx} out of range (0-{filters.shape[0]-1})")
        
        # Get the single filter and convert to PWM format
        filt = filters[filter_idx].cpu().numpy()
        
        # ACGT to match standard genomic convention
        bases = ['A', 'C', 'G', 'T']
        
        # Create dataframe for logomaker
        pwm_df = self._convert_filter_to_pwm(filt)
        
        # Create figure and plot logo
        fig, ax = plt.subplots(figsize=figsize)
        logo = logomaker.Logo(pwm_df, ax=ax)
        
        # Customize the plot
        ax.set_title(f"Filter {filter_idx}")
        ax.set_ylabel("Information Content")
        ax.set_xlabel("Position")
        
        return fig
    
    def _convert_filter_to_pwm(self, filt: np.ndarray) -> pd.DataFrame:
        """
        Convert a convolutional filter to position weight matrix format for logomaker.
        
        Args:
            filt: Filter of shape [4, filter_length]
            
        Returns:
            Pandas DataFrame in logomaker format
        """
        # ACGT to match standard genomic convention
        bases = ['A', 'C', 'G', 'T']
        
        # Convert filter values to information content
        # Shift by min value and normalize
        norm_filt = filt - filt.min(axis=0, keepdims=True)
        
        # Avoid division by zero
        sum_per_pos = norm_filt.sum(axis=0, keepdims=True)
        sum_per_pos[sum_per_pos == 0] = 1.0
        
        pwm = norm_filt / sum_per_pos
        
        # Create dataframe for logomaker
        pwm_df = pd.DataFrame(pwm.T, columns=bases)
        
        return pwm_df
    
    def plot_all_filters(self, 
                          n_cols: int = 4, 
                          figsize_per_filter: Tuple[int, int] = (5, 2),
                          max_filters: Optional[int] = None) -> plt.Figure:
        """
        Plot all filters as sequence logos in a grid.
        
        Args:
            n_cols: Number of columns in the grid
            figsize_per_filter: Size of each individual filter plot (width, height)
            max_filters: Maximum number of filters to plot. If None, plot all.
            
        Returns:
            The matplotlib figure object
        """
        filters = self.get_filters()
        n_filters = filters.shape[0]
        
        if max_filters is not None:
            n_filters = min(n_filters, max_filters)
        
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig_width = n_cols * figsize_per_filter[0]
        fig_height = n_rows * figsize_per_filter[1]
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Convert axes to flat array for easier indexing
        if n_rows > 1 and n_cols > 1:
            axes_flat = axes.flatten()
        elif n_rows == 1 and n_cols > 1:
            axes_flat = axes
        elif n_cols == 1 and n_rows > 1:
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        # Plot each filter
        for i in range(n_filters):
            # Get the PWM for this filter
            filt = filters[i].cpu().numpy()
            pwm_df = self._convert_filter_to_pwm(filt)
            
            # Create logo in the appropriate subplot
            logo = logomaker.Logo(pwm_df, ax=axes_flat[i])
            
            # Customize the plot
            axes_flat[i].set_title(f"Filter {i}")
            
            # Only set y-label for leftmost plots
            if i % n_cols == 0:
                axes_flat[i].set_ylabel("Information Content")
            else:
                axes_flat[i].set_ylabel("")
                
            # Only set x-label for bottom plots
            if i >= n_filters - n_cols:
                axes_flat[i].set_xlabel("Position")
            else:
                axes_flat[i].set_xlabel("")
        
        # Hide empty subplots
        for i in range(n_filters, len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    
def plot_filter_activations(activations: torch.Tensor, 
                           sequence: Optional[str] = None,
                           top_k: int = 5,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot the activations of convolutional filters across a sequence.
    
    Args:
        activations: Activation tensor of shape [1, n_filters, seq_length]
        sequence: Optional DNA sequence string to display below the plot
        top_k: Number of top filters to highlight
        figsize: Figure size (width, height)
        
    Returns:
        The matplotlib figure object
    """
    # Ensure activations is the right shape
    if len(activations.shape) != 3:
        raise ValueError(f"Expected activations of shape [1, n_filters, seq_length], got {activations.shape}")
    
    # Get the dimensions
    n_filters = activations.shape[1]
    seq_length = activations.shape[2]
    
    # Convert to numpy
    act = activations[0].cpu().numpy()  # Shape: [n_filters, seq_length]
    
    # Find the top-k filters by maximum activation
    max_act_per_filter = np.max(act, axis=1)
    top_filters = np.argsort(max_act_per_filter)[-top_k:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap of all activations
    sns.heatmap(act, cmap='viridis', ax=ax)
    
    # Highlight the top-k filters
    for i, filt_idx in enumerate(top_filters):
        ax.axhline(y=filt_idx + 0.5, color='red', linestyle='-', linewidth=1, alpha=0.7)
        ax.text(-5, filt_idx + 0.5, f"Filter {filt_idx}", 
                va='center', ha='right', color='red', fontweight='bold')
    
    # If sequence provided, show it on x-axis
    if sequence is not None:
        if len(sequence) != seq_length:
            print(f"Warning: Sequence length ({len(sequence)}) does not match activation length ({seq_length})")
            sequence = sequence[:seq_length]
            
        # Replace x-tick labels with sequence
        ax.set_xticks(np.arange(0.5, len(sequence) + 0.5))
        ax.set_xticklabels(list(sequence), fontsize=8, rotation=0)
        
    ax.set_ylabel("Filter")
    ax.set_xlabel("Position")
    ax.set_title("Filter Activations Across Sequence")
        
    plt.tight_layout()
    return fig


def get_activated_motifs(model: nn.Module, 
                        sequences: torch.Tensor, 
                        layer_name: Optional[str] = None,
                        threshold: float = 0.5,
                        return_scores: bool = False) -> Union[List[List[str]], Tuple[List[List[str]], List[List[float]]]]:
    """
    Extract DNA subsequences that activate specific filters in the model.
    
    Args:
        model: The PyTorch model to analyze
        sequences: Tensor of one-hot encoded DNA sequences [batch, 4, seq_length]
        layer_name: Name of the convolutional layer to analyze
        threshold: Activation threshold for considering a motif
        return_scores: Whether to return activation scores along with motifs
        
    Returns:
        If return_scores is False: List of lists of motif sequences per filter
        If return_scores is True: Tuple of (motif_sequences, activation_scores)
    """
    # Create a hook to get intermediate activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register the hook
    visualizer = MotifVisualizer(model, layer_name)
    conv_layer = visualizer._conv_layer
    handle = conv_layer.register_forward_hook(get_activation('conv'))
    
    # Run the model to get activations
    with torch.no_grad():
        _ = model(sequences)
        
    # Get the activations and remove hook
    act = activations['conv']  # Shape: [batch, n_filters, seq_length]
    handle.remove()
    
    # Get filter details
    filters = visualizer.get_filters()
    n_filters = filters.shape[0]
    filter_length = filters.shape[2]
    
    # Convert one-hot sequences back to ACGT
    bases = ['A', 'C', 'G', 'T']
    sequence_strings = []
    
    for seq in sequences:
        seq_np = seq.cpu().numpy()
        seq_str = ""
        for i in range(seq_np.shape[1]):
            idx = np.argmax(seq_np[:, i])
            seq_str += bases[idx]
        sequence_strings.append(seq_str)
    
    # Extract motifs for each filter
    motifs_per_filter = [[] for _ in range(n_filters)]
    scores_per_filter = [[] for _ in range(n_filters)]
    
    for b, seq_str in enumerate(sequence_strings):
        for f in range(n_filters):
            # Find positions where this filter activates above threshold
            filter_act = act[b, f].cpu().numpy()
            high_act_pos = np.where(filter_act > threshold)[0]
            
            # Extract subsequences at these positions
            for pos in high_act_pos:
                end_pos = pos + filter_length
                if end_pos <= len(seq_str):
                    motif = seq_str[pos:end_pos]
                    score = float(filter_act[pos])
                    
                    motifs_per_filter[f].append(motif)
                    scores_per_filter[f].append(score)
    
    if return_scores:
        return motifs_per_filter, scores_per_filter
    else:
        return motifs_per_filter