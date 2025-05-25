"""Utilities for handling model predictions."""

import os
import numpy as np
import h5py
import json
from typing import List, Dict, Any, Optional, Union


def save_predictions(
    predictions: List[Dict[str, Any]],
    output_dir: str,
    config: Dict[str, Any]
):
    """Save model predictions to disk.
    
    Args:
        predictions: List of prediction dictionaries
        output_dir: Directory to save predictions
        config: Configuration dictionary
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group predictions by type
    all_preds = []
    all_metadata = []
    
    for batch in predictions:
        if batch is None:
            continue
            
        # Extract predictions and convert to numpy
        preds = batch.get('predictions')
        if isinstance(preds, list):
            all_preds.extend([p.cpu().numpy() for p in preds])
        else:
            all_preds.append(preds.cpu().numpy())
            
        # Extract metadata if available
        if 'metadata' in batch:
            all_metadata.extend(batch['metadata'])
    
    # Concatenate predictions
    if all_preds:
        all_preds = np.concatenate(all_preds)
        
        # Save predictions based on format
        save_format = config.get('prediction_format', 'h5')
        
        if save_format == 'h5':
            _save_h5(all_preds, all_metadata, output_dir, config)
        elif save_format == 'npy':
            _save_npy(all_preds, all_metadata, output_dir, config)
        elif save_format == 'csv':
            _save_csv(all_preds, all_metadata, output_dir, config)
        else:
            # Default to H5 format
            _save_h5(all_preds, all_metadata, output_dir, config)


def _save_h5(
    predictions: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_dir: str,
    config: Dict[str, Any]
):
    """Save predictions in H5 format.
    
    Args:
        predictions: Numpy array of predictions
        metadata: List of metadata dictionaries
        output_dir: Directory to save predictions
        config: Configuration dictionary
    """
    output_file = os.path.join(output_dir, 'predictions.h5')
    
    with h5py.File(output_file, 'w') as f:
        # Save predictions
        f.create_dataset('predictions', data=predictions, compression='gzip')
        
        # Save metadata if available
        if metadata:
            # Convert metadata to a serializable format
            metadata_json = json.dumps(metadata)
            f.create_dataset('metadata', data=np.string_(metadata_json))
        
        # Save config info
        f.attrs['config'] = json.dumps(config)
    
    print(f"Saved predictions to {output_file}")


def _save_npy(
    predictions: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_dir: str,
    config: Dict[str, Any]
):
    """Save predictions in NPY format.
    
    Args:
        predictions: Numpy array of predictions
        metadata: List of metadata dictionaries
        output_dir: Directory to save predictions
        config: Configuration dictionary
    """
    # Save predictions
    output_file = os.path.join(output_dir, 'predictions.npy')
    np.save(output_file, predictions)
    
    # Save metadata if available
    if metadata:
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
    
    # Save config info
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    print(f"Saved predictions to {output_file}")


def _save_csv(
    predictions: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_dir: str,
    config: Dict[str, Any]
):
    """Save predictions in CSV format.
    
    Args:
        predictions: Numpy array of predictions
        metadata: List of metadata dictionaries
        output_dir: Directory to save predictions
        config: Configuration dictionary
    """
    try:
        import pandas as pd
        
        # Convert predictions to dataframe
        df = pd.DataFrame(predictions)
        
        # Add metadata if available
        if metadata:
            for i, meta in enumerate(metadata):
                for key, value in meta.items():
                    if i == 0:
                        df[key] = None
                    df.at[i, key] = value
        
        # Save as CSV
        output_file = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(output_file, index=False)
        
        print(f"Saved predictions to {output_file}")
    
    except ImportError:
        print("Warning: pandas not installed. Falling back to NPY format.")
        _save_npy(predictions, metadata, output_dir, config)
