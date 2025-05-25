"""
Utilities for handling model predictions.

This module provides functions for processing, converting, and analyzing
predictions from genomic deep learning models.
"""

import os
import numpy as np
import torch
import h5py
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple, List
import logging

logger = logging.getLogger(__name__)

def convert_legacy_predictions(
    predictions: Union[str, np.ndarray, Dict],
    from_format: str,
    to_format: str,
    output_path: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Union[np.ndarray, str]:
    """
    Convert predictions from legacy formats to new formats.
    
    Args:
        predictions: Predictions to convert (path to file or loaded predictions)
        from_format: Source format ('uavarprior', 'fugep', 'numpy', 'hdf5', 'csv')
        to_format: Target format ('numpy', 'hdf5', 'csv')
        output_path: Path to save the converted predictions
        metadata: Optional metadata to include with the predictions
        
    Returns:
        Converted predictions or path to the saved file
    """
    # Load predictions if a path is provided
    if isinstance(predictions, str):
        if os.path.exists(predictions):
            if from_format == 'uavarprior':
                predictions = _load_uavarprior_predictions(predictions)
            elif from_format == 'fugep':
                predictions = _load_fugep_predictions(predictions)
            elif from_format == 'numpy':
                predictions = np.load(predictions)
            elif from_format == 'hdf5':
                with h5py.File(predictions, 'r') as f:
                    predictions = {key: f[key][()] for key in f.keys()}
            elif from_format == 'csv':
                predictions = pd.read_csv(predictions).values
            else:
                raise ValueError(f"Unsupported source format: {from_format}")
        else:
            raise FileNotFoundError(f"Predictions file not found: {predictions}")
    
    # Convert and save in the target format
    if to_format == 'numpy':
        if output_path:
            if not output_path.endswith('.npy'):
                output_path += '.npy'
            
            if isinstance(predictions, dict):
                # If it's a dictionary, extract the actual predictions
                if 'predictions' in predictions:
                    np.save(output_path, predictions['predictions'])
                elif 'logits' in predictions:
                    np.save(output_path, predictions['logits'])
                else:
                    # Save the first array in the dict
                    key = next(iter(predictions))
                    np.save(output_path, predictions[key])
            else:
                np.save(output_path, predictions)
            
            return output_path
        else:
            # Return the predictions array
            if isinstance(predictions, dict):
                if 'predictions' in predictions:
                    return predictions['predictions']
                elif 'logits' in predictions:
                    return predictions['logits']
                else:
                    return next(iter(predictions.values()))
            else:
                return predictions
                
    elif to_format == 'hdf5':
        if output_path is None:
            raise ValueError("output_path must be provided for HDF5 format")
            
        if not output_path.endswith('.h5') and not output_path.endswith('.hdf5'):
            output_path += '.h5'
            
        with h5py.File(output_path, 'w') as f:
            if isinstance(predictions, dict):
                # Save each key in the dictionary
                for key, value in predictions.items():
                    f.create_dataset(key, data=value)
            else:
                # Save as 'predictions' dataset
                f.create_dataset('predictions', data=predictions)
                
            # Add metadata if provided
            if metadata:
                metadata_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata_group.attrs[key] = value
                    else:
                        try:
                            metadata_group.create_dataset(key, data=value)
                        except Exception as e:
                            logger.warning(f"Could not save metadata '{key}': {str(e)}")
            
        return output_path
        
    elif to_format == 'csv':
        if output_path is None:
            raise ValueError("output_path must be provided for CSV format")
            
        if not output_path.endswith('.csv'):
            output_path += '.csv'
            
        if isinstance(predictions, dict):
            if 'predictions' in predictions:
                data = predictions['predictions']
            elif 'logits' in predictions:
                data = predictions['logits']
            else:
                # Use the first array in the dict
                data = next(iter(predictions.values()))
        else:
            data = predictions
            
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        
        # Add metadata as columns if provided
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == len(df):
                    df[key] = value
        
        df.to_csv(output_path, index=False)
        return output_path
    
    else:
        raise ValueError(f"Unsupported target format: {to_format}")


def _load_uavarprior_predictions(file_path: str) -> Dict:
    """
    Load predictions from UAVarPrior format.
    
    Args:
        file_path: Path to the predictions file
        
    Returns:
        Dictionary containing the predictions
    """
    # UAVarPrior may use various formats
    try:
        if file_path.endswith('.npy'):
            return {'predictions': np.load(file_path)}
        elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            with h5py.File(file_path, 'r') as f:
                return {key: f[key][()] for key in f.keys()}
        elif file_path.endswith('.csv'):
            return {'predictions': pd.read_csv(file_path).values}
        elif file_path.endswith('.pt') or file_path.endswith('.pth'):
            return {'predictions': torch.load(file_path, map_location=torch.device('cpu')).numpy()}
        else:
            raise ValueError(f"Unsupported file extension for UAVarPrior predictions: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading UAVarPrior predictions: {str(e)}")


def _load_fugep_predictions(file_path: str) -> Dict:
    """
    Load predictions from FuGEP format.
    
    Args:
        file_path: Path to the predictions file
        
    Returns:
        Dictionary containing the predictions
    """
    # FuGEP typically uses HDF5 format with specific structure
    try:
        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            with h5py.File(file_path, 'r') as f:
                result = {}
                
                # FuGEP may have a specific structure
                if 'predictions' in f:
                    result['predictions'] = f['predictions'][()]
                    
                    # Load metadata if available
                    if 'metadata' in f:
                        for key in f['metadata'].keys():
                            result[f'metadata/{key}'] = f['metadata'][key][()]
                else:
                    # Load all datasets
                    for key in f.keys():
                        result[key] = f[key][()]
                        
                return result
        elif file_path.endswith('.npy'):
            return {'predictions': np.load(file_path)}
        elif file_path.endswith('.csv'):
            return {'predictions': pd.read_csv(file_path).values}
        else:
            raise ValueError(f"Unsupported file extension for FuGEP predictions: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading FuGEP predictions: {str(e)}")


def save_predictions(
    predictions: torch.Tensor,
    output_path: str,
    format: str = 'hdf5',
    metadata: Optional[Dict] = None,
    sequence_ids: Optional[List] = None,
    target_names: Optional[List[str]] = None
) -> str:
    """
    Save model predictions to file.
    
    Args:
        predictions: Model predictions tensor
        output_path: Path to save the predictions
        format: Output format ('hdf5', 'numpy', 'csv')
        metadata: Optional metadata to include
        sequence_ids: Optional list of sequence identifiers
        target_names: Optional list of target names for columns
        
    Returns:
        Path to the saved predictions
    """
    # Convert tensor to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.cpu().numpy()
    else:
        predictions_np = predictions
    
    # Prepare metadata
    all_metadata = {}
    if metadata:
        all_metadata.update(metadata)
    
    if sequence_ids is not None:
        all_metadata['sequence_ids'] = sequence_ids
    
    # Save in the specified format
    if format.lower() == 'hdf5':
        if not output_path.endswith('.h5') and not output_path.endswith('.hdf5'):
            output_path += '.h5'
            
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('predictions', data=predictions_np)
            
            # Add metadata
            if all_metadata or target_names:
                metadata_group = f.create_group('metadata')
                
                if target_names:
                    metadata_group.create_dataset('target_names', data=np.array(target_names, dtype=h5py.special_dtype(vlen=str)))
                
                for key, value in all_metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata_group.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        if isinstance(value[0], str):
                            metadata_group.create_dataset(key, data=np.array(value, dtype=h5py.special_dtype(vlen=str)))
                        else:
                            metadata_group.create_dataset(key, data=value)
                    else:
                        try:
                            metadata_group.attrs[key] = str(value)
                        except Exception as e:
                            logger.warning(f"Could not save metadata '{key}': {str(e)}")
    
    elif format.lower() == 'numpy':
        if not output_path.endswith('.npy'):
            output_path += '.npy'
            
        np.save(output_path, predictions_np)
        
        # Save metadata in a separate file if provided
        if all_metadata or target_names:
            metadata_path = output_path.replace('.npy', '_metadata.npz')
            
            save_dict = all_metadata.copy()
            if target_names:
                save_dict['target_names'] = target_names
                
            np.savez(metadata_path, **save_dict)
    
    elif format.lower() == 'csv':
        if not output_path.endswith('.csv'):
            output_path += '.csv'
            
        # Create DataFrame
        df = pd.DataFrame(predictions_np)
        
        # Set column names if provided
        if target_names:
            df.columns = target_names
            
        # Add metadata as columns
        for key, value in all_metadata.items():
            if isinstance(value, (list, np.ndarray)) and len(value) == len(df):
                df[key] = value
        
        df.to_csv(output_path, index=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output_path


def combine_predictions(
    prediction_files: List[str],
    output_path: str,
    format: str = 'hdf5'
) -> str:
    """
    Combine predictions from multiple files.
    
    Args:
        prediction_files: List of paths to prediction files
        output_path: Path to save the combined predictions
        format: Output format ('hdf5', 'numpy', 'csv')
        
    Returns:
        Path to the combined predictions
    """
    all_predictions = []
    all_metadata = {}
    
    for file_path in prediction_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prediction file not found: {file_path}")
        
        # Load predictions based on file extension
        if file_path.endswith('.npy'):
            preds = np.load(file_path)
            all_predictions.append(preds)
            
            # Check for metadata file
            metadata_path = file_path.replace('.npy', '_metadata.npz')
            if os.path.exists(metadata_path):
                metadata = dict(np.load(metadata_path))
                all_metadata.update(metadata)
                
        elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            with h5py.File(file_path, 'r') as f:
                if 'predictions' in f:
                    preds = f['predictions'][()]
                    all_predictions.append(preds)
                    
                    # Load metadata
                    if 'metadata' in f:
                        metadata_group = f['metadata']
                        for key in metadata_group.keys():
                            all_metadata[key] = metadata_group[key][()]
                        
                        # Load attributes
                        for key, value in metadata_group.attrs.items():
                            all_metadata[key] = value
                else:
                    raise ValueError(f"Could not find predictions dataset in {file_path}")
                    
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            
            # Infer which columns are predictions vs metadata
            metadata_columns = []
            for col in df.columns:
                if col.startswith('metadata_') or col in ['sequence_id', 'chrm', 'pos', 'ref', 'alt']:
                    metadata_columns.append(col)
            
            # Extract predictions
            pred_cols = [col for col in df.columns if col not in metadata_columns]
            preds = df[pred_cols].values
            all_predictions.append(preds)
            
            # Extract metadata
            for col in metadata_columns:
                all_metadata[col] = df[col].values
        
        else:
            raise ValueError(f"Unsupported file extension for predictions: {file_path}")
    
    # Combine predictions
    combined_predictions = np.vstack(all_predictions)
    
    # Save combined predictions
    return save_predictions(
        predictions=combined_predictions,
        output_path=output_path,
        format=format,
        metadata=all_metadata
    )