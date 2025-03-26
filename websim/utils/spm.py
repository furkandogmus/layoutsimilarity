"""
Spatial Pyramid Matching (SPM) module.

This module implements SPM for multi-resolution feature comparison, creating 
feature pyramids for efficient matching.
"""

import numpy as np
from typing import Optional, Tuple, List, Union


def build_spm(base_data: np.ndarray, num_levels: int, num_features: int) -> Optional[np.ndarray]:
    """Build a Spatial Pyramid feature vector from the base data.
    
    Args:
        base_data: Base data at the finest level (2D or 3D array)
        num_levels: Number of levels in the pyramid
        num_features: Number of features per cell
        
    Returns:
        1D feature vector with pyramid levels concatenated, or None if error
    """
    try:
        # Check dimensionality of base_data
        if base_data.ndim == 2:
            # For 2D data, like LC maps, reshape to 3D with feature dimension
            data = np.zeros((base_data.shape[0], base_data.shape[1], num_features), dtype=np.float32)
            # One-hot encode the category labels
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    if base_data[r, c] < num_features:
                        data[r, c, base_data[r, c]] = 1.0
        else:
            # For 3D data, like HOG histograms, use as is
            data = base_data

        # Create an empty vector to hold all pyramid features
        # Initialize with zeros
        total_vector_size = _calculate_total_spm_size(data.shape[0], data.shape[1], 
                                                    num_levels, num_features)
        spm_vector = np.zeros(total_vector_size, dtype=np.float32)
        
        # Fill the vector with pyramid level features
        _fill_spm_levels(spm_vector, data, num_levels, num_features)
        
        return spm_vector
    
    except Exception as e:
        print(f"Error building SPM: {e}")
        return None


def _calculate_total_spm_size(height: int, width: int, num_levels: int, 
                             num_features: int) -> int:
    """Calculate the total size of the SPM feature vector.
    
    Args:
        height: Height of the base data grid
        width: Width of the base data grid
        num_levels: Number of levels in the pyramid
        num_features: Number of features per cell
        
    Returns:
        Total size of the SPM feature vector
    """
    total_size = 0
    for level in range(num_levels):
        # Calculate grid size at this level
        grid_h = max(1, height // (2 ** (num_levels - 1 - level)))
        grid_w = max(1, width // (2 ** (num_levels - 1 - level)))
        # Add size for this level
        level_size = grid_h * grid_w * num_features
        total_size += level_size
    return total_size


def _fill_spm_levels(spm_vector: np.ndarray, data: np.ndarray, num_levels: int, 
                    num_features: int) -> None:
    """Fill the SPM vector with features from all pyramid levels.
    
    Args:
        spm_vector: Vector to fill with SPM features
        data: Base data at the finest level (3D array)
        num_levels: Number of levels in the pyramid
        num_features: Number of features per cell
    """
    offset = 0
    height, width = data.shape[0], data.shape[1]
    
    for level in range(num_levels):
        # Calculate grid size at this level
        grid_h = max(1, height // (2 ** (num_levels - 1 - level)))
        grid_w = max(1, width // (2 ** (num_levels - 1 - level)))
        
        # Calculate cell size (how many fine cells per grid cell)
        cell_h = height // grid_h
        cell_w = width // grid_w
        
        # Iterate through grid cells at this level
        for gh in range(grid_h):
            for gw in range(grid_w):
                # Calculate feature values for this grid cell
                # by pooling from finer cells
                y_start = gh * cell_h
                y_end = min((gh + 1) * cell_h, height)
                x_start = gw * cell_w
                x_end = min((gw + 1) * cell_w, width)
                
                # Extract region and calculate average feature values
                region = data[y_start:y_end, x_start:x_end, :]
                cell_features = np.mean(region, axis=(0, 1))
                
                # Place features in the SPM vector
                spm_vector[offset:offset + num_features] = cell_features
                offset += num_features 