"""
Similarity calculation module for comparing web page features.

This module provides various similarity metrics for comparing feature vectors,
particularly for layout components and HOG features.
"""

import numpy as np
from typing import Union, List, Tuple

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector as numpy array
        v2: Second vector as numpy array
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    
    # Handle NaN values
    v1 = np.nan_to_num(v1, 0)
    v2 = np.nan_to_num(v2, 0)
    
    # Ensure non-negative values
    v1 = np.maximum(v1, 0)
    v2 = np.maximum(v2, 0)
    
    # Calculate cosine similarity
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity


def euclidean_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate similarity based on Euclidean distance.
    
    Args:
        v1: First vector as numpy array
        v2: Second vector as numpy array
        
    Returns:
        Similarity score based on Euclidean distance (0 to 1)
    """
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    
    # Handle NaN values
    v1 = np.nan_to_num(v1, 0)
    v2 = np.nan_to_num(v2, 0)
    
    # Ensure non-negative values
    v1 = np.maximum(v1, 0)
    v2 = np.maximum(v2, 0)
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(v1 - v2)
    
    # Convert distance to similarity (1 when vectors are identical, 0 when maximally different)
    similarity = 1 / (1 + distance)
    return similarity


def histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """Calculate histogram intersection between two histograms.
    
    Args:
        h1: First histogram as numpy array
        h2: Second histogram as numpy array
        
    Returns:
        Combined similarity score using histogram intersection and other metrics
    """
    h1 = np.array(h1, dtype=np.float64)
    h2 = np.array(h2, dtype=np.float64)
    
    # Handle NaN values
    h1 = np.nan_to_num(h1, 0)
    h2 = np.nan_to_num(h2, 0)
    
    # Ensure non-negative values
    h1 = np.maximum(h1, 0)
    h2 = np.maximum(h2, 0)
    
    # L1 normalize the histograms
    sum1 = np.sum(h1)
    sum2 = np.sum(h2)
    
    if sum1 > 0:
        h1 = h1 / sum1
    if sum2 > 0:
        h2 = h2 / sum2
    
    # Calculate intersection
    intersection = np.minimum(h1, h2)
    intersection_sim = np.sum(intersection)
    
    # Calculate other similarities
    cos_sim = cosine_similarity(h1, h2)
    euc_sim = euclidean_similarity(h1, h2)
    
    # Combine similarities with weights
    combined_sim = 0.5 * intersection_sim + 0.3 * cos_sim + 0.2 * euc_sim
    
    # Apply sigmoid-like scaling for better discrimination
    combined_sim = 2 / (1 + np.exp(-5 * combined_sim)) - 1
    
    return combined_sim


def spm_kernel_direct_hik(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate similarity using Histogram Intersection Kernel (HIK) with combined metrics.
    
    Args:
        x1: First feature vector as numpy array
        x2: Second feature vector as numpy array
        
    Returns:
        Similarity score combining multiple metrics (0 to 1)
    """
    x1 = np.array(x1, dtype=np.float64)
    x2 = np.array(x2, dtype=np.float64)
    
    # Handle NaN values
    x1 = np.nan_to_num(x1, 0)
    x2 = np.nan_to_num(x2, 0)
    
    # Ensure non-negative values
    x1 = np.maximum(x1, 0)
    x2 = np.maximum(x2, 0)
    
    # L1 normalize the vectors
    sum1 = np.sum(x1)
    sum2 = np.sum(x2)
    
    if sum1 > 0:
        x1 = x1 / sum1
    if sum2 > 0:
        x2 = x2 / sum2
    
    # Calculate intersection
    intersection = np.minimum(x1, x2)
    intersection_sim = np.sum(intersection)
    
    # Calculate other similarities
    cos_sim = cosine_similarity(x1, x2)
    euc_sim = euclidean_similarity(x1, x2)
    
    # Combine similarities with weights
    combined_sim = 0.5 * intersection_sim + 0.3 * cos_sim + 0.2 * euc_sim
    
    # Apply sigmoid-like scaling for better discrimination
    combined_sim = 2 / (1 + np.exp(-3 * combined_sim)) - 1
    
    return combined_sim


def spm_kernel_paper_eq2(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate similarity using the original SPM kernel equation.
    
    Args:
        x1: First feature vector as numpy array
        x2: Second feature vector as numpy array
        
    Returns:
        Similarity score using the original SPM paper equation
    """
    x1 = np.array(x1, dtype=np.float64)
    x2 = np.array(x2, dtype=np.float64)
    
    # Handle NaN values
    x1 = np.nan_to_num(x1, 0)
    x2 = np.nan_to_num(x2, 0)
    
    # Ensure non-negative values
    x1 = np.maximum(x1, 0)
    x2 = np.maximum(x2, 0)
    
    # L1 normalize the vectors
    sum1 = np.sum(x1)
    sum2 = np.sum(x2)
    
    if sum1 > 0:
        x1 = x1 / sum1
    if sum2 > 0:
        x2 = x2 / sum2
    
    # Using the paper's equation 2 approach: direct min(x1,x2) sum
    intersection = np.minimum(x1, x2)
    similarity = np.sum(intersection)
    
    return similarity 