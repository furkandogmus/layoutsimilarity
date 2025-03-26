"""
Feature extraction module for extracting layout and visual features from web pages.

This module provides functions to extract layout component (LC) and
Histogram of Oriented Gradients (HOG) features from processed web pages.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional, Union

from ..config import (
    TARGET_SIZE, LC_CATEGORIES, HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, 
    HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM, HOG_WHITESPACE_BIN, NUM_HOG_FEATURES,
    HOG_WHITESPACE_MAG_THRESHOLD
)

# --- Layout Component (LC) Functions ---

def _categorize_element(element_data: Dict[str, Any]) -> int:
    """Categorize a raw element into an LC type.
    
    Args:
        element_data: Dictionary containing element metadata
        
    Returns:
        Integer code representing the element category
    """
    tag = element_data.get('tag', '').upper()
    text = element_data.get('text', '')

    if tag in ['IMG', 'SVG', 'CANVAS']:
        return LC_CATEGORIES['image']
    elif tag in ['VIDEO']:  # Explicit video tag
        return LC_CATEGORIES['animation']
    elif tag in ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'FORM']:
        return LC_CATEGORIES['form']
    elif text:  # If not categorized above but has text
        return LC_CATEGORIES['text']

    # Default fallback if no specific category matches
    return LC_CATEGORIES['text']


def extract_lc_features(adjusted_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process adjusted elements data into layout components with categorized types.
    
    Args:
        adjusted_elements: List of element dictionaries with adjusted coordinates
        
    Returns:
        List of layout components with categorized types and coordinates
    """
    layout_components = []
    for elem in adjusted_elements:
        comp_type = _categorize_element(elem)
        layout_components.append({
            'x': elem['adj_x'],
            'y': elem['adj_y'],
            'w': elem['adj_w'],
            'h': elem['adj_h'],
            'type': comp_type,
            'tag': elem['tag']  # Keep tag for potential HOG prep use
        })
    return layout_components


def create_lc_map(layout_components: List[Dict[str, Any]], 
                 width: int = TARGET_SIZE, 
                 height: int = TARGET_SIZE) -> np.ndarray:
    """Create the Layout Component map (grid).
    
    Args:
        layout_components: List of layout component dictionaries
        width: Width of the output map
        height: Height of the output map
        
    Returns:
        2D numpy array where each cell contains the category code of the component
    """
    # Initialize with whitespace code
    lc_map = np.full((height, width), LC_CATEGORIES['whitespace'], dtype=np.uint8)

    # Draw components onto the map
    for comp in layout_components:
        # Clamp coordinates and dimensions to map boundaries
        x1 = max(0, int(np.round(comp['x'])))
        y1 = max(0, int(np.round(comp['y'])))
        x2 = min(width, int(np.round(comp['x'] + comp['w'])))
        y2 = min(height, int(np.round(comp['y'] + comp['h'])))
        comp_type_code = comp['type']

        if x1 < x2 and y1 < y2:  # Ensure valid rectangle
            lc_map[y1:y2, x1:x2] = comp_type_code

    return lc_map

# --- HOG Feature Functions ---

def _prepare_image_for_hog(image_bgr: np.ndarray, 
                          layout_components: List[Dict[str, Any]]) -> np.ndarray:
    """Prepare image for HOG extraction by modifying text regions and adding guidelines.
    
    Args:
        image_bgr: BGR image as numpy array
        layout_components: List of layout component dictionaries
        
    Returns:
        Modified BGR image ready for HOG feature extraction
    """
    modified_image = image_bgr.copy()

    for comp in layout_components:
        x1 = max(0, int(np.round(comp['x'])))
        y1 = max(0, int(np.round(comp['y'])))
        x2 = min(TARGET_SIZE, int(np.round(comp['x'] + comp['w'])))
        y2 = min(TARGET_SIZE, int(np.round(comp['y'] + comp['h'])))

        if x1 >= x2 or y1 >= y2:
            continue

        if comp['type'] == LC_CATEGORIES['text']:
            # Fill text areas with white to remove text-specific features
            modified_image[y1:y2, x1:x2] = (255, 255, 255)
        elif comp['type'] == LC_CATEGORIES['image']:
            # Draw diagonal line (top-left to bottom-right) - Red
            cv2.line(modified_image, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 1)
        elif comp['type'] == LC_CATEGORIES['animation']:
            # Draw other diagonal (top-right to bottom-left) - Blue
            cv2.line(modified_image, (x2 - 1, y1), (x1, y2 - 1), (255, 0, 0), 1)

    return modified_image


def extract_hog_features(processed_image_bgr: np.ndarray, 
                        layout_components: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Extract HOG features suitable for SPM, including the whitespace bin concept.
    
    Args:
        processed_image_bgr: Processed BGR image as numpy array
        layout_components: List of layout component dictionaries
        
    Returns:
        3D numpy array of shape (num_cells_y, num_cells_x, NUM_HOG_FEATURES)
        representing histograms for the finest SPM level grid, or None if error
    """
    try:
        prepared_image = _prepare_image_for_hog(processed_image_bgr, layout_components)
        gray_image = cv2.cvtColor(prepared_image, cv2.COLOR_BGR2GRAY)

        # Calculate gradients with a small kernel to avoid noise
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude and Angle
        magnitude = cv2.magnitude(grad_x, grad_y)
        angle = cv2.phase(grad_x, grad_y, angleInDegrees=True) % 180

        # Normalize magnitude to [0, 1] range
        magnitude = magnitude / (np.max(magnitude) + 1e-10)

        cell_h, cell_w = HOG_PIXELS_PER_CELL
        num_cells_y = TARGET_SIZE // cell_h
        num_cells_x = TARGET_SIZE // cell_w

        base_hog_histograms = np.zeros((num_cells_y, num_cells_x, NUM_HOG_FEATURES), dtype=np.float32)
        orientation_bin_size = 180.0 / HOG_ORIENTATIONS

        for r_cell in range(num_cells_y):
            for c_cell in range(num_cells_x):
                y1 = r_cell * cell_h
                y2 = y1 + cell_h
                x1 = c_cell * cell_w
                x2 = x1 + cell_w

                mag_patch = magnitude[y1:y2, x1:x2]
                ang_patch = angle[y1:y2, x1:x2]

                # Improved whitespace detection
                avg_magnitude = np.mean(mag_patch)
                is_whitespace_cell = avg_magnitude < HOG_WHITESPACE_MAG_THRESHOLD

                if HOG_WHITESPACE_BIN and is_whitespace_cell:
                    base_hog_histograms[r_cell, c_cell, -1] = 1.0  # Normalized value
                else:
                    # Orientation binning with improved interpolation
                    for r_pix in range(cell_h):
                        for c_pix in range(cell_w):
                            pix_angle = ang_patch[r_pix, c_pix]
                            pix_mag = mag_patch[r_pix, c_pix]

                            bin_float = pix_angle / orientation_bin_size
                            bin_low = int(np.floor(bin_float)) % HOG_ORIENTATIONS
                            bin_high = (bin_low + 1) % HOG_ORIENTATIONS

                            # Improved interpolation weights
                            weight_high = bin_float - bin_low
                            weight_low = 1.0 - weight_high

                            base_hog_histograms[r_cell, c_cell, bin_low] += weight_low * pix_mag
                            base_hog_histograms[r_cell, c_cell, bin_high] += weight_high * pix_mag

        # L1 normalization with epsilon to avoid division by zero
        norm = np.sum(base_hog_histograms, axis=2, keepdims=True) + 1e-10
        base_hog_histograms = np.divide(base_hog_histograms, norm)

        return base_hog_histograms

    except Exception as e:
        print(f"Error extracting HOG features: {e}")
        import traceback
        traceback.print_exc()
        return None 