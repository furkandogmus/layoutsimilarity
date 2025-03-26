"""
Configuration settings for the Web Visual Similarity Engine.

This module contains all the configuration parameters used across the application.
"""

import os
import numpy as np

# --- Directory Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "feature_index")
INDEX_INFO_FILE = os.path.join(BASE_DIR, "index_info.json")

# --- Processing Configuration ---
TARGET_SIZE = 1024  # Target dimension for analysis (top 1024x1024 region)
SELENIUM_WAIT_TIME = 5  # Wait time for page load (seconds)
SCREENSHOT_WIDTH = 1920  # Screenshot width for WebDriver

# --- Layout Component (LC) Configuration ---
LC_CATEGORIES = {
    'whitespace': 0,  # Using 0 for easier map initialization
    'text': 1,
    'image': 2,
    'form': 3,
    'animation': 4,  # Basic animation detection (e.g., video, maybe gif)
}
NUM_LC_FEATURES = len(LC_CATEGORIES)
MIN_ELEMENT_SIZE = 8  # Minimum element size to consider (pixels)

# --- HOG Configuration ---
HOG_ORIENTATIONS = 9  # Standard in HOG literature
HOG_WHITESPACE_BIN = True  # Enable the 10th bin concept for whitespace
NUM_HOG_FEATURES = HOG_ORIENTATIONS + (1 if HOG_WHITESPACE_BIN else 0)

# --- Spatial Pyramid Matching (SPM) Configuration ---
SPM_LEVELS = 4  # Number of levels (L in the paper, e.g., 0, 1, 2, 3 for L=4)

# Calculate HOG cell size based on SPM levels
HOG_CELL_SIZE = TARGET_SIZE // (2**(SPM_LEVELS - 1))
HOG_PIXELS_PER_CELL = (HOG_CELL_SIZE, HOG_CELL_SIZE)
HOG_CELLS_PER_BLOCK = (2, 2)  # Affects normalization
HOG_BLOCK_NORM = 'L2-Hys'  # As mentioned in related literature

# Gradient magnitude threshold to consider a cell as 'whitespace' for HOG
HOG_WHITESPACE_MAG_THRESHOLD = 0.25

# --- Query Configuration ---
TOP_N_RESULTS = 3  # Number of top results to show

# --- Feature Combination Weights (for final score) ---
WEIGHT_LC = 0.7  # Weight for layout component features
WEIGHT_HOG = 0.3  # Weight for HOG features

# --- Ensure weights sum to 1.0 ---
total_weight = WEIGHT_LC + WEIGHT_HOG
if total_weight != 0:
    WEIGHT_LC = WEIGHT_LC / total_weight
    WEIGHT_HOG = WEIGHT_HOG / total_weight 