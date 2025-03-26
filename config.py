# web_similarity/config.py
import numpy as np

# --- Processing Configuration ---
TARGET_SIZE = 1024  # Target dimension for analysis (top 1024x1024 region)
# Wait time for page load (adjust based on network/page complexity)
SELENIUM_WAIT_TIME = 5

# --- Layout Component (LC) Configuration ---
LC_CATEGORIES = {
    # Using 0 for whitespace might be slightly easier for map initialization
    'whitespace': 0,
    'text': 1,
    'image': 2,
    'form': 3,
    'animation': 4, # Basic animation detection (e.g., video, maybe gif)
    # Add more categories if needed
}
NUM_LC_FEATURES = len(LC_CATEGORIES)
# Minimum element size to consider (pixels)
MIN_ELEMENT_SIZE = 8

# --- HOG Configuration ---
# Paper mentions 10 bins (9 orientation + 1 whitespace), let's reflect that
HOG_ORIENTATIONS = 9
HOG_WHITESPACE_BIN = True # Enable the 10th bin concept
NUM_HOG_FEATURES = HOG_ORIENTATIONS + (1 if HOG_WHITESPACE_BIN else 0)

# Note: HOG_PIXELS_PER_CELL should align with the finest SPM level.
# If SPM_LEVELS = 4, finest grid is 8x8 cells -> cell size = 1024/8 = 128
# If SPM_LEVELS = 3, finest grid is 4x4 cells -> cell size = 1024/4 = 256
# Let's default to SPM_LEVELS = 4 for now
_SPM_LEVELS_FOR_HOG = 4 # Define this temporarily to calculate HOG cell size
HOG_PIXELS_PER_CELL = (TARGET_SIZE // (2**(_SPM_LEVELS_FOR_HOG - 1)),
                       TARGET_SIZE // (2**(_SPM_LEVELS_FOR_HOG - 1)))
HOG_CELLS_PER_BLOCK = (2, 2) # Adjust as needed; affects normalization
HOG_BLOCK_NORM = 'L2-Hys' # As mentioned in paper
# Gradient magnitude threshold to consider a cell as 'whitespace' for HOG
HOG_WHITESPACE_MAG_THRESHOLD = 0.25  # Increased threshold for better whitespace detection

# --- Spatial Pyramid Matching (SPM) Configuration ---
SPM_LEVELS = 4  # Number of levels (L in the paper, e.g., 0, 1, 2, 3 for L=4)

# --- Indexing Configuration ---
INDEX_DIR = "feature_index"
INDEX_INFO_FILE = "index_info.json" # Stores URLs corresponding to saved files
URL_HASH_METHOD = "sha1" # Method to create filenames for URLs

# --- Query Configuration ---
TOP_N_RESULTS = 3  # Show fewer results

# --- Feature Combination Weights (for final score) ---
# Adjusted weights to give more importance to layout components
WEIGHT_LC = 0.7  # Reduced from 0.9 for more balanced similarity scores
WEIGHT_HOG = 0.3  # Increased from 0.1 for more balanced similarity scores