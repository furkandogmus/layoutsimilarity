# web_similarity/spm.py
import numpy as np
from config import TARGET_SIZE # Needed? Maybe not directly

def build_spm(base_features, num_levels, num_feature_bins):
    """
    Builds a spatial pyramid histogram vector.

    Args:
        base_features: EITHER the TARGET_SIZExTARGET_SIZE LC map (dtype=int, shape=(H, W))
                       OR the base HOG histograms (shape=(h_cells, w_cells, num_feature_bins)).
                       Assumes base_features HOG corresponds to the grid of the finest SPM level.
        num_levels: Number of pyramid levels (L).
        num_feature_bins: Number of bins per histogram (e.g., NUM_LC_FEATURES or NUM_HOG_FEATURES).

    Returns:
        A flattened numpy array representing the weighted concatenated histograms.
        Returns None if input is invalid.
    """
    pyramid_vectors = []

    is_lc_map = base_features.ndim == 2
    is_hog_base = base_features.ndim == 3

    if not is_lc_map and not is_hog_base:
        print("Error: Invalid base_features shape for SPM.")
        return None

    if is_hog_base:
        finest_level = num_levels - 1
        expected_cells_side = 2**finest_level
        if base_features.shape[0] != expected_cells_side or base_features.shape[1] != expected_cells_side:
            print(f"Error: HOG base grid {base_features.shape[:2]} doesn't match finest SPM level grid ({expected_cells_side}x{expected_cells_side})")
            return None

    base_height, base_width = TARGET_SIZE, TARGET_SIZE

    for level in range(num_levels):
        num_cells_per_side = 2**level
        cell_height_img = base_height // num_cells_per_side
        cell_width_img = base_width // num_cells_per_side

        level_histograms = np.zeros((num_cells_per_side * num_cells_per_side * num_feature_bins), dtype=np.float32)
        hist_idx = 0

        for r_cell in range(num_cells_per_side):
            for c_cell in range(num_cells_per_side):
                y1_img = r_cell * cell_height_img
                y2_img = (r_cell + 1) * cell_height_img
                x1_img = c_cell * cell_width_img
                x2_img = (c_cell + 1) * cell_width_img

                cell_hist = np.zeros(num_feature_bins, dtype=np.float32)

                if is_lc_map:
                    cell_region = base_features[y1_img:y2_img, x1_img:x2_img]
                    hist, _ = np.histogram(cell_region.flatten(),
                                         bins=np.arange(num_feature_bins + 1),
                                         range=(0, num_feature_bins))
                    cell_hist = hist.astype(np.float32)

                elif is_hog_base:
                    finest_cells_side = base_features.shape[0]
                    start_r_fine = r_cell * (finest_cells_side // num_cells_per_side)
                    end_r_fine = (r_cell + 1) * (finest_cells_side // num_cells_per_side)
                    start_c_fine = c_cell * (finest_cells_side // num_cells_per_side)
                    end_c_fine = (c_cell + 1) * (finest_cells_side // num_cells_per_side)

                    cell_hist = np.sum(base_features[start_r_fine:end_r_fine, start_c_fine:end_c_fine, :], axis=(0, 1))

                # L1 Normalize with epsilon to avoid division by zero
                norm = np.sum(cell_hist) + 1e-10
                cell_hist = cell_hist / norm

                level_histograms[hist_idx : hist_idx + num_feature_bins] = cell_hist
                hist_idx += num_feature_bins

        # Simplified weighting scheme for better numerical stability
        if level == 0:
            weight = 1.0 / (2**(num_levels - 1))
        elif level == num_levels - 1:
            weight = 1.0 / 2.0
        else:
            weight = 1.0 / (2**(num_levels - level))

        # Apply weight and ensure no NaN values
        weighted_hist = level_histograms * weight
        weighted_hist = np.nan_to_num(weighted_hist, nan=0.0)
        pyramid_vectors.append(weighted_hist)

    # Concatenate and ensure final vector is valid
    spm_vector = np.concatenate(pyramid_vectors)
    spm_vector = np.nan_to_num(spm_vector, nan=0.0)
    
    # Final L1 normalization
    norm = np.sum(spm_vector) + 1e-10
    spm_vector = spm_vector / norm

    return spm_vector