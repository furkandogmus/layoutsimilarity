#!/usr/bin/env python3
# compare_urls_custom.py - Compare similarity between two URLs with custom weights

import argparse
import logging
import json
import os
import numpy as np
from page_processor import PageProcessor
from feature_extractor import extract_lc_features, create_lc_map, extract_hog_features
from spm import build_spm
from similarity import spm_kernel_direct_hik
import config

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_features_from_url(url):
    """Process a URL and extract features."""
    logger.info(f"Processing URL: {url}")
    
    # Process the page
    processor = PageProcessor(headless=True)
    process_result = processor.process_url(url)
    processor.close()
    
    if not process_result:
        logger.error(f"Failed to process URL: {url}")
        return None
    
    processed_image, adjusted_elements, _, _ = process_result
    
    # Extract LC
    layout_components = extract_lc_features(adjusted_elements)
    lc_map = create_lc_map(layout_components)
    lc_spm = build_spm(lc_map, config.SPM_LEVELS, config.NUM_LC_FEATURES)
    
    # Extract HOG
    base_hog_histograms = extract_hog_features(processed_image, layout_components)
    if base_hog_histograms is None:
        logger.error("Failed to extract HOG features.")
        return None
    
    hog_spm = build_spm(base_hog_histograms, config.SPM_LEVELS, config.NUM_HOG_FEATURES)
    
    if lc_spm is None or hog_spm is None:
        logger.error("Failed to build SPM vectors.")
        return None
    
    logger.info("Features extracted successfully.")
    return lc_spm, hog_spm

def compare_urls(url1, url2, weight_lc=0.5, weight_hog=0.5, scaling_factor=3.0):
    """Compare two URLs with custom weights."""
    # Load features for both URLs
    features1 = load_features_from_url(url1)
    features2 = load_features_from_url(url2)
    
    if features1 is None or features2 is None:
        logger.error("Failed to extract features for one or both URLs")
        return
    
    # Calculate similarities
    lc_sim = spm_kernel_direct_hik(features1[0], features2[0])
    hog_sim = spm_kernel_direct_hik(features1[1], features2[1])
    
    # Combine scores with custom weights and apply scaling
    combined_score = (weight_lc * lc_sim) + (weight_hog * hog_sim)
    scaled_score = 2 / (1 + np.exp(-scaling_factor * combined_score)) - 1
    
    # Print results
    print(f"\n--- Similarity Result (Custom Weights) ---")
    print(f"URL 1: {url1}")
    print(f"URL 2: {url2}")
    print(f"Weights: LC={weight_lc:.2f}, HOG={weight_hog:.2f}, Scaling Factor={scaling_factor:.2f}")
    print(f"LC Similarity: {lc_sim:.4f}")
    print(f"HOG Similarity: {hog_sim:.4f}")
    print(f"Combined Similarity Score: {scaled_score:.4f}")
    
    # Interpret the similarity
    if scaled_score > 0.95:
        print("Interpretation: Very high similarity - almost identical pages")
    elif scaled_score > 0.85:
        print("Interpretation: High similarity - similar layout and content")
    elif scaled_score > 0.75:
        print("Interpretation: Moderate similarity - some common elements")
    elif scaled_score > 0.65:
        print("Interpretation: Low similarity - different content but some common patterns")
    else:
        print("Interpretation: Very low similarity - significantly different pages")

def main():
    parser = argparse.ArgumentParser(description="Compare similarity between two URLs with custom weights")
    parser.add_argument("url1", help="First URL to compare")
    parser.add_argument("url2", help="Second URL to compare")
    parser.add_argument("--weight-lc", type=float, default=0.5, help="Weight for layout component similarity (default: 0.5)")
    parser.add_argument("--weight-hog", type=float, default=0.5, help="Weight for HOG similarity (default: 0.5)")
    parser.add_argument("--scaling", type=float, default=3.0, help="Scaling factor for sigmoid (default: 3.0)")
    args = parser.parse_args()
    
    # Normalize weights to sum to 1
    total_weight = args.weight_lc + args.weight_hog
    if total_weight == 0:
        logger.error("Weights cannot both be zero")
        return
    
    weight_lc = args.weight_lc / total_weight
    weight_hog = args.weight_hog / total_weight
    
    compare_urls(args.url1, args.url2, weight_lc, weight_hog, args.scaling)

if __name__ == "__main__":
    main() 