"""
URL comparison module for web visual similarity engine.

This module provides functionality to directly compare two URLs based on
their visual features, without using the index.
"""

import os
import logging
import argparse
from typing import Dict, Tuple, Optional, Any

import numpy as np

from .core.page_processor import PageProcessor
from .core.feature_extractor import extract_lc_features, create_lc_map, extract_hog_features
from .utils.spm import build_spm
from .core.similarity import spm_kernel_direct_hik
from .config import (
    SPM_LEVELS, NUM_LC_FEATURES, NUM_HOG_FEATURES,
    WEIGHT_LC, WEIGHT_HOG
)

# Set up module logger
logger = logging.getLogger(__name__)


class WebComparer:
    """Compare two web pages based on visual features."""
    
    def __init__(self, weight_lc: float = WEIGHT_LC, weight_hog: float = WEIGHT_HOG, 
                scaling_factor: float = 3.0):
        """Initialize the comparer with custom weights.
        
        Args:
            weight_lc: Weight for layout component similarity (default from config)
            weight_hog: Weight for HOG similarity (default from config)
            scaling_factor: Scaling factor for sigmoid function
        """
        # Normalize weights to sum to 1
        total_weight = weight_lc + weight_hog
        if total_weight <= 0:
            raise ValueError("Weights must be positive and at least one must be non-zero")
        
        self.weight_lc = weight_lc / total_weight
        self.weight_hog = weight_hog / total_weight
        self.scaling_factor = scaling_factor
    
    def extract_features(self, url: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract features from a URL.
        
        Args:
            url: URL to extract features from
            
        Returns:
            Tuple of LC and HOG feature vectors, or None if extraction fails
        """
        processor = PageProcessor(headless=True)
        logger.info(f"Processing URL: {url}")
        
        process_result = processor.process_url(url)
        processor.close()
        
        if not process_result:
            logger.error(f"Failed to process URL: {url}")
            return None
        
        adjusted_elements, processed_image, _, _ = process_result
        
        # Extract LC features
        layout_components = extract_lc_features(adjusted_elements)
        lc_map = create_lc_map(layout_components)
        lc_spm = build_spm(lc_map, SPM_LEVELS, NUM_LC_FEATURES)
        
        # Extract HOG features
        base_hog_histograms = extract_hog_features(processed_image, layout_components)
        if base_hog_histograms is None:
            logger.error(f"Failed to extract HOG features for {url}")
            return None
        
        hog_spm = build_spm(base_hog_histograms, SPM_LEVELS, NUM_HOG_FEATURES)
        
        if lc_spm is None or hog_spm is None:
            logger.error(f"Failed to build SPM vectors for {url}")
            return None
        
        logger.info("Features extracted successfully.")
        return lc_spm, hog_spm
    
    def compare(self, url1: str, url2: str) -> Optional[Dict[str, float]]:
        """Compare two URLs directly.
        
        Args:
            url1: First URL to compare
            url2: Second URL to compare
            
        Returns:
            Dictionary of similarity scores, or None if feature extraction fails
        """
        # Extract features for both URLs
        features1 = self.extract_features(url1)
        features2 = self.extract_features(url2)
        
        if features1 is None or features2 is None:
            logger.error("Failed to extract features for one or both URLs")
            return None
        
        # Calculate similarities
        lc_sim = spm_kernel_direct_hik(features1[0], features2[0])
        hog_sim = spm_kernel_direct_hik(features1[1], features2[1])
        
        # Combine scores with weights
        combined_score = (self.weight_lc * lc_sim) + (self.weight_hog * hog_sim)
        
        # Apply sigmoid-like scaling for better discrimination
        scaled_score = 2 / (1 + np.exp(-self.scaling_factor * combined_score)) - 1
        
        return {
            'lc_similarity': lc_sim,
            'hog_similarity': hog_sim,
            'combined_similarity': scaled_score,
            'weights': {
                'lc': self.weight_lc,
                'hog': self.weight_hog
            },
            'scaling_factor': self.scaling_factor
        }
    
    def interpret_similarity(self, similarity: float) -> str:
        """Interpret a similarity score as a human-readable description.
        
        Args:
            similarity: Similarity score
            
        Returns:
            Human-readable interpretation of the similarity
        """
        if similarity > 0.95:
            return "Very high similarity - almost identical pages"
        elif similarity > 0.85:
            return "High similarity - similar layout and content"
        elif similarity > 0.75:
            return "Moderate similarity - some common elements"
        elif similarity > 0.65:
            return "Low similarity - different content but some common patterns"
        else:
            return "Very low similarity - significantly different pages"
    
    def format_comparison(self, url1: str, url2: str, results: Dict[str, Any]) -> str:
        """Format comparison results as a string.
        
        Args:
            url1: First URL compared
            url2: Second URL compared
            results: Dictionary of similarity scores from compare()
            
        Returns:
            Formatted string with comparison results
        """
        if not results:
            return f"Could not compare URLs: {url1} and {url2}"
        
        output = [
            "\n--- Similarity Result ---",
            f"URL 1: {url1}",
            f"URL 2: {url2}",
            f"Weights: LC={results['weights']['lc']:.2f}, HOG={results['weights']['hog']:.2f}, "
            f"Scaling Factor={results['scaling_factor']:.2f}",
            f"LC Similarity: {results['lc_similarity']:.4f}",
            f"HOG Similarity: {results['hog_similarity']:.4f}",
            f"Combined Similarity Score: {results['combined_similarity']:.4f}",
            f"Interpretation: {self.interpret_similarity(results['combined_similarity'])}"
        ]
        
        return "\n".join(output)


# Simple command-line interface for comparing
def main():
    """Command-line interface for comparing URLs."""
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Compare two web pages for visual similarity")
    parser.add_argument("url1", help="First URL to compare")
    parser.add_argument("url2", help="Second URL to compare")
    parser.add_argument("--weight-lc", type=float, default=WEIGHT_LC, 
                       help=f"Weight for layout component similarity (default: {WEIGHT_LC})")
    parser.add_argument("--weight-hog", type=float, default=WEIGHT_HOG, 
                       help=f"Weight for HOG similarity (default: {WEIGHT_HOG})")
    parser.add_argument("--scaling", type=float, default=3.0, 
                       help="Scaling factor for sigmoid (default: 3.0)")
    
    args = parser.parse_args()
    
    comparer = WebComparer(
        weight_lc=args.weight_lc,
        weight_hog=args.weight_hog,
        scaling_factor=args.scaling
    )
    
    results = comparer.compare(args.url1, args.url2)
    print(comparer.format_comparison(args.url1, args.url2, results))


if __name__ == "__main__":
    main() 