#!/usr/bin/env python3
"""
Custom URL comparison script with enhanced differentiation.

This script provides a more nuanced comparison between web pages by:
1. Applying stronger scaling to differentiate similar pages
2. Adding content analysis for text-based similarity
3. Providing more detailed breakdowns of similarity scores
"""

import os
import logging
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple

from websim.compare import WebComparer
from websim.core.page_processor import PageProcessor
from websim.core.feature_extractor import extract_lc_features, create_lc_map
from websim.utils.spm import build_spm
from websim.core.similarity import spm_kernel_direct_hik
from websim.config import SPM_LEVELS, NUM_LC_FEATURES

# Configure basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedComparer(WebComparer):
    """Enhanced web page comparer with better discrimination."""
    
    def __init__(self, weight_lc: float = 0.6, weight_hog: float = 0.3, 
                 weight_content: float = 0.1, scaling_factor: float = 6.0):
        """Initialize with custom weights including content analysis.
        
        Args:
            weight_lc: Weight for layout component similarity (default: 0.6)
            weight_hog: Weight for HOG similarity (default: 0.3)
            weight_content: Weight for content/text similarity (default: 0.1)
            scaling_factor: Scaling factor for sigmoid (default: 6.0)
        """
        # Normalize weights to sum to 1
        total_weight = weight_lc + weight_hog + weight_content
        if total_weight <= 0:
            raise ValueError("Weights must be positive and at least one must be non-zero")
        
        self.weight_lc = weight_lc / total_weight
        self.weight_hog = weight_hog / total_weight
        self.weight_content = weight_content / total_weight
        self.scaling_factor = scaling_factor
    
    def extract_content_features(self, url: str) -> np.ndarray:
        """Extract text content features from a URL.
        
        Args:
            url: URL to extract features from
            
        Returns:
            Feature vector based on text content
        """
        processor = PageProcessor(headless=True)
        logger.info(f"Processing URL for content analysis: {url}")
        
        process_result = processor.process_url(url)
        processor.close()
        
        if not process_result:
            logger.error(f"Failed to process URL for content: {url}")
            return np.zeros(10)  # Return empty feature vector
        
        adjusted_elements, _, _, _ = process_result
        
        # Extract text from elements
        text_elements = [elem for elem in adjusted_elements if 'text' in elem and elem.get('text')]
        
        # Simple content analysis - count distribution of characters
        all_text = " ".join([elem.get('text', '') for elem in text_elements])
        
        # Simplified text features - word length distribution
        if not all_text:
            return np.zeros(10)
            
        words = all_text.split()
        if not words:
            return np.zeros(10)
            
        # Create a histogram of word lengths (up to 10 bins)
        word_lengths = [min(len(word), 10) for word in words]
        hist, _ = np.histogram(word_lengths, bins=10, range=(1, 11), density=True)
        
        return hist
    
    def compare(self, url1: str, url2: str) -> Dict[str, Any]:
        """Compare two URLs with enhanced differentiation.
        
        Args:
            url1: First URL to compare
            url2: Second URL to compare
            
        Returns:
            Dictionary of similarity scores
        """
        # Quick check if URLs are identical
        identical_urls = url1 == url2
        
        # Extract standard features
        features1 = self.extract_features(url1)
        features2 = self.extract_features(url2)
        
        # Extract content features
        content_features1 = self.extract_content_features(url1)
        content_features2 = self.extract_content_features(url2)
        
        if features1 is None or features2 is None:
            logger.error("Failed to extract features for one or both URLs")
            return {}
        
        # Calculate standard similarities
        lc_sim = spm_kernel_direct_hik(features1[0], features2[0])
        hog_sim = spm_kernel_direct_hik(features1[1], features2[1])
        
        # Calculate content similarity (cosine similarity)
        content_sim = np.dot(content_features1, content_features2) / (
            np.linalg.norm(content_features1) * np.linalg.norm(content_features2)
        ) if np.linalg.norm(content_features1) * np.linalg.norm(content_features2) > 0 else 0
        
        # Calculate raw combined score 
        raw_score = (
            self.weight_lc * lc_sim + 
            self.weight_hog * hog_sim + 
            self.weight_content * content_sim
        )
        
        # Apply steeper sigmoid scaling for better discrimination
        # If URLs are identical, ensure a very high score
        if identical_urls:
            scaled_score = 0.99
        else:
            sigmoid_input = self.scaling_factor * (raw_score - 0.4)  # Lower threshold to 0.4
            scaled_score = 2 / (1 + np.exp(-sigmoid_input)) - 1
            # Ensure the score is in [0, 1]
            scaled_score = max(0, min(0.98, scaled_score))  # Cap at 0.98 for non-identical URLs
        
        return {
            'lc_similarity': lc_sim,
            'hog_similarity': hog_sim,
            'content_similarity': content_sim,
            'raw_score': raw_score,
            'adjusted_similarity': scaled_score,
            'identical_urls': identical_urls,
            'weights': {
                'lc': self.weight_lc,
                'hog': self.weight_hog,
                'content': self.weight_content
            },
            'scaling_factor': self.scaling_factor
        }
    
    def interpret_similarity(self, similarity: float, identical_urls: bool = False) -> str:
        """Provide a more nuanced interpretation of similarity.
        
        Args:
            similarity: Adjusted similarity score
            identical_urls: Whether URLs being compared are identical
            
        Returns:
            Human-readable interpretation of the similarity
        """
        if identical_urls:
            return "Identical pages - Same URL"
        elif similarity > 0.95:
            return "Near identical - same template and very similar content"
        elif similarity > 0.85:
            return "Very similar - same template with similar content"
        elif similarity > 0.70:
            return "Similar - same template but different content"
        elif similarity > 0.55:
            return "Moderately similar - similar layout patterns but distinct content"
        elif similarity > 0.35:
            return "Somewhat similar - some common design elements but mostly different"
        elif similarity > 0.15:
            return "Mostly different - few shared characteristics"
        else:
            return "Completely different pages"
    
    def format_comparison(self, url1: str, url2: str, results: Dict[str, Any]) -> str:
        """Format comparison results as a detailed string.
        
        Args:
            url1: First URL compared
            url2: Second URL compared
            results: Dictionary of similarity scores from compare()
            
        Returns:
            Formatted string with comparison results
        """
        if not results:
            return f"Could not compare URLs: {url1} and {url2}"
        
        identical_urls = results.get('identical_urls', False)
        
        output = [
            "\n--- Enhanced Similarity Result ---",
            f"URL 1: {url1}",
            f"URL 2: {url2}",
            "\n--- Similarity Breakdown ---",
            f"Layout Structure: {results['lc_similarity']:.4f}",
            f"Visual Appearance: {results['hog_similarity']:.4f}",
            f"Content Analysis: {results['content_similarity']:.4f}",
            f"\nRaw Combined Score: {results['raw_score']:.4f}",
            f"Adjusted Similarity: {results['adjusted_similarity']:.4f}",
            f"\nInterpretation: {self.interpret_similarity(results['adjusted_similarity'], identical_urls)}",
            "\n--- Analysis Details ---",
            f"Weights: Layout={results['weights']['lc']:.2f}, " +
            f"Visual={results['weights']['hog']:.2f}, " +
            f"Content={results['weights']['content']:.2f}",
            f"Scaling Factor: {results['scaling_factor']:.2f}"
        ]
        
        return "\n".join(output)


def main():
    """Command-line interface for enhanced URL comparison."""
    parser = argparse.ArgumentParser(description="Compare two web pages with enhanced differentiation")
    parser.add_argument("url1", help="First URL to compare")
    parser.add_argument("url2", help="Second URL to compare")
    parser.add_argument("--weight-lc", type=float, default=0.6, 
                       help="Weight for layout structure similarity (default: 0.6)")
    parser.add_argument("--weight-hog", type=float, default=0.3, 
                       help="Weight for visual appearance similarity (default: 0.3)")
    parser.add_argument("--weight-content", type=float, default=0.1, 
                       help="Weight for content analysis (default: 0.1)")
    parser.add_argument("--scaling", type=float, default=6.0, 
                       help="Scaling factor for discrimination (default: 6.0)")
    
    args = parser.parse_args()
    
    comparer = EnhancedComparer(
        weight_lc=args.weight_lc,
        weight_hog=args.weight_hog,
        weight_content=args.weight_content,
        scaling_factor=args.scaling
    )
    
    results = comparer.compare(args.url1, args.url2)
    print(comparer.format_comparison(args.url1, args.url2, results))


if __name__ == "__main__":
    main() 