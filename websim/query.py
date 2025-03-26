"""
URL querying module for web visual similarity engine.

This module provides functionality to query the index for similar web pages
based on visual features extracted from a query URL.
"""

import os
import json
import logging
import heapq
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .core.page_processor import PageProcessor, get_url_hash
from .core.feature_extractor import extract_lc_features, create_lc_map, extract_hog_features
from .utils.spm import build_spm
from .core.similarity import spm_kernel_direct_hik
from .config import (
    INDEX_DIR, INDEX_INFO_FILE, SPM_LEVELS, 
    NUM_LC_FEATURES, NUM_HOG_FEATURES,
    WEIGHT_LC, WEIGHT_HOG, TOP_N_RESULTS
)

# Set up module logger
logger = logging.getLogger(__name__)


class WebQuerier:
    """Query for similar web pages based on visual features."""
    
    def __init__(self):
        """Initialize the querier with default configuration."""
        # Load the feature index
        self.index_data, self.index_info = self._load_index()
        
    def _load_index(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, str]]:
        """Load the feature index from disk.
        
        Returns:
            Tuple of:
                - Dictionary mapping URL hashes to feature data
                - Dictionary mapping URL hashes to original URLs
        """
        index_data = {}
        index_info = {}
        
        if not os.path.exists(INDEX_DIR):
            logger.error(f"Index directory not found: {INDEX_DIR}")
            return {}, {}
        
        try:
            with open(INDEX_INFO_FILE, 'r') as f:
                index_info = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load index info: {e}")
            return {}, {}
        
        logger.info(f"Loading features for {len(index_info)} indexed pages...")
        loaded_count = 0
        
        # Load features in parallel
        def load_features(url_hash):
            try:
                lc_path = os.path.join(INDEX_DIR, f"{url_hash}_lc.npy")
                hog_path = os.path.join(INDEX_DIR, f"{url_hash}_hog.npy")
                
                if os.path.exists(lc_path) and os.path.exists(hog_path):
                    lc_spm = np.load(lc_path)
                    hog_spm = np.load(hog_path)
                    return url_hash, {'lc_spm': lc_spm, 'hog_spm': hog_spm, 'url': index_info[url_hash]}
                return None
            except Exception as e:
                logger.warning(f"Could not load features for {index_info.get(url_hash, url_hash)}: {e}")
                return None
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(load_features, url_hash) for url_hash in index_info]
            for future in futures:
                result = future.result()
                if result:
                    url_hash, data = result
                    index_data[url_hash] = data
                    loaded_count += 1
        
        logger.info(f"Successfully loaded features for {loaded_count} pages.")
        return index_data, index_info
    
    def extract_query_features(self, query_url: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract features from the query URL.
        
        Args:
            query_url: URL to extract features from
            
        Returns:
            Tuple of LC and HOG feature vectors, or None if extraction fails
        """
        processor = PageProcessor(headless=True)
        logger.info(f"Processing query URL: {query_url}")
        
        process_result = processor.process_url(query_url)
        processor.close()
        
        if not process_result:
            logger.error(f"Failed to process query URL: {query_url}")
            return None
        
        adjusted_elements, processed_image, _, _ = process_result
        
        # Extract LC features
        layout_components = extract_lc_features(adjusted_elements)
        lc_map = create_lc_map(layout_components)
        query_lc_spm = build_spm(lc_map, SPM_LEVELS, NUM_LC_FEATURES)
        
        # Extract HOG features
        base_hog_histograms = extract_hog_features(processed_image, layout_components)
        if base_hog_histograms is None:
            logger.error("Failed to extract HOG features for query.")
            return None
        
        query_hog_spm = build_spm(base_hog_histograms, SPM_LEVELS, NUM_HOG_FEATURES)
        
        if query_lc_spm is None or query_hog_spm is None:
            logger.error("Failed to build SPM vectors for query.")
            return None
        
        logger.info("Query features extracted successfully.")
        return query_lc_spm, query_hog_spm
    
    def calculate_similarity(self, query_features: Tuple[np.ndarray, np.ndarray], 
                          corpus_features: Dict[str, np.ndarray]) -> float:
        """Calculate similarity between query and corpus features.
        
        Args:
            query_features: Tuple of LC and HOG feature vectors from query
            corpus_features: Dictionary containing corpus feature vectors
            
        Returns:
            Combined similarity score
        """
        query_lc_spm, query_hog_spm = query_features
        corpus_lc_spm = corpus_features['lc_spm']
        corpus_hog_spm = corpus_features['hog_spm']
        
        # Calculate similarities
        sim_lc = spm_kernel_direct_hik(query_lc_spm, corpus_lc_spm)
        sim_hog = spm_kernel_direct_hik(query_hog_spm, corpus_hog_spm)
        
        # Combine scores with weights
        combined_score = (WEIGHT_LC * sim_lc) + (WEIGHT_HOG * sim_hog)
        
        # Apply sigmoid-like scaling for better discrimination
        scaled_score = 2 / (1 + np.exp(-3 * combined_score)) - 1
        
        return scaled_score
    
    def query(self, query_url: str) -> List[Tuple[float, str, bool]]:
        """Find similar web pages to the query URL.
        
        Args:
            query_url: URL to find similar pages for
            
        Returns:
            List of tuples containing (similarity_score, url, is_self_match)
        """
        if not self.index_data:
            logger.error("No index data available. Please run the indexer first.")
            return []
        
        query_features = self.extract_query_features(query_url)
        if query_features is None:
            logger.error("Could not extract features from query URL.")
            return []
        
        query_hash = get_url_hash(query_url)
        seen_urls = set()  # Keep track of seen URLs
        
        logger.info("Calculating similarities against index...")
        
        # Calculate similarities in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            url_to_hash = {}  # Map URLs to their hashes
            
            for url_hash, data in self.index_data.items():
                if data['url'] not in seen_urls:
                    futures.append(
                        executor.submit(self.calculate_similarity, query_features, data)
                    )
                    seen_urls.add(data['url'])
                    url_to_hash[data['url']] = url_hash
            
            # Collect results
            seen_urls_list = list(seen_urls)
            all_scores = []
            
            for i, future in enumerate(futures):
                try:
                    score = future.result()
                    url = seen_urls_list[i]
                    # Mark if this is the query URL itself
                    is_self = (url_to_hash[url] == query_hash)
                    
                    # Boost scores for self-matches to ensure they appear at the top
                    if is_self:
                        score = 0.99
                    
                    all_scores.append((score, url, is_self))
                except Exception as e:
                    logger.error(f"Error calculating similarity: {e}")
        
        # Sort by score (descending)
        all_scores.sort(reverse=True)
        
        # Make sure self-match is included
        has_self = any(item[2] for item in all_scores)
        results = []
        
        if has_self:
            # Add self-match first
            self_item = next((item for item in all_scores if item[2]), None)
            if self_item:
                results.append(self_item)
            
            # Add non-self matches up to TOP_N_RESULTS
            for item in all_scores:
                if not item[2] and len(results) < TOP_N_RESULTS:
                    results.append(item)
        else:
            # No self-match, just take top N
            results = all_scores[:TOP_N_RESULTS]
        
        return results
    
    def format_results(self, query_url: str, results: List[Tuple[float, str, bool]]) -> str:
        """Format query results as a string.
        
        Args:
            query_url: The original query URL
            results: List of result tuples from query()
            
        Returns:
            Formatted string with query results
        """
        if not results:
            return f"No similar pages found for: {query_url}"
        
        output = [f"\n--- Top {len(results)} Results for: {query_url} ---"]
        
        for i, (score, url, is_self) in enumerate(results):
            self_marker = " (QUERY URL)" if is_self else ""
            output.append(f"{i+1}. Score: {score:.4f} - URL: {url}{self_marker}")
        
        return "\n".join(output)


# Simple command-line interface for querying
def main():
    """Command-line interface for querying similar URLs."""
    import argparse
    
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Query for visually similar web pages")
    parser.add_argument("query_url", help="URL to find similar pages for")
    
    args = parser.parse_args()
    
    querier = WebQuerier()
    results = querier.query(args.query_url)
    print(querier.format_results(args.query_url, results))


if __name__ == "__main__":
    main() 