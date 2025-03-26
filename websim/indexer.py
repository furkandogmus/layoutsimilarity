"""
URL indexing module for web visual similarity engine.

This module provides functionality to index web pages by extracting
their visual features and storing them for later comparison.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .core.page_processor import PageProcessor, get_url_hash
from .core.feature_extractor import extract_lc_features, create_lc_map, extract_hog_features
from .utils.spm import build_spm
from .config import (
    INDEX_DIR, INDEX_INFO_FILE, SPM_LEVELS, 
    NUM_LC_FEATURES, NUM_HOG_FEATURES
)

# Set up module logger
logger = logging.getLogger(__name__)


class WebIndexer:
    """Index web pages for similarity comparison."""
    
    def __init__(self):
        """Initialize the indexer with default configuration."""
        # Ensure index directory exists
        if not os.path.exists(INDEX_DIR):
            os.makedirs(INDEX_DIR, exist_ok=True)
        
        # Load existing index info if available
        self.index_info = self._load_index_info()
        
    def _load_index_info(self) -> Dict[str, str]:
        """Load the existing index information.
        
        Returns:
            Dictionary mapping URL hashes to their original URLs
        """
        if not os.path.exists(INDEX_INFO_FILE):
            return {}
            
        try:
            with open(INDEX_INFO_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading index info: {e}")
            return {}
    
    def _save_index_info(self) -> None:
        """Save the current index information to disk."""
        try:
            with open(INDEX_INFO_FILE, 'w') as f:
                json.dump(self.index_info, f, indent=2)
            logger.info(f"Saved index info to {INDEX_INFO_FILE}")
        except IOError as e:
            logger.error(f"Error saving index info: {e}")
    
    def process_url(self, url: str) -> bool:
        """Process a single URL and add it to the index.
        
        Args:
            url: URL to process and index
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Processing URL: {url}")
        url_hash = get_url_hash(url)
        
        # Skip if already indexed
        if url_hash in self.index_info:
            logger.info(f"URL already indexed: {url}")
            return True
        
        # Process the page
        processor = PageProcessor(headless=True)
        process_result = processor.process_url(url)
        processor.close()
        
        if not process_result:
            logger.error(f"Failed to process URL: {url}")
            return False
        
        adjusted_elements, processed_image, _, _ = process_result
        
        # Extract LC features
        layout_components = extract_lc_features(adjusted_elements)
        lc_map = create_lc_map(layout_components)
        lc_spm = build_spm(lc_map, SPM_LEVELS, NUM_LC_FEATURES)
        
        # Extract HOG features
        base_hog_histograms = extract_hog_features(processed_image, layout_components)
        if base_hog_histograms is None:
            logger.error(f"Failed to extract HOG features for {url}")
            return False
        
        hog_spm = build_spm(base_hog_histograms, SPM_LEVELS, NUM_HOG_FEATURES)
        
        if lc_spm is None or hog_spm is None:
            logger.error(f"Failed to build SPM vectors for {url}")
            return False
        
        # Save features to disk
        try:
            lc_path = os.path.join(INDEX_DIR, f"{url_hash}_lc.npy")
            hog_path = os.path.join(INDEX_DIR, f"{url_hash}_hog.npy")
            
            np.save(lc_path, lc_spm)
            np.save(hog_path, hog_spm)
            
            # Update index info
            self.index_info[url_hash] = url
            self._save_index_info()
            
            logger.info(f"Successfully indexed URL: {url}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving index data for {url}: {e}")
            return False
    
    def index_urls(self, urls: List[str], max_workers: int = 4) -> None:
        """Index multiple URLs in parallel.
        
        Args:
            urls: List of URLs to index
            max_workers: Maximum number of parallel workers
        """
        # Filter out already indexed URLs
        new_urls = [url for url in urls if get_url_hash(url) not in self.index_info]
        if not new_urls:
            logger.info("All URLs are already indexed.")
            return
        
        logger.info(f"Indexing {len(new_urls)} URLs with {max_workers} workers")
        
        # Process URLs in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(self.process_url, new_urls),
                total=len(new_urls),
                desc="Indexing URLs"
            ))
        
        # Report results
        success_count = sum(1 for r in results if r)
        logger.info(f"Indexed {success_count} out of {len(new_urls)} URLs")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        stats = {
            "total_urls": len(self.index_info),
            "index_file_size_mb": 0,
            "index_files": []
        }
        
        if os.path.exists(INDEX_DIR):
            files = [f for f in os.listdir(INDEX_DIR) if f.endswith('.npy')]
            stats["index_files"] = files
            
            # Calculate total size
            total_size = sum(os.path.getsize(os.path.join(INDEX_DIR, f)) for f in files)
            stats["index_file_size_mb"] = total_size / (1024 * 1024)
        
        return stats
    
    def clear_index(self) -> None:
        """Clear the entire index."""
        if os.path.exists(INDEX_DIR):
            for file in os.listdir(INDEX_DIR):
                if file.endswith('.npy'):
                    os.remove(os.path.join(INDEX_DIR, file))
        
        if os.path.exists(INDEX_INFO_FILE):
            os.remove(INDEX_INFO_FILE)
        
        self.index_info = {}
        logger.info("Index cleared")


# Simple command-line interface for indexing
def main():
    """Command-line interface for indexing URLs."""
    import argparse
    
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Index web pages for visual similarity")
    parser.add_argument("urls", nargs="*", help="URLs to index")
    parser.add_argument("--file", "-f", help="File containing URLs to index, one per line")
    parser.add_argument("--clear", action="store_true", help="Clear the existing index")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--workers", "-w", type=int, default=4, 
                       help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    indexer = WebIndexer()
    
    if args.clear:
        indexer.clear_index()
        return
    
    if args.stats:
        stats = indexer.get_index_stats()
        print("\nIndex Statistics:")
        print(f"Total URLs indexed: {stats['total_urls']}")
        print(f"Index size: {stats['index_file_size_mb']:.2f} MB")
        print(f"Index files: {len(stats['index_files'])}")
        return
    
    urls_to_index = []
    
    # Add URLs from command line
    if args.urls:
        urls_to_index.extend(args.urls)
    
    # Add URLs from file
    if args.file:
        try:
            with open(args.file, 'r') as f:
                file_urls = [line.strip() for line in f if line.strip()]
                urls_to_index.extend(file_urls)
        except IOError as e:
            logger.error(f"Error reading URL file: {e}")
    
    if not urls_to_index:
        logger.error("No URLs provided. Use --help for usage information.")
        return
    
    # Index the URLs
    indexer.index_urls(urls_to_index, max_workers=args.workers)


if __name__ == "__main__":
    main() 