# web_similarity/main_indexer.py
import os
import sys
import json
import time
import logging
import numpy as np
from page_processor import PageProcessor, get_url_hash
from feature_extractor import extract_lc_features, create_lc_map, extract_hog_features
from spm import build_spm
from config import (INDEX_DIR, INDEX_INFO_FILE, SPM_LEVELS, NUM_LC_FEATURES,
                    NUM_HOG_FEATURES)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_index_info():
    """Loads or creates the index info file."""
    if os.path.exists(INDEX_INFO_FILE):
        with open(INDEX_INFO_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_index_info(index_info):
    """Saves the index info file."""
    with open(INDEX_INFO_FILE, 'w') as f:
        json.dump(index_info, f, indent=2)

def process_url(processor, url):
    """Processes a single URL and saves its features."""
    url_hash = get_url_hash(url)
    logging.info(f"Processing URL: {url}")

    # Process the URL
    process_result = processor.process_url(url)
    if not process_result:
        logging.error(f"Failed to process URL: {url}")
        return False

    processed_image, adjusted_elements, _, _ = process_result

    # Extract LC features
    layout_components = extract_lc_features(adjusted_elements)
    lc_map = create_lc_map(layout_components)
    lc_spm = build_spm(lc_map, SPM_LEVELS, NUM_LC_FEATURES)

    # Extract HOG features
    base_hog_histograms = extract_hog_features(processed_image, layout_components)
    if base_hog_histograms is None:
        logging.error(f"Failed to extract HOG features for {url}")
        return False
    hog_spm = build_spm(base_hog_histograms, SPM_LEVELS, NUM_HOG_FEATURES)

    if lc_spm is None or hog_spm is None:
        logging.error(f"Failed to build SPM vectors for {url}")
        return False

    # Save features
    lc_path = os.path.join(INDEX_DIR, f"{url_hash}_lc.npy")
    hog_path = os.path.join(INDEX_DIR, f"{url_hash}_hog.npy")
    np.save(lc_path, lc_spm)
    np.save(hog_path, hog_spm)

    return True

def main():
    # Create index directory if it doesn't exist
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)

    # Load existing index info
    index_info = load_index_info()

    # List of URLs to index
    urls_to_index = [
        "http://example.com",
        "https://www.google.com",
        "https://www.wikipedia.org",
        "https://github.com",
        # Yeni eklenecek URL'ler
        "https://www.youtube.com",
        "https://www.amazon.com",
        "https://www.twitter.com",
        "https://www.linkedin.com",
        "https://www.instagram.com",
        # Aynı sitelerin farklı endpoint'leri
        "https://github.com/about",
        "https://github.com/features",
        "https://www.amazon.com/gp/bestsellers",
        "https://www.amazon.com/gp/help/customer/display.html",
        "https://en.wikipedia.org/wiki/Main_Page",
        "https://en.wikipedia.org/wiki/Special:Random",
        # Haber siteleri
        "https://www.bbc.com",
        "https://www.bbc.com/news",
        "https://www.bbc.com/sport",
        "https://www.cnn.com",
        "https://www.cnn.com/world",
        "https://www.cnn.com/politics"
    ]

    # Initialize WebDriver once
    processor = PageProcessor(headless=True)
    start_time = time.time()
    processed_count = 0

    try:
        for url in urls_to_index:
            url_hash = get_url_hash(url)
            
            # Skip if already indexed
            if url_hash in index_info:
                logging.info(f"Skipping already indexed URL: {url}")
                continue

            if process_url(processor, url):
                index_info[url_hash] = url
                processed_count += 1
                logging.info(f"Successfully indexed: {url}")

    finally:
        # Always close the WebDriver
        processor.close()

    # Save updated index info
    save_index_info(index_info)

    # Print summary
    total_time = time.time() - start_time
    logging.info("--- Indexing Complete ---")
    logging.info(f"Processed {processed_count}/{len(urls_to_index)} URLs.")
    logging.info(f"Total time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()