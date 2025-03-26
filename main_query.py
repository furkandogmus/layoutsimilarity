# web_similarity/main_query.py
import os
import sys
import json
import numpy as np
import heapq
import logging
from concurrent.futures import ThreadPoolExecutor

from config import (INDEX_DIR, INDEX_INFO_FILE, TOP_N_RESULTS, WEIGHT_LC, WEIGHT_HOG,
                    SPM_LEVELS, NUM_LC_FEATURES, NUM_HOG_FEATURES)
from page_processor import PageProcessor, get_url_hash
from feature_extractor import extract_lc_features, create_lc_map, extract_hog_features
from spm import build_spm
from similarity import spm_kernel_direct_hik

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_index():
    """Loads the pre-computed feature index."""
    index_data = {}
    index_info = {}

    if not os.path.exists(INDEX_DIR):
        logging.error(f"Index directory not found: {INDEX_DIR}")
        return None, None

    try:
        with open(INDEX_INFO_FILE, 'r') as f:
            index_info = json.load(f)
    except FileNotFoundError:
        logging.error(f"Index info file not found: {INDEX_INFO_FILE}")
        return None, None
    except json.JSONDecodeError:
        logging.error(f"Could not decode index info file: {INDEX_INFO_FILE}")
        return None, None

    logging.info(f"Loading features for {len(index_info)} indexed pages...")
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
            logging.warning(f"Could not load features for {index_info[url_hash]}: {e}")
            return None

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_features, url_hash) for url_hash in index_info]
        for future in futures:
            result = future.result()
            if result:
                url_hash, data = result
                index_data[url_hash] = data
                loaded_count += 1

    logging.info(f"Successfully loaded features for {loaded_count} pages.")
    if loaded_count == 0:
        return None, None
    return index_data, index_info

def get_query_features(query_url):
    """Processes the query URL to get its features."""
    processor = PageProcessor(headless=True)
    logging.info(f"Processing query URL: {query_url}")
    process_result = processor.process_url(query_url)
    processor.close()

    if not process_result:
        logging.error(f"Failed to process query URL: {query_url}")
        return None

    processed_image, adjusted_elements, _, _ = process_result

    # Extract LC
    layout_components = extract_lc_features(adjusted_elements)
    lc_map = create_lc_map(layout_components)
    query_lc_spm = build_spm(lc_map, SPM_LEVELS, NUM_LC_FEATURES)

    # Extract HOG
    base_hog_histograms = extract_hog_features(processed_image, layout_components)
    if base_hog_histograms is None:
        logging.error("Failed to extract HOG features for query.")
        return None
    query_hog_spm = build_spm(base_hog_histograms, SPM_LEVELS, NUM_HOG_FEATURES)

    if query_lc_spm is None or query_hog_spm is None:
        logging.error("Failed to build SPM vectors for query.")
        return None

    logging.info("Query features extracted successfully.")
    return query_lc_spm, query_hog_spm

def calculate_similarity(query_features, corpus_features):
    """Calculate similarity between query and corpus features."""
    query_lc_spm, query_hog_spm = query_features
    corpus_lc_spm = corpus_features['lc_spm']
    corpus_hog_spm = corpus_features['hog_spm']
    
    # Calculate similarities
    sim_lc = spm_kernel_direct_hik(query_lc_spm, corpus_lc_spm)
    sim_hog = spm_kernel_direct_hik(query_hog_spm, corpus_hog_spm)
    
    # Combine scores with weights and apply sigmoid-like scaling
    combined_score = (WEIGHT_LC * sim_lc) + (WEIGHT_HOG * sim_hog)
    
    # Increasing scaling factor from 2 to 3 for higher scores
    scaled_score = 2 / (1 + np.exp(-3 * combined_score)) - 1
    
    return scaled_score

def main(query_url):
    try:
        index_data, _ = load_index()
        if not index_data:
            print("Could not load index. Please run main_indexer.py first or check index directory.")
            return

        query_features = get_query_features(query_url)
        if query_features is None:
            print("Could not process query URL. Please check the URL or network connection.")
            return

        query_hash = get_url_hash(query_url)
        seen_urls = set()  # Keep track of seen URLs
        
        logging.info("Calculating similarities against index...")
        
        # Calculate similarities in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            url_to_hash = {}  # Map URLs to their hashes
            
            for url_hash, data in index_data.items():
                # Include the query URL itself
                if data['url'] not in seen_urls:
                    futures.append(
                        executor.submit(calculate_similarity, query_features, data)
                    )
                    seen_urls.add(data['url'])
                    url_to_hash[data['url']] = url_hash
            
            # Collect results with proper synchronization
            seen_urls_list = list(seen_urls)  # Convert set to list for indexing
            
            # Process all results first before filtering
            all_scores = []
            for i, future in enumerate(futures):
                try:
                    score = future.result()
                    if score is not None:
                        url = seen_urls_list[i]  # Get corresponding URL from the list
                        # Mark if this is the query URL itself
                        is_self = (url_to_hash[url] == query_hash)
                        
                        # Boost scores for self-matches to ensure they appear at the top
                        if is_self:
                            # Always set self-matches to 0.99 to ensure they're at the top
                            score = 0.99
                            
                        all_scores.append((score, url, is_self))
                except Exception as e:
                    logging.error(f"Error calculating similarity for {seen_urls_list[i]}: {e}")
            
            # Sort all results by score (descending) and ensure self is included
            all_scores.sort(reverse=True)
            
            # Take top N results, ensuring self-match is included if available
            has_self = any(item[2] for item in all_scores)
            results_to_show = []
            
            if has_self:
                # Add self-match first
                self_item = next((item for item in all_scores if item[2]), None)
                if self_item:
                    results_to_show.append(self_item)
                    
                # Add other top items up to TOP_N_RESULTS
                for item in all_scores:
                    if not item[2] and len(results_to_show) < TOP_N_RESULTS:
                        results_to_show.append(item)
            else:
                # No self-match, just take top N
                results_to_show = all_scores[:TOP_N_RESULTS]

        print(f"\n--- Top {len(results_to_show)} Results for: {query_url} ---")
        if not results_to_show:
            print("No similar pages found in the index.")
        else:
            for i, (score, url, is_self) in enumerate(results_to_show):
                self_marker = " (QUERY URL)" if is_self else ""
                print(f"{i+1}. Score: {score:.4f} - URL: {url}{self_marker}")
    
    except KeyboardInterrupt:
        print("\nQuery processing interrupted.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"An error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_query.py <query_url>")
    else:
        query_url_arg = sys.argv[1]
        main(query_url_arg)