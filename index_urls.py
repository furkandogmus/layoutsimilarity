#!/usr/bin/env python3
"""
Command-line script for indexing web pages for visual similarity.

This script allows users to index web pages by extracting and storing their
visual features for later similarity comparison.
"""

import sys
import logging
from websim.indexer import main as indexer_main

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the indexer CLI
    sys.exit(indexer_main()) 