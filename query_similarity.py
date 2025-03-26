#!/usr/bin/env python3
"""
Command-line script for querying similar web pages.

This script allows users to find visually similar web pages by querying
the index with a URL and retrieving the most similar pages.
"""

import sys
import logging
from websim.query import main as query_main

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the query CLI
    sys.exit(query_main()) 