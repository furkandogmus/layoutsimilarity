#!/usr/bin/env python3
"""
Command-line script for comparing two web pages for visual similarity.

This script allows users to directly compare two web pages based on their
visual features, without using the index.
"""

import sys
import logging
from websim.compare import main as compare_main

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the compare CLI
    sys.exit(compare_main()) 