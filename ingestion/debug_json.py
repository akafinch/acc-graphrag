#!/usr/bin/env python
"""
Debug the structure of processed JSON files.
"""

import json
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def debug_json(file_path):
    """Print the structure of a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            logger.info(f"JSON is a dictionary with {len(data)} keys")
            for key in data.keys():
                value = data[key]
                if isinstance(value, list):
                    logger.info(f"  Key '{key}' contains a list with {len(value)} items")
                    if value and isinstance(value[0], dict):
                        logger.info(f"    First item keys: {list(value[0].keys())}")
                elif isinstance(value, dict):
                    logger.info(f"  Key '{key}' contains a dictionary with {len(value)} keys")
                    logger.info(f"    Keys: {list(value.keys())}")
                else:
                    logger.info(f"  Key '{key}' contains: {type(value)}")
        elif isinstance(data, list):
            logger.info(f"JSON is a list with {len(data)} items")
            if data and isinstance(data[0], dict):
                logger.info(f"  First item keys: {list(data[0].keys())}")
        else:
            logger.info(f"JSON is a {type(data)}")
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Please provide a file path")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    debug_json(file_path) 