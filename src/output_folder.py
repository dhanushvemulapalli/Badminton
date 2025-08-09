"""
Simple utility to create output directories.
This module is now superseded by the config.py ProjectPaths.create_directories() method.
"""

import os
from config import ProjectPaths

def create_output_directories():
    """Create all necessary output directories using the config system."""
    ProjectPaths.create_directories()
    print(f"✅ Created output directories in: {ProjectPaths.OUTPUT_DIR}")

if __name__ == "__main__":
    create_output_directories()
