#!/usr/bin/env python3
"""
Badminton Pose Analysis System
Main entry point for processing videos and analyzing badminton techniques.
"""

import sys
import os

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main_processor import process_all_videos

def main():
    """Main entry point for the badminton analysis system."""
    try:
        print("🏸 Starting Badminton Pose Analysis System...")
        print("=" * 50)
        process_all_videos()
        print("=" * 50)
        print("✅ Analysis complete! Check the Output directory for results.")
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        print("Please check that all required dependencies are installed:")
        print("pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
