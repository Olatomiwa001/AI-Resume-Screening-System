#!/usr/bin/env python3
"""
Script to generate demo data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_synthetic_data import generate_dataset

if __name__ == "__main__":
    print("=" * 60)
    print("AI Resume Screener - Demo Data Generator")
    print("=" * 60)
    
    generate_dataset(
        output_dir="./data/demo",
        num_resumes=20,
        num_jobs=4
    )
    
    print("\n" + "=" * 60)
    print("Done! You can now:")
    print("1. Run the training notebook: jupyter notebook notebooks/training_demo.ipynb")
    print("2. Launch the app: streamlit run streamlit_app.py")
    print("=" * 60)