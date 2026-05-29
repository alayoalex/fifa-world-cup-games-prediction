"""
Script to download or generate data and save to data/raw/
"""
import os
from pathlib import Path


def main():
    """
    Main function to acquire and save raw data
    """
    # Define paths
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_path = project_dir / 'data' / 'raw'

    # Your data acquisition logic here
    print(f"Raw data will be saved to: {raw_data_path}")

    # Example:
    # df = fetch_data_from_source()
    # df.to_csv(raw_data_path / 'dataset.csv', index=False)


if __name__ == '__main__':
    main()
