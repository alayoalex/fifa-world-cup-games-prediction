"""
Script to engineer features from processed data
"""
import pandas as pd
from pathlib import Path


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from the input dataframe

    Args:
        df: Input dataframe

    Returns:
        DataFrame with new features
    """
    # Your feature engineering logic here
    # Example:
    # df['new_feature'] = df['col1'] * df['col2']

    return df


def main():
    """
    Main function to build features
    """
    project_dir = Path(__file__).resolve().parents[2]

    # Load processed data
    processed_data_path = project_dir / 'data' / 'processed'

    # Create features
    # df = pd.read_csv(processed_data_path / 'dataset.csv')
    # df_with_features = create_features(df)

    # Save
    # df_with_features.to_csv(processed_data_path / 'dataset_features.csv', index=False)

    print("Features created successfully")


if __name__ == '__main__':
    main()
