"""
Script to make predictions using trained models
"""
import joblib
from pathlib import Path


def load_model(model_path):
    """
    Load a trained model from disk

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded model
    """
    return joblib.load(model_path)


def make_predictions(model, X):
    """
    Make predictions using the trained model

    Args:
        model: Trained model
        X: Features to predict on

    Returns:
        Predictions
    """
    return model.predict(X)


def main():
    """
    Main function to load model and make predictions
    """
    project_dir = Path(__file__).resolve().parents[2]

    # Load model
    # model_path = project_dir / 'models' / 'trained' / 'model.pkl'
    # model = load_model(model_path)

    # Load data for prediction
    # data_path = project_dir / 'data' / 'processed' / 'test_data.csv'
    # X = pd.read_csv(data_path)

    # Make predictions
    # predictions = make_predictions(model, X)

    # Save predictions
    # output_path = project_dir / 'models' / 'predictions' / 'predictions.csv'
    # pd.DataFrame({'predictions': predictions}).to_csv(output_path, index=False)

    print("Predictions made and saved successfully")


if __name__ == '__main__':
    main()
