"""End-to-end pipeline."""
import pandas as pd
from src.data.loader import load_dataset
from src.data.preprocessor import basic_cleaning, encode_categorical_features
from src.features.engineering import create_features
from src.models.train import train_best_model, save_model
from src.models.evaluate import full_train_evaluate


def run_full_pipeline(data_path: str = None) -> pd.DataFrame:
    """
    Execute the complete pipeline from raw CSV to trained model.
    
    Returns:
        Final processed DataFrame ready for modeling.
    """
    print("1. Loading data...")
    df = load_dataset(data_path)
    
    print("2. Basic cleaning...")
    df = basic_cleaning(df)
    
    print("3. Encoding categorical columns...")
    df = encode_categorical_features(df)
    
    print("4. Feature engineering...")
    df_processed = create_features(df)
    
    print("5. Removing duplicates after feature engineering...")
    df_processed.drop_duplicates(inplace=True)
    
    X = df_processed.drop('target_value', axis=1)
    y = df_processed['target_value']
    
    print("6. Training final model...")
    model, _, _, _ = full_train_evaluate(X, y)
    
    print("7. Saving model...")
    save_model(model)
    
    return df_processed