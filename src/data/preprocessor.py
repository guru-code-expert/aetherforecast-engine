"""Basic cleaning and initial transformations."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial cleaning:
    - Remove obvious duplicates
    - Handle missing values in connection count
    """
    df = df.copy()
    
    # Anonymized column names
    df.rename(columns={
        'Airline': 'carrier',
        'Source': 'origin',
        'Destination': 'destination',
        'Total_Stops': 'connections',
        'Duration': 'duration_str',
        'Additional_Info': 'service_info',
        'Dep_Time': 'departure_time',
        'Date_of_Journey': 'journey_date',
        'Price': 'target_value'
    }, inplace=True)

    # Fill missing connections with 0 (direct)
    df['connections'].fillna(0, inplace=True)
    
    # Drop completely unnecessary columns early if present
    cols_to_drop = ['Route', 'Arrival_Time']  # example columns that may exist
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical string columns using OrdinalEncoder."""
    df = df.copy()
    cat_columns = ['carrier', 'origin', 'destination', 'connections', 'service_info']
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    for col in cat_columns:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[[col]])
    
    return df