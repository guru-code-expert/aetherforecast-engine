"""Advanced feature engineering."""
import pandas as pd
import numpy as np
from config.settings import REFERENCE_DATE


def duration_to_hours_rounded(duration_str: str) -> int:
    """
    Convert duration strings like '5h 30m' → integer hours (round up if ≥30 min).
    """
    if pd.isna(duration_str):
        return 0
    
    parts = str(duration_str).replace('h ', 'h').replace('m', '').split('h')
    hours = 0
    minutes = 0
    
    if len(parts) == 1:
        if 'm' in duration_str:
            minutes = int(parts[0])
        else:
            hours = int(parts[0])
    else:
        hours = int(parts[0]) if parts[0] else 0
        minutes_part = parts[1].strip()
        minutes = int(minutes_part) if minutes_part else 0
    
    if minutes >= 30:
        hours += 1
    return hours


def departure_time_transform(time_str: str) -> int:
    """Custom anonymized transformation of departure time."""
    if pd.isna(time_str):
        return 0
    h, m = map(int, str(time_str).split(':'))
    total_minutes = h * 60 + m
    converted = (24 * 60 - total_minutes + 45) // 60  # simplified integer result
    return converted


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = df.copy()
    
    # Duration → rounded hours
    df['duration_hours'] = df['duration_str'].apply(duration_to_hours_rounded)
    
    # Departure time transformation
    df['dep_time_feature'] = df['departure_time'].apply(departure_time_transform)
    
    # Journey date → days since reference
    df['journey_date'] = pd.to_datetime(df['journey_date'], errors='coerce')
    reference = pd.to_datetime(REFERENCE_DATE)
    df['days_since_ref'] = (df['journey_date'] - reference).dt.days
    
    # Final encoded numeric versions
    numeric_cols = [
        'carrier', 'origin', 'destination', 'connections',
        'service_info', 'duration_hours', 'dep_time_feature', 'days_since_ref'
    ]
    
    # Ordinal encode any remaining non-numeric engineered features if needed
    for col in ['duration_hours', 'dep_time_feature', 'days_since_ref']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    return df[numeric_cols + ['target_value']]