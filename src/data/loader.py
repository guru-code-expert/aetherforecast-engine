"""Data ingestion module."""
import pandas as pd
from typing import Tuple
from config.settings import DATA_PATH


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw dataset.

    Args:
        path: Path to CSV file.

    Returns:
        Raw DataFrame.
    """
    df = pd.read_csv(path)
    return df.reset_index(drop=True)