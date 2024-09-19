import pandas as pd
import numpy as np
from typing import List


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing data.

    Args:
    df (pd.DataFrame): The input dataframe with raw data.

    Returns:
    pd.DataFrame: Dataframe with newly created features.
    """
    df = df.assign(
        liquidation_diff=df["long_liquidations"] - df["short_liquidations"],
        volume_diff=df["buy_volume"] - df["sell_volume"]
    )
    return df


def shift_feature(df: pd.DataFrame, conti_cols: List[str], intervals: List[int]) -> List[pd.Series]:
    """
    Generate shifted features for continuous variables.

    Args:
    df (pd.DataFrame): The input dataframe with continuous features.
    conti_cols (List[str]): List of continuous column names.
    intervals (List[int]): List of intervals for shifting.

    Returns:
    List[pd.Series]: List of series with shifted features.
    """
    return [
        df[conti_col].shift(interval).rename(f"{conti_col}_{interval}")
        for conti_col in conti_cols for interval in intervals
    ]
