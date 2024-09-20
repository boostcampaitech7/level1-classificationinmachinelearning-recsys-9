import os
import pandas as pd
from typing import List, Dict


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load raw training and test data.

    Args:
    data_path (str): The directory path where raw data files are stored.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
    """
    train_df = pd.read_csv(os.path.join(data_path, "train.csv")).assign(_type="train")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).assign(_type="test")
    df = pd.concat([train_df, test_df], axis=0)
    return df


def load_external_files(data_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load external hourly data files.

    Args:
    data_path (str): The directory path where external hourly data files are stored.

    Returns:
    Dict[str, pd.DataFrame]: Dictionary mapping file names to dataframes.
    """
    file_names = [f for f in os.listdir(data_path) if f.startswith("HOURLY_") and f.endswith(".csv")]
    file_dict = {
        f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names
    }
    return file_dict


def merge_data(df: pd.DataFrame, file_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge external hourly data with the main dataframe.

    Args:
    df (pd.DataFrame): The main dataframe containing train and test data.
    file_dict (Dict[str, pd.DataFrame]): Dictionary of external hourly dataframes.

    Returns:
    pd.DataFrame: Dataframe with merged external data.
    """
    for _file_name, _df in file_dict.items():
        _rename_rule = {col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID" for col in _df.columns}
        _df = _df.rename(_rename_rule, axis=1)
        df = df.merge(_df, on="ID", how="left")
    return df
