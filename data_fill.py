import matplotlib.axes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime as dt
from typing import *
from pandas.core.groupby import DataFrameGroupBy

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from "owid-covid-data.csv" and "owid-covid-filled.csv files.
        :return:
            Data about covid
        :rtype:
            Tuple[pd.DataFrame, pd.DataFrame]
    """
    pd.set_option("display.max_columns", None)
    filepath = "owid-covid-data.csv"
    filled_filepath = "owid-covid-filled.csv"
    df = pd.read_csv(filepath, parse_dates=["date"])
    filled_df = pd.read_csv(filled_filepath, parse_dates=["date"], index_col=0)
    return df, filled_df

def save_data(df: pd.DataFrame) -> None:
    """
    Save data to "owid-covid-filled.csv" file.
        :param df:
            Dataframe to save
        :type df:
            pd.DataFrame
    """
    df.to_csv("owid-covid-filled.csv")

def fill_data(df: pd.DataFrame) -> None:
    """
    Fill missing data
        :param df:
            Dataframe to fill
        :type df:
            pd.DataFrame
    """
    df = df.groupby(["location", "date"], as_index=False).apply(merge_and_fill) # Merge rows with the same location and date
    df["total_cases"] = consecutive_day_same_loc(df, "total_cases") # Forward fill where location is the same and consecutive day
    df["total_cases"].fillna(0) # Fill the rest with 0
    df["total_deaths"] = consecutive_day_same_loc(df, "total_cases") # Forward fill where location is the same and consecutive day
    df["total_deaths"].fillna(0) # Fill the rest with 0
    df["continent"] = df.apply(assign_country, axis=1) # Assign continent based on ISO code
    df["new_cases_smoothed"] = df.index.to_series().apply(lambda idx: calculate_last_week_mean(df, idx, "new_cases")) # Mean of the last 7 days

def empty_percent(df: pd.DataFrame) -> pd.Series:
    """
    Retrieve percentage of missing data from each column.\n
    Show only higher than 0.
        :param df:
        :type df:
            pd.DataFrame
        :return:
            Column to percentage series
        :rtype:
            pd.Series
    """
    missing_vl_by_col = df.isnull().sum()
    num_rows, num_cols = df.shape
    missing_percent_by_col = (missing_vl_by_col / num_rows * 100).round(2)
    non_zero_columns = missing_percent_by_col[missing_percent_by_col != 0]
    return non_zero_columns

def assign_country(df: pd.DataFrame) -> str:
    """
    Assign continent based on ISO code starting with "OWID", if the "continent" is missing.
        :param df:
        :type df:
            pd.DataFrame
        :return:
            Continent
        :rtype:
            str
    """
    iso_to_continent = {
        'OWID_EUN': 'Europe',
        'OWID_ASI': 'Asia',
        'OWID_EUR': 'Europe',
        'OWID_AFR': 'Africa',
        'OWID_NAM': 'North America',
        'OWID_OCE': 'Oceania',
        'OWID_SAM': 'South America'
    }
    if pd.isna(df["continent"]):
        return iso_to_continent.get(df["iso_code"], df["continent"])
    else:
        return df["continent"]

def check_dates(ser: pd.Series) -> bool:
    """
    Check if days in Series column are in arithmetic sequence.\n
    Must be datetime type
        :param ser:
            Series to check
        :type ser:
            pd.Series
        :return:   
        :rtype:
            bool
    """
    ser.sort_values()
    values = len(ser.diff().value_counts())
    if values == 2: # Arithmetic sequence
        return True
    else: # Not arithmetic sequence
        return False
    
def calculate_last_week_mean(df: pd.DataFrame, index_ser: pd.Series, col_name: str) -> float:
    """
    Calculate mean from the last 7 days.
        :param df:
            Dataframe
        :type date_col:
            pd.DataFrame
        :param index_ser:
            Series of indexes. For example pd.DataFrame.index.to_series()
        :type index_ser:
            pd.Series
        :param col_name:
            Name of column to calculate mean from
        :type col_name:
            str
        :return:
            Mean of last 7 days values.
        :rtype:
            float
    """
    span: int = 7
    current_date = df.loc[index_ser, "date"]
    start_date = current_date - pd.Timedelta(days=span)
    condition = (df["date"] > start_date) & (df["date"] <= current_date) & (df.index >= index_ser - span)
    window = df.loc[condition, col_name]
    return round(window.sum() / 7, 3)

def consecutive_day_same_loc(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Forward fill column if the preceding row is the same location and consecutive day.
        :param df:
            Dataframe
        :type df:
            pd.Dataframe
        :param col_name:
            Name of the column to perform the operation on.
        :type col_name:
            str
        :return:
            Series of forward filled column.
        :rtype:
            pd.Series
    """
    df = df.sort_values(["location", "date"])
    condition = (df["location"] == df["location"].shift(1)) & (df["date"].diff() == pd.Timedelta(days=1))
    return df[col_name].where(~condition, df[col_name].ffill())

def merge_and_fill(group: DataFrameGroupBy) -> pd.DataFrame:
    """
    Merge rows based on groupby.
        :param group:
            Dataframe grouped by column or row.
        :type group:
            DataFrameGroupBy
        :return:
            Merged dataframe
        :rtype:
            pd.DataFrame
    """
    return group.ffill().bfill()[0]

# Load data
df, filled_df = load_data()

# print(empty_percent(filled_df))
# print(empty_percent(df))

# fill_data(df)
# save_data(df)