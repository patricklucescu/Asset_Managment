import os
import pandas as pd
import numpy as np


def database_processing(df, dtindex,asset_removal):
    """ Reindex dataset by dtindex and clean the dataset of unwanted assets(if asset_removal is true)"""
    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))
    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns=['Date'])

    if asset_removal:
        for asset in df0.columns:
            if pd.isna(df0.loc[dtindex[0]][asset]):
                df0 = df0.drop(columns=[asset])
        return df0
    else:
        return df0


def returns_cleaning(input_returns, df, rolling_window, today):
    """ If rolling window is available, remove assets that do not have enough observations
     If not, just remove assets that do not have at least one return available"""
    available_assets = []
    if rolling_window:
        for asset in input_returns.columns[1:]:
            if all(df.loc[today - rolling_window:today][asset] > 0):
                available_assets.append(asset)
        returns = input_returns.loc[today - rolling_window:today, available_assets]

    else:
        for asset in input_returns.columns[1:]:
            if df.loc[today][asset] > 0 and df.loc[today - 1][asset] > 0:
                available_assets.append(asset)
        returns = input_returns.loc[:today, available_assets]
    return returns, available_assets








