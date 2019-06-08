import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance as cv
import HMM as hmm

if __name__ == '__main__':

    # SET PARAMETERS
    risk_aversion = 1
    window = 150
    HMM_parameters = [[2,0.1,50]]

    # set dates (and freq)
    dtindex = pd.bdate_range('1993-01-01', '2002-12-31', weekmask='Fri', freq='C')
    rebalancing_period = window
    rebalancing_dates = dtindex[window - 1::rebalancing_period]
    df = pd.read_csv('asset_returns.csv', delimiter=',')
    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date']))
    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns=['Date'])
    input_returns = df0.pct_change().fillna(0)
    input_returns = input_returns.iloc[1:, :]

    # Set Error Vector
    error = []

    for parameters in HMM_parameters:
        er_df = pd.DataFrame(columns=df.columns)
        K = parameters[0]
        p = parameters[1]
        iter = parameters[2]

        for date in dtindex[window - 1:]:
            today = date
            returns = input_returns.loc[:today, :]  # .tail(window)
            last = returns.index[-1]

            if today in rebalancing_dates:
                print(today)
                posteriori_prob, mu_s, cov_s, pred_ret = hmm.expectation_maximization(returns, K, iter, p)
                actual_ret = input_returns.loc[today+1, :]  # Modify
                actual_ret = actual_ret.values
                abs_error = np.max(abs(pred_ret-actual_ret))
                er_df.append(abs_error)




