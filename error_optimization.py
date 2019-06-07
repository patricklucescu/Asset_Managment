import os
import pandas as pd
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance as cv
import HMM as hmm

if __name__ == '__main__':

    # SET PARAMETERS
    risk_aversion = 1
    window = 150
    HMM_parameters = [[2,0.1,50],[3,0.1,50],[4,0.1,50],[5,0.1,50]]

    # set dates (and freq)
    dtindex = pd.bdate_range('2000-06-16', '2018-12-09', weekmask='Fri', freq='C')
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
        max_er = 0
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
                actual_ret = input_returns.loc[today+1, :] ## Modify
                actual_ret = actual_ret.values
                maxx = np.max(abs(pred_ret-actual_ret))
                max_er = np.max([maxx,max_er])
        error.append(max_er)




