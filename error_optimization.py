import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance as cv
import HMM as hmm
#from hmmlearn import hmm
from sklearn.utils import check_random_state

if __name__ == '__main__':

    # SET PARAMETERS
    risk_aversion = 1
    window = 52
    HMM_parameters = [[2, 0.1, 50],[3, 0.1, 50],[4, 0.1, 50],[5, 0.1, 50],[6, 0.1, 50],[7, 0.1, 50],[8, 0.1, 50]]

    # set dates (and freq)
    dtindex = pd.bdate_range('1993-01-01', '2002-12-31', weekmask='Fri', freq='C')
    rebalancing_period = window
    rebalancing_dates = dtindex[window - 1::rebalancing_period]
    df = pd.read_csv('markets_new.csv', delimiter=',')
    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date']))
    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns=['Date'])
    input_returns = df0.pct_change().fillna(0)
    input_returns = input_returns.iloc[1:, :]
    num_assets = len(input_returns.columns)
    # Set Error Vector
    error = []
    sign = []

    for parameters in HMM_parameters:
        good_sign = 0
        er_value = 0
        counter = 0
        K = parameters[0]
        p = parameters[1]
        iter = parameters[2]
        for date in dtindex[window - 1:]:
            today = date
            returns = input_returns.loc[:today, :]  # .tail(window)
            last = returns.index[-1]

            if today in rebalancing_dates:
                print(today)
                print(K)
                # check available assets
                available_assets = []
                today_prices = list(df0.loc[today].values)
                for i in range(0, num_assets):
                    if today_prices[i] > 0:
                        available_assets.append(str(i + 1))
                returns = input_returns.loc[:today, available_assets]

                # model = hmm.GaussianHMM(n_components=K, n_iter=iter)
                # model.transmat_ = trans_matrix
                # model.fit(returns)
                # states = model.predict(returns)
                # transmat_cdf = np.cumsum(model.transmat_, axis=1)
                # random_state = check_random_state(model.random_state)
                # next_state = (transmat_cdf[states[-1]] > random_state.rand()).argmax()
                # pred_ret = model._generate_sample_from_state(next_state)

                posteriori_prob, mu_s, cov_s, pred_ret = hmm.expectation_maximization(returns, K, iter, p)
                actual_ret = input_returns.loc[today+1, available_assets].values
                abs_error = abs(pred_ret-actual_ret)
                counter += len(abs_error)
                er_value += sum(abs_error)
                for i in range(0, len(abs_error)):
                    if actual_ret[i]*pred_ret[i] >= 0:
                        good_sign += 1
        error.append(er_value/counter)
        sign.append(good_sign/counter)



plt.plot([2,3,4,5,6,7,8],error,'bo')