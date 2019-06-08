import os
import pandas as pd
import osqp
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance as cv
import HMM as hmm


def HMM(X,K,p,iter):
    posteriori_prob, mu_s, cov_s, pred = hmm.expectation_maximization(X, K, iter, p)

    return pred

def optimize(x, ra, method=None):
    ## SET HMM PARAMETERS
    K = 4
    p = 0.1
    iter = 50

    ret = x.mean().fillna(0).values

    cov = ra * pd.DataFrame(data=cv.oas(x)[0],index=x.columns, columns=x.columns).fillna(0)

    ret = HMM(x,K,p,iter)

    problem = osqp.OSQP()
    k = len(ret)

    # Setup workspace
    """
    setup(self, P=None, q=None, A=None, l=None, u=None, **settings):
            Setup OSQP solver problem of the form
            minimize     1/2 x' * P * x + q' * x
            subject to   l <= A * x <= u
    """
    A = np.concatenate((pd.np.ones((1, k)), np.eye(k)), axis=0)
    sA = sparse.csr_matrix(A)
    l = np.hstack([1, np.zeros(k)])
    u = np.ones(k + 1)
    sCov = sparse.csr_matrix(cov)

    problem.setup(sCov, -ret, sA, l, u)

    # Solve problem
    res = problem.solve()
    pr = pd.Series(data=res.x, index=x.columns)
    return pr


if __name__ == '__main__':

    # load data ###################################################################
    risk_aversion = 1 # I assume is for the Mean Variance Optimization
    window = 150 # after how many weeks do you want to rebalance

    # set dates (and freq)
    dtindex = pd.bdate_range('1993-01-01', '2018-12-09', weekmask='Fri', freq='C')
    rebalancing_period = window
    rebalancing_dates = dtindex[window-1::rebalancing_period]
    df = pd.read_csv('markets_new.csv', delimiter=',')
    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date']))
    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns=['Date'])

    input_returns = df0.pct_change().fillna(0)
    input_returns = input_returns.iloc[1:, :]
    weights = pd.DataFrame(data=np.nan, columns=input_returns.columns, index=input_returns.index)
    num_assets = len(input_returns.columns)
    assets = input_returns.columns
    for date in dtindex[window - 1:]:
        today = date
        returns = input_returns.loc[:today, :] #.tail(window)
        last = returns.index[-1]

        if today in rebalancing_dates:  # re-optimize and get new weights
            print(today)
            available_assets =[]
            today_prices = list(df0.loc[today].values)
            for i in range(0,num_assets):
                if today_prices[i] > 0:
                    available_assets.append(str(i+1))
            returns = input_returns.loc[:today, available_assets]
            weights.loc[today, :] = [0] * 14
            weights.loc[today, available_assets] = optimize(returns, risk_aversion)
        else:  # no re-optimization, re-balance the weights
            weights.loc[today, :] = weights.loc[last, :] * (1 + returns.loc[today, :]) \
                                    / (1 + (weights.loc[last, :] * returns.loc[today, :]).sum())

    pnl = (weights.shift(1)*input_returns).sum(axis=1)
    one = np.ones(964)
    plt.figure
    pnl.cumsum().plot()
    plt.title('Sharpe : {:.3f}'.format(pnl.mean()/pnl.std()*np.sqrt(52)))

    plt.figure
    df0.pct_change().cumsum().plot()

    plt.figure
    weights.plot()

    plt.show()
