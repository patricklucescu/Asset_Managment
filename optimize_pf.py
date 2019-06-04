import os
import pandas as pd
import osqp
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance as cv
import LSTM as ls


def read_batch_multi(X, batch_size, future=10, nr_taps=2, batch_start=0):
    num_features = 1
    s = X
    x = np.zeros((num_features, batch_size, s.shape[1]*nr_taps))
    y = np.zeros((num_features, batch_size, s.shape[1]))
    for i in range(nr_taps):
        x[0, :, i * s.shape[1]:(i + 1) * s.shape[1]] = s.iloc[i:-future - nr_taps + i + 1, :]
    y[0, :, :] = s.rolling(future).mean().iloc[future+nr_taps-1:, :]
    return x, y


def RNNLSTM(X):

    def generate_batch(N):
        assert(X.shape[0] > N)
        x, y = read_batch_multi(X, N, future=future, nr_taps=nr_taps)
        ys = 0.1*y + 0.9*y.mean(axis=2)[:, :, np.newaxis]       # shrinkage

        return x, ys

    future = 50
    nr_taps = 5
    x_s = X.shape[1]     # columns

    model = ls.Model(input_size=nr_taps*x_s, output_size=x_s)
    model.build()
    for epoch in range(10):
        epoch_error = model.train_batch(generate_batch=generate_batch, batch_size=X.shape[0]-future-nr_taps+1)
        print(epoch_error)
    last_x = np.zeros((1, 1, nr_taps * x_s))
    for i in range(nr_taps):
        last_x[0, 0, i*x_s:(i+1)*x_s] = X.iloc[-nr_taps+i, :]

    out = model.predict_batch(last_x)

    return out[0].ravel()


def optimize(x, ra):

    ret = x.mean().fillna(0).values

    cov = ra * pd.DataFrame(data=cv.oas(x)[0],index=x.columns, columns=x.columns).fillna(0)

    ret = RNNLSTM(x)

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
    risk_aversion = 1
    window = 150

    # set dates (and freq)
    dtindex = pd.bdate_range('2000-06-16', '2018-12-09', weekmask='Fri', freq='C')
    rebalancing_period = window
    rebalancing_dates = dtindex[window-1::rebalancing_period]
    df = pd.read_csv('asset_returns.csv', delimiter=',')
    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date']))
    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns = ['Date'])

    input_returns = df0.pct_change().fillna(0)
    input_returns = input_returns.iloc[1:, :]
    weights = pd.DataFrame(data=np.nan, columns=input_returns.columns, index=input_returns.index)
    for date in dtindex[window - 1:]:
        today = date
        returns = input_returns.loc[:today, :] #.tail(window)
        last = returns.index[-2]

        if today in rebalancing_dates:  # re-optimize and get new weights
            print(today)
            weights.loc[today, :] = optimize(returns, risk_aversion)
        else:  # no re-optimization, re-balance the weights
            weights.loc[today, :] = weights.loc[last, :] * (1 + returns.loc[today, :]) \
                                    / (1 + (weights.loc[last, :] * returns.loc[today, :]).sum())

    pnl = (weights.shift(1)*input_returns).sum(axis=1)

    plt.figure
    pnl.cumsum().plot()
    plt.title('Sharpe : {:.3f}'.format(pnl.mean()/pnl.std()*np.sqrt(52)))

    plt.figure
    df0.pct_change().cumsum().plot()

    plt.figure
    weights.plot()

    plt.show()
