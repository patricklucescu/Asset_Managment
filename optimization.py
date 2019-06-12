import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import covariance as cv
from HMM import expectation_maximization
import osqp


def optimize(returns, risk_aversion, parameters):
    K, p, iterations = parameters[0], parameters[1], parameters[2]

    # Predict the returns
    posteriori_prob, mu_s, cov_s, predicted_return = expectation_maximization(returns, K, iterations, p)

    cov = risk_aversion * pd.DataFrame(data=cv.oas(returns)[0],index=returns.columns, columns=returns.columns).fillna(0)
    problem = osqp.OSQP()
    k = len(predicted_return)

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

    problem.setup(sCov, -predicted_return, sA, l, u)

    # Solve problem
    res = problem.solve()
    pr = pd.Series(data=res.x, index=returns.columns)
    return pr
