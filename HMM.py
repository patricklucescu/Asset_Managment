import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def init_A(p, K):
    # p: switching probability
    return np.ones((K, K)) * p + np.eye(K) * (1 - K*p)

def get_artificial_data(K, T, lmu, lcv):
    P = 50
    data = pd.DataFrame(0, index=range(0, T), columns=range(0, 2))
    for i in range(0, T):
        s = [j for j in range(0, K) if P*j/K <= np.mod(i,P) < P*(j+1)/K][0]
        data.iloc[i, :] = lmu[s] + np.matmul(np.linalg.cholesky(lcv[s]), np.random.randn(2)).T

    return data

def by_mean(X):
    Factor = .01
    sigma = np.max(X.std())
    sigma_0 = Factor * sigma
    m = X.shape[0]
    mu_0 = np.mean(X.mean() / X.std())
    return ((np.sum(X, axis=0) / sigma ** 2 + mu_0 / sigma_0 ** 2) / (
            m / sigma ** 2 + 1 / sigma_0 ** 2)).values

def expectation_maximization(data, K, iter = 50, p = .3):

    # efficient implementation using alpha/beta algorithm (Bishop, chap. 13.2.2)
    epsilon = .00001
    L = data.shape[0]
    dim = data.shape[1]

    # inits
    p0 = (np.ones((K, 1))/K).ravel()
    ah = np.ones((K, L))/K              # fwd message
    bh = np.ones((K, L))/K              # bkw message

    A = init_A(p, K)         # state transition matrix
    mu = 0.1*np.random.rand(dim, K)         # mu
    sa = np.zeros((dim, dim, K))        # covariance matrix
    for k in range(0, K):
        sa[:, :, k] = np.eye(dim)

    for j in range(0, iter):
        print(j)

        # E STEP ##################################################################################################
        em = np.zeros((K, L))
        for k in range(0, K):
            for i in range(0, L):
                # emission probabilities
                em[k, i] = multivariate_normal.pdf(data.iloc[i, :].values, mean=mu[:, k], cov=sa[:, :, k],allow_singular=True)

        # forward message
        ah[:, 0] = p0 * em[:, 0] / np.matmul(p0.T, em[:, 0])
        c = np.ones((L, 1))
        for i in range(1, L):
            aux = em[:, i] * np.matmul(A.T, ah[:, i-1])
            c[i] = np.sum(aux)
            if c[i] == 0:
                print('scaling factor 0')
            ah[:, i] = aux / c[i]
        pX = np.prod(c)
        # backward
        bh[:, L-1] = np.ones(K)
        for i in range(L-2, -1, -1):
            if c[i+1] == 0:
                print('scaling factor 0')
            bh[:, i] = np.matmul(A, em[:, i+1] * bh[:, i+1] / c[i+1])

        g = np.zeros((K, L))
        h = np.zeros((K, K, L))
        for i in range(0, L):
            g[:, i] = ah[:, i] * bh[:, i]
            if i > 0:
                for q in range(0, K):
                    for p in range(0, K):
                        h[q, p, i] = ah[q, i-1]*em[p, i]*A[q, p]*bh[p, i]/c[i]

        # M STEP ##################################################################################################
        p0 = g[:, 0] / np.sum(g[:, 0])
        se = sa
        Factor = .01
        maxInd = np.argmax(g, axis=0)
        for k in range(0, K):
            #mu[:, k] = np.matmul(data.values.T, g[k, :].T) / np.max((np.sum(g[k, :]), epsilon))
            #mu[:, k] = 0.1*aux1 + 0.9*np.ones((dim,))*0.01
            dataLocal = data.loc[maxInd == k, :]
            if dataLocal.shape[0] > 1:
                mu[:, k] = by_mean(dataLocal)
            else:
                mu[:, k] = 0

            for i in range(0, L):
                se[:, :, k] = se[:, :, k] + g[k, i] * np.matmul(np.atleast_2d((data.iloc[i, :].values - mu[:, k])).T, np.atleast_2d((data.iloc[i, :].values - mu[:, k])))
            sa[:, :, k] = se[:, :, k] / np.max((np.sum(g[k, :]), epsilon))

        for q in range(0, K):
            for p in range(0, K):
                A[q, p] = h[q, p, :].sum() / np.max((h[q, :, :].sum(), epsilon))
        pred = np.matmul(mu, np.matmul(A.T,g[:, L-1]))
    return g, mu, sa, pred


def multivariate_gaussian(pos, mu, Sigma):

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

if __name__ == '__main__':

    file = 'input_data.xlsx'
    data = pd.read_excel(file, 'input')
    dates = data['Date']
    ids = [1, 2, 6]
    data = data.iloc[:, ids]
    names = data.columns

    K = 4             # number of clusters
    iter = 50         # number of iterations (of EM algo)

    posteriori_prob, mu_s, cov_s, pred = expectation_maximization(data, K, iter=iter)
    # regimes
    plt.plot(posteriori_prob.T)

    # annotated underlyings
    maxInd = np.argmax(posteriori_prob, axis=0)
    sp = data.iloc[:, 0].cumsum()
    ty = data.iloc[:, 1].cumsum()
    vx = data.iloc[:, 2].cumsum()
    plt.figure()
    for i in range(0, K):
        nsp = np.nan * sp
        nty = np.nan * ty
        nvx = np.nan * vx
        nsp[maxInd == i] = sp[maxInd == i]
        nty[maxInd == i] = ty[maxInd == i]
        nvx[maxInd == i] = vx[maxInd == i]
        plt.subplot(311), plt.plot(nsp), plt.title('SPX')
        plt.subplot(312), plt.plot(nty), plt.title('T10')
        plt.subplot(313), plt.plot(nvx), plt.title('VIX')

    g = pd.DataFrame(posteriori_prob.T, index=dates, columns=range(0, K))
    g.to_csv('regimes.csv')
    plt.show()
