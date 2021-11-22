# import all packages and set plots to be embedded inline
# Used https://towardsdatascience.com/implement-a-gaussian-process-from-scratch-2a074a470bce
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from pyDOE import lhs


class GaussianProcess:
    def __init__(self, n_restarts, optimizer):
        self.n_restarts = n_restarts
        self.optimizer = optimizer

    def Corr(self, X1, X2, theta):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            K[i, :] = np.exp(-np.sum((theta*(X1[i, :]-X2)**2), axis=1))
        return K


    def Neglikelihood(self, theta):
        theta = 10**theta
        n = self.X.shape[0]
        one = np.ones((n,1))

        K = self.Corr(self.X, self.X, theta) + np.eye(n)*1e-10
        inv_K = np.linalg.inv(K)

        mu = (one.T @ inv_K @ self.y) / (one.T @ inv_K @ one)

        SigmaSqr = (self.y - mu * one).T @ inv_K @ (self.y - mu * one) / n


        DetK = np.linalg.det(K)
        LnLike = -(n / 2) * np.log(SigmaSqr) - 0.5 * np.log(DetK)

        self.K, self.inv_K, self.mu, self.SigmaSqr = K, inv_K, mu, SigmaSqr

        return -LnLike.flatten()



    def fit(self, X, y):
        self.X, self.y = X, y
        lb, ub = -3, 2

        # Generate random starting points (Latin Hypercube)
        lhd = lhs(self.X.shape[1], samples=self.n_restarts)

        # Scale random samples to the given bounds
        initial_points = (ub - lb) * lhd + lb

        # Create A Bounds instance for optimization
        bnds = Bounds(lb * np.ones(X.shape[1]), ub * np.ones(X.shape[1]))

        # Run local optimizer on all points
        opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
        opt_func = np.zeros((self.n_restarts, 1))
        for i in range(self.n_restarts):
            res = minimize(self.Neglikelihood, initial_points[i, :], method=self.optimizer,
                           bounds=bnds)
            opt_para[i, :] = res.x
            opt_func[i, :] = res.fun

        # Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]

        # Update attributes
        self.NegLnlike = self.Neglikelihood(self.theta)



    def predict(self, X_test):
        n = self.X.shape[0]
        one = np.ones((n, 1))

        # Construct correlation matrix between test and train data
        k = self.Corr(self.X, X_test, 10 ** self.theta)

        # Mean prediction
        f = self.mu + k.T @ self.inv_K @ (self.y - self.mu * one)

        # Variance prediction
        SSqr = self.SigmaSqr * (1 - np.diag(k.T @ self.inv_K @ k))

        return f.flatten(), SSqr.flatten()

    def score(self, X_test, y_test):
        y_pred, SSqr = self.predict(X_test)
        RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))

        return RMSE

def Test_1D(X):
    """1D Test Function"""

    y = (X * 6 - 2) ** 2 * np.sin(X * 12 - 4)

    return y

    # Training data
X_train = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1], ndmin=2).T
y_train = Test_1D(X_train)

# Testing data
X_test = np.linspace(0.0, 1, 100).reshape(-1, 1)
y_test = Test_1D(X_test)

# GP model training
GP = GaussianProcess(n_restarts=10, optimizer='L-BFGS-B')
GP.fit(X_train, y_train)

# GP model predicting
y_pred, y_pred_SSqr = GP.predict(X_test)


def plotfunc():
    x = np.linspace(0,1,200)
    print(range(x.shape[0]))
    y = ((x * 6 - 2) ** 2 * np.sin(x * 12 - 4))
    ci_low = y_pred - 1.96*y_pred_SSqr
    ci_hi = y_pred + 1.96*y_pred_SSqr
    fig, ax = plt.subplots()
    ax.plot(x, y, 'r')
    print(X_test.shape[0], ci_low.shape[0], ci_hi.shape[0])
    #ax.fill_between(np.array(X_test), np.array(ci_low), np.array(ci_hi), alpha=0.2)
    ax.plot(X_test, ci_low)
    ax.plot(X_test, ci_hi)
    ax.plot(X_train, y_train, 'o')
    plt.show()

plotfunc()
