import numpy as np
from numpy.linalg import cholesky
from scipy.optimize import minimize

"""
Gaussian Process Implementation Class
"""
class gp_implementation:
    def __init__(self, kernel, params, dims):
        self.kernel = kernel.lower()
        self.params = params
        self.dims = dims
        self.mean = None
        self.cov = None

    def train(self, train, test, y):
        # evaluate kernel
        if self.dims == 2:
            Kxx = self.evaluate_kernel([train[0], train[0], train[1], train[1]])
            Ksx = self.evaluate_kernel([test[0], train[0], test[1], train[1]])
            Kss = self.evaluate_kernel([test[0], test[0], test[1], test[1]])

        else:
            Kxx = self.evaluate_kernel([train, train])
            Ksx = self.evaluate_kernel([test, train])
            Kss = self.evaluate_kernel([test, test])

        mean = Ksx @ np.linalg.inv(Kxx) @ y
        cov = Kss - Ksx @ np.linalg.inv(Kxx) @ Ksx.T
        return mean, cov


    def evaluate_kernel(self, x):
        if self.kernel == "rbf":
            if self.dims == 2:
                sigma = self.params[0]
                l1 = self.params[1]
                l2 = self.params[2]
                x1 = x[0]
                x2 = x[1]
                x3 = x[2]
                x4 = x[3]
                return (sigma ** 2 *
                    self.rbf(x1, x2, l1) *
                    self.rbf(x3, x4, l2))

            else:
                sigma = self.params[0]
                l = self.params[1]
                x1 = x[0]
                x2 = x[1]
                return sigma ** 2 * self.rbf(x1, x2, l)

        return 0

    def rbf(self, x1, x2, l):
        # Probably faster way to calculate this
        # Currently use ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T * y
        x1_norm = np.sum((x1) ** 2, axis=-1)
        x2_norm = np.sum((x2) ** 2, axis=-1)
        return np.exp(-(x1_norm[:, None] + x2_norm - 2 * np.dot(x1, x2.T)) / (2 *(l ** 2)))

    def create_grid(self,start, end, num):
        i = np.linspace(start, end, num)
        x, y = np.meshgrid(i, i)
        return x.reshape(-1)[:, None], y.reshape(-1)[:, None]

    ## Work in progress
    # Need to figure out a way to optimize parameters without having
    # variables explode
    def likelihood_fx(self, train, y):

        def fx(params):
            eval_param_class = gp_implementation(self.kernel, params, self.dims)
            Kxx = eval_param_class.evaluate_kernel(train)
            # See equation 5.8 from Rasmussen and Williams
            return ((np.sum(np.log(np.diagonal(Kxx))) + 0.5 * y.T.dot(np.linalg.inv(cholesky(Kxx)).dot(y))) +
                        0.5 * len(train[0]) * np.log(2*np.pi))

        return fx

    def min_params(self, train, y, start):
        optimize_params = minimize(self.likelihood_fx(train, y), start, method = "L-BFGS-B")
        return optimize_params