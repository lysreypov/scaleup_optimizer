import numpy as np
from scipy.stats import norm
from .base import BaseAcquisitionFunction

class ProbabilityOfImprovement(BaseAcquisitionFunction):
    def __init__(self, xi=0.1):
        self.xi = xi

    def evaluation(self, X, Y_sample, model):
        """
        Computes probability of improvement acquisition function.
        """
        mu, sigma = model.predict(X, return_std=True)
        mu_sample_opt = np.min(Y_sample)

        with np.errstate(divide='warn', invalid='ignore'):
            Z = (mu_sample_opt - mu - self.xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0

        return pi