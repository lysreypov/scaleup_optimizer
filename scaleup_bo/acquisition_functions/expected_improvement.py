import numpy as np
from scipy.stats import norm
from .base import BaseAcquisitionFunction

class ExpectedImprovement(BaseAcquisitionFunction):
    def __init__(self, xi=0.1):
        self.xi = xi

    def evaluation(self, X, Y_sample, model):
        """
        Computes the Expected Improvement acquisition function.
        """
        mu, sigma = model.predict(X, return_std=True)
        mu_sample_opt = np.min(Y_sample)

        with np.errstate(divide='warn', invalid='ignore'):
            imp = mu_sample_opt - mu - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei
