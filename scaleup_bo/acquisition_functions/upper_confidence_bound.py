import numpy as np
from .base import BaseAcquisitionFunction

class UpperConfidenceBound(BaseAcquisitionFunction):
    def __init__(self, kappa=2.0):
        self.kappa = kappa

    def evaluation(self, X, Y_sample, model):
        """
        Computes upper confidence bound acquisition function.
        """
        mu, sigma = model.predict(X, return_std=True)
        ucb = mu + self.kappa * sigma

        return ucb
