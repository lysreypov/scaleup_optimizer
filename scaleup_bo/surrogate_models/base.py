from abc import ABC, abstractmethod
import numpy as np


class BaseSurrogateModel(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Fit the surrogate model to the data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict the objective function values for given inputs."""
        pass

    @abstractmethod
    def log_marginal_likelihood(self, params):
        """Compute the negative log marginal likelihood with regularization."""
        pass

    @abstractmethod
    def optimize_hyperparameters(self, X_new, y_new):
        """Update the surrogate model with new data."""
        pass
