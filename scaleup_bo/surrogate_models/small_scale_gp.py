import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize
from .base import BaseSurrogateModel

class SmallScaleGaussianProcess(BaseSurrogateModel):
    def __init__(self, kernel, alpha=0.1):
        """
        Initialize the LargeScaleGaussianProcess.

        Parameters:
        - kernel: The kernel function for the GP.
        - alpha: Regularization parameter to ensure numerical stability.
        """
        self.kernel = kernel
        self.alpha = alpha
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit the Gaussian Process model to the training data.

        Parameters:
        - X_train: Training inputs of shape (N, D).
        - y_train: Training targets of shape (N,).
        """

        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")
        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array.")
        if len(X_train) != len(y_train):
            raise ValueError("Number of samples in X_train and y_train must be equal.")

        self.X_train = X_train
        self.y_train = y_train
        self.K = self.kernel(X_train) + self.alpha * np.eye(len(X_train))
        try:
            self.L = cholesky(self.K, lower=True)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Cholesky decomposition failed: {e}")
        self.alpha_ = cho_solve((self.L, True), y_train)


    def predict(self, X, return_std=False):
        """
        Predict the objective function values for given inputs.

        Parameters:
        - X: New input points of shape (M, D).
        - return_std: If True, also return the standard deviation.

        Returns:
        - y_mean: Predicted means of shape (M,).
        - y_std (optional): Predicted standard deviations of shape (M,).
        """
        K_trans = self.kernel(X, self.X_train)
        y_mean = K_trans @ self.alpha_
        v = cho_solve((self.L, True), K_trans.T)
        y_cov = self.kernel(X) - K_trans @ v

        if return_std:
            y_std = np.diag(y_cov)
            return y_mean, y_std
        else:
            return y_mean

    def log_marginal_likelihood(self, params):
        """
        Compute the negative log marginal likelihood with regularization.

        Parameters:
        - params: Hyperparameters array. For simplicity, assume only length_scale.

        Returns:
        - Negative log marginal likelihood with regularization.
        """
        length_scale = params[0]
        self.kernel.length_scale = length_scale
        K = self.kernel(self.X_train) + (self.alpha + 1e-6) * np.eye(len(self.X_train))
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return np.inf  # Return a large value to penalize invalid hyperparameters
        alpha_ = cho_solve((L, True), self.y_train)
        log_likelihood = -0.5 * self.y_train.T @ alpha_
        log_likelihood -= np.sum(np.log(np.diag(L)))
        log_likelihood -= 0.5 * len(self.y_train) * np.log(2 * np.pi)

        regularization = 0.1 * (length_scale**2 + self.alpha**2)

        return -log_likelihood + regularization


    def optimize_hyperparameters(self):
        """
        Optimize the hyperparameters of the kernel by minimizing the negative log marginal likelihood.
        """
        # Initial guess and bounds for length scale 
        initial_guess = [self.kernel.length_scale]
        length_scale_bounds = self.kernel.length_scale_bounds

        # Combine bounds into a single list
        bounds = [length_scale_bounds]

        result = minimize(self.log_marginal_likelihood, initial_guess,
                          method="L-BFGS-B", bounds=bounds)
        self.kernel.length_scale = result.x[0]
        self.fit(self.X_train, self.y_train)

