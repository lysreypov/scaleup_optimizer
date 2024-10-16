import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize
from .base import BaseSurrogateModel

class LargeScaleGaussianProcess(BaseSurrogateModel):
    def __init__(self, kernel, sigma_S=0.1, lambda_S=1.0, lambda_delta=1.0):
        """
        Initialize the LargeScaleGaussianProcess.

        Parameters:
        - kernel: The kernel function for the GP.
        - sigma_S: Noise variance for experimental system (default: 0.1).
        - sigma_L: Noise variance for production system (default: None, set to sigma_S if not provided).
        - lambda_S: Regularization for experimental system (default: 1.0).
        - lambda_delta: Additional regularization for production (default: 1.0).
        """
        self.kernel = kernel
        self.sigma_S = sigma_S
        self.sigma_L = None
        self.lambda_S = lambda_S
        self.lambda_delta = lambda_delta
        self.X_train_S = None
        self.y_train_S = None
        self.X_train_L = None
        self.y_train_L = None

    def fit(self, X_train_S, y_train_S, X_train_L, y_train_L):
        """
        Fit the GP model to both experimental and production training data.

        Parameters:
        - X_train_S: Training inputs for the experimental system.
        - y_train_S: Training targets for the experimental system.
        - X_train_L: Training inputs for the production system.
        - y_train_L: Training targets for the production system.

        """
        if not (X_train_S.ndim == 2 or X_train_L.ndim == 2):
            raise ValueError("X_train must be a 2D array.")
    
        if not (y_train_S.ndim == 1 or y_train_L.ndim == 1):
            raise ValueError("y_train must be a 1D array.")
        
        if len(X_train_S) != len(y_train_S):
            raise ValueError("Number of samples in X_train must be equal.")
        
        if len(X_train_L) != len(y_train_L):
            raise ValueError("Number of samples in X_train must be equal.")

        self.X_train_S = X_train_S
        self.y_train_S = y_train_S
        self.X_train_L = X_train_L
        self.y_train_L = y_train_L

        # Initialize sigma_L if not provided
        if self.sigma_L is None:
          self.sigma_L = self.sigma_S

        # Kernel matrices
        self.K_S = self.kernel(X_train_S, X_train_S)
        self.K_L = self.kernel(X_train_L, X_train_L)
        self.K_SL = self.kernel(X_train_S, X_train_L)

        # Combined matrix A
        upper_left = self.lambda_S ** 2 * self.K_S + self.sigma_S ** 2 * np.eye(len(X_train_S))
        lower_right = (self.lambda_S ** 2 + self.lambda_delta ** 2) * self.K_L + self.sigma_L ** 2 * np.eye(len(X_train_L))
        upper_right = self.K_SL
        lower_left = self.K_SL.T

        self.A = np.block([
            [upper_left, upper_right],
            [lower_left, lower_right]
        ])

        # Combined target values
        self.y_train_combined = np.concatenate([y_train_S, y_train_L])

        # Cholesky decomposition with error handling
        try:
            self.L = cholesky(self.A + 1e-10 * np.eye(len(self.A)), lower=True)
            self.alpha_ = cho_solve((self.L, True), self.y_train_combined)
        except LinAlgError:
            raise ValueError("Cholesky decomposition failed. Matrix A may not be positive definite.")

    def predict(self, X, return_std=False):
        """
        Predict the output for new input data.

        Parameters:
        - X: New input points of shape (M, D).
        - return_std: If True, return standard deviation.

        Returns:
        - y_mean: Predicted mean values.
        - y_std (optional): Predicted standard deviations.
        """
        # Covariance vectors
        k_S = self.kernel(self.X_train_S, X)
        k_L = self.kernel(self.X_train_L, X)

        b = np.vstack([k_S, k_L])

        y_mean = b.T @ self.alpha_
        v = cho_solve((self.L, True), b)
        y_cov = self.kernel(X, X) - b.T @ v

        if return_std:
            y_std = np.diag(y_cov)
            return y_mean.flatten(), y_std
        else:
            return y_mean

    def update_sigma_with_new_sample(self, X_new, y_new):
        """
        This method predicts the outputs for the new input samples, 
        calculates the residuals between the predicted and actual outputs, 
        and updates the noise variance (sigma_L) based on these residuals.
        """
        # Predict new samples
        y_pred, _ = self.predict(X_new, return_std=True)
        # Calculate residuals correctly
        residuals = y_new - y_pred
        # Recalculate sigma_L based on residuals
        self.sigma_L = np.sqrt(np.mean(residuals ** 2))

        return self.sigma_L

    def _update_kernel_matrices(self):
        """
        Update kernel matrices after changing training data.
        """
        self.K_S = self.kernel(self.X_train_S, self.X_train_S)
        self.K_L = self.kernel(self.X_train_L, self.X_train_L)
        self.K_SL = self.kernel(self.X_train_S, self.X_train_L)

        self.upper_left = self.lambda_S ** 2 * self.K_S + self.sigma_S ** 2 * np.eye(len(self.X_train_S))
        self.lower_right = (self.lambda_S ** 2 + self.lambda_delta ** 2) * self.K_L + self.sigma_L ** 2 * np.eye(len(self.X_train_L))

    def log_marginal_likelihood(self, params):
        """
        Compute the log marginal likelihood of the model.

        Parameters:
        - params: Hyperparameters for optimization.

        Returns:
        - Negative log marginal likelihood.
        """
        length_scale = params[0]
        self.kernel.length_scale = length_scale

        self._update_kernel_matrices()

        A = np.block([
            [self.upper_left, self.K_SL],
            [self.K_SL.T, self.lower_right]
        ])

        L = cholesky(A + 1e-10 * np.eye(len(A)), lower=True)
        alpha_ = cho_solve((L, True), self.y_train_combined)

        log_likelihood = -0.5 * self.y_train_combined.T @ alpha_
        log_likelihood -= np.sum(np.log(np.diagonal(L)))
        log_likelihood -= 0.5 * len(A) * np.log(2 * np.pi)

        regularization = 0.1 * length_scale**2

        return -log_likelihood + regularization

    def optimize_hyperparameters(self):
        """
        Optimize hyperparameters using negative log marginal likelihood.
        """
        # Initial guess and bounds for length scale
        initial_guess = [self.kernel.length_scale]
        length_scale_bounds = self.kernel.length_scale_bounds

        # Combine bounds into a single list
        bounds = [length_scale_bounds]

        result = minimize(self.log_marginal_likelihood, initial_guess,
                          method="L-BFGS-B", bounds=bounds)
        self.kernel.length_scale = result.x[0]
        self.fit(self.X_train_S, self.y_train_S, self.X_train_L, self.y_train_L)
