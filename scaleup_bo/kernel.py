import numpy as np

class RBF:
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 10)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    def __call__(self, X: np.ndarray, Y=None, eval_gradient=False):
        """
        Compute the Radial Basis Function (RBF) kernel matrix.

        Returns:
        - K: np.ndarray
            The RBF kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        # Calculate the pairwise squared Euclidean distances
        dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        # Compute the RBF kernel matrix
        K = np.exp(-0.5 * dists / self.length_scale ** 2)

        return K

