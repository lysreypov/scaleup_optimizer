import numpy as np
from skopt.space import Real, Integer, Categorical

class Scale:
    def __init__(self, search_space):
        self.search_space = search_space

        # Identify index of parameter required log transformation
        self.log_indices = self._identify_log_indices()

    def _identify_log_indices(self):
        log_indices = []
        for i, dim in enumerate(self.search_space):
            if isinstance(dim, (Real, Integer)):
                if dim.high / dim.low > 100:
                    log_indices.append(i)
        return log_indices

    def normalize(self, X):
        """
        Normalize parameters to [0,1]; and apply log-transform to the large multitude value
        """
        X_norm = []
        for i, dim in enumerate(self.search_space):
            if isinstance(dim, (Real, Integer)):
              if i == self.log_indices[0]:
                X_norm.append((np.log(X[:, i].astype(float)) - np.log(dim.low)) / (np.log(dim.high) - np.log(dim.low)))
              else:
                X_norm.append((X[:, i].astype(float) - dim.low) / (dim.high - dim.low))
            elif isinstance(dim, Categorical):
                X_norm.append(np.array([dim.categories.index(x) for x in X[:, i]]) / (len(dim.categories) - 1))
        return np.array(X_norm).T

    def denormalize(self, X_norm):
        """
        Denormalize parameters back to the original scale and apply exp to the large multitude value
        """
        X = []
        for i, dim in enumerate(self.search_space):
            if isinstance(dim, Real):
              if i in self.log_indices:
                  X.append(np.exp(X_norm[:, i] * (np.log(dim.high) - np.log(dim.low)) + np.log(dim.low)))
              else:
                  X.append(X_norm[:, i] * (dim.high - dim.low) + dim.low)
            elif isinstance(dim, Integer):
                X.append(np.round(X_norm[:, i] * (dim.high - dim.low) + dim.low).astype(int))
            elif isinstance(dim, Categorical):
                X.append([dim.categories[(int(np.round(x * (len(dim.categories) - 1))))] for x in X_norm[:, i]])

        return np.array(X, dtype=object).T
