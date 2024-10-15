import numpy as np
from skopt.space import Real, Integer, Categorical

def initialize_random_samples(n_initial_points, search_space):
    """
    Generate random samples from the specified search space.

    Returns:
    np.ndarray
        An array of shape (n_initial_points, n_dimensions) containing random samples
        from the defined search space, where each sample is drawn according to the type 
        of parameter (Real, Integer, or Categorical).

    The function handles three types of parameters:
    - Real: Uniform samples are drawn from a continuous range defined by low and high.
    - Integer: Random integer samples are drawn from a discrete range defined by low and high.
    - Categorical: Random samples are drawn from a predefined set of categories.
    """
    X_sample = []
    for dim in search_space:
        if isinstance(dim, Real):
            samples = np.random.uniform(dim.low, dim.high, n_initial_points)
        elif isinstance(dim, Integer):
            samples = np.random.randint(dim.low, dim.high + 1, n_initial_points)
        elif isinstance(dim, Categorical):
            samples = np.random.choice(dim.categories, n_initial_points)
        else:
            raise ValueError("Unsupported parameter type in search space")
        X_sample.append(samples)

    X_sample = np.array(X_sample, dtype=object).T  
    return X_sample

def ensure_scalar(y):
    """Ensure the objective function returns a scalar value."""
    if np.isscalar(y):
        return y
    elif np.size(y) == 1:
        return np.asscalar(y)
    else:
        raise ValueError(f"The user-provided objective function must return a scalar value. Received: {y}")


