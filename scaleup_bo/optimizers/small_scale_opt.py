import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from .base import BaseOptimizer
from ..acquisition_functions import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from ..surrogate_models import SmallScaleGaussianProcess
from ..kernel import RBF
from ..scale import Scale
from ..utils import initialize_random_samples, ensure_scalar

class SmallScaleBayesianOptimizer(BaseOptimizer):
    def __init__(self, objective_func, search_space, acq_func='EI', gp=None, n_calls=10, n_initial_points=1):
        self.objective_func = objective_func
        self.search_space = search_space
        self.acq_func_type = acq_func
        self.acq_func = None
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.X_sample = None
        self.Y_sample = None
        self.best_params = None
        self.best_score = float('inf')
        self.gp = None
        self.scale = Scale(self.search_space)

        # Initialize gp model
        if self.gp is None:
            self.gp = SmallScaleGaussianProcess(kernel=RBF(1.0, (1e-5, 100)), alpha=0.1)
        else:
            self.gp = gp

        # Run the optimization process
        self.optimize()

    def propose_next_point(self):
        """
        Proposes the next point to evaluate in the parameter space.
        
        This function utilizes the acquisition function 
        to explore the search space and find the point that minimizes 
        the objective function by sampling points in the normalized 
        space and then optimizing the negative of acquisition function .
        
        Returns:
            np.ndarray: The proposed point in normalized space.
        """
        min_val = float('inf')
        min_x = None

        def min_obj(X):
            """
            Objective function for minimization.
            Computes the negative acquisition function to maximize improvement.
            """
            if self.acq_func_type == 'EI':
                self.acq_func = ExpectedImprovement()
            elif self.acq_func_type == 'PI':
                self.acq_func = ProbabilityOfImprovement()
            elif self.acq_func_type == 'UCB':
                self.acq_func = UpperConfidenceBound()
            else:
                raise ValueError("Invalid acquisition function: {}".format(self.acq_func))

            acquisition_value = self.acq_func.evaluation(X.reshape(1, -1), self.X_sample, self.Y_sample, self.gp)

            return -acquisition_value 

        # Use normalized bounds
        bounds = [(0, 1) for _ in self.search_space]

        for _ in range(100):  # Sample 100 initial points for optimization
            x0 = np.random.uniform(0, 1, len(self.search_space))
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x

        return min_x

    def optimize(self):
        """
        Optimizes the Gaussian Process model to find the best parameters 
        using Bayesian optimization.

        This function initializes the optimizer with random samples, normalizes the samples, 
        and iteratively proposes new points for evaluation based on the 
        acquisition function until the specified number of calls is reached.

        Returns:
            None: Updates the best parameters and their corresponding score.
        """
        # Initialize with random samples
        self.X_sample = initialize_random_samples(self.n_initial_points, self.search_space)
        self.Y_sample = np.array([self.objective_func(x) for x in self.X_sample])

        # Normalize initial samples
        X_sample_norm = self.scale.normalize(self.X_sample)

        # Train the initial Gaussian Process model
        self.gp.fit(X_sample_norm, self.Y_sample)

        for _ in range(self.n_calls):
            X_next_norm = self.propose_next_point()

            X_next = self.scale.denormalize(X_next_norm.reshape(1, -1)).flatten()
            
            # Evaluate the objective function at the next point
            Y_next = ensure_scalar(self.objective_func(X_next))

            self.X_sample = np.vstack((self.X_sample, X_next))
            self.Y_sample = np.append(self.Y_sample, Y_next)

            X_sample_norm = self.scale.normalize(self.X_sample)

            # Optimize the hyperparameters of the Gaussian Process (length scale)
            self.gp.optimize_hyperparameters()

            # Update the GP model
            self.gp.fit(X_sample_norm, self.Y_sample)

        best_index = np.argmin(self.Y_sample)
        self.best_params = self.X_sample[best_index].tolist()
        self.best_score = self.Y_sample[best_index]

