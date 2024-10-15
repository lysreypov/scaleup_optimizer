from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    @abstractmethod
    def propose_next_point():
        """
        Propose the next point to evaluate using acquisition function.
        
        This method should return a point in the input space that is to be evaluated
        by the objective function.
        """
        pass

    @abstractmethod
    def optimize():
        """
        Run the Bayesian Optimization loop.
        
        This method should perform the optimization process, iterating through 
        calls to the acquisition function to propose new points and evaluate 
        the objective function until a stopping criterion is met.
        """
        pass


