from abc import ABC, abstractmethod
import numpy as np


class BaseAcquisitionFunction(ABC):
    @abstractmethod
    def evaluation():
        """
        Computes the acquisition function.
        """
        pass