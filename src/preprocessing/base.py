"""
Base classes and interfaces for the preprocessing pipeline.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessors in the MultiShield pipeline.
    Enforces a standard sklearn-like interface.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X, y=None):
        """Fit the preprocessor statistics on the training set."""
        pass
        
    @abstractmethod
    def transform(self, X):
        """Apply the preprocessing transformation to the data."""
        pass
        
    def fit_transform(self, X, y=None):
        """Fit the preprocessor on the data and then transform it."""
        self.fit(X, y)
        return self.transform(X)
        
    def _check_is_fitted(self):
        """Ensure the preprocessor has been fitted before transforming."""
        if not self.is_fitted:
            raise RuntimeError(f"This {self.__class__.__name__} instance is not fitted yet. "
                               "Call 'fit' with appropriate arguments before using this estimator.")
