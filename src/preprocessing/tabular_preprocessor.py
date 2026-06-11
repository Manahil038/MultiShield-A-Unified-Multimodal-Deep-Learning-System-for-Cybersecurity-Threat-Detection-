import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

from .base import BasePreprocessor

class TabularPreprocessor(BasePreprocessor):
    """
    Modular preprocessor for tabular datasets (UNSW-NB15).
    Handles missing values, log-transforms, scaling, encoding, and class imbalance.
    """
    def __init__(self, numeric_cols: List[str], categorical_cols: List[str], seed: int = 42, augment_p: float = 0.0):
        super().__init__(seed=seed)
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.augment_p = augment_p
        
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.smote = SMOTE(random_state=seed)
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler and encoder on training data to prevent leakage."""
        # 1. Fill missing values (EDA found attack_cat has missing -> implies Normal, but we handle X features here)
        X_clean = X.copy()
        for col in self.categorical_cols:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna('Unknown')
                
        for col in self.numeric_cols:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
                
        # 2. Log-transform numeric features that we identified as heavy-tailed
        X_num = np.log1p(X_clean[self.numeric_cols].clip(lower=0))
        
        # 3. Fit scaler
        self.scaler.fit(X_num)
        
        # 4. Fit encoder
        self.encoder.fit(X_clean[self.categorical_cols])
        
        self.is_fitted = True
        return self

    def _augment_noise(self, X_num: np.ndarray) -> np.ndarray:
        """Gaussian noise injection on numerical features (Augmentation 1)."""
        np.random.seed(self.seed)
        noise = np.random.normal(0, 0.01, X_num.shape)
        return X_num + noise

    def _augment_masking(self, X_cat: np.ndarray, p: float = 0.05) -> np.ndarray:
        """Random feature masking on encoded tabular data (Augmentation 2)."""
        np.random.seed(self.seed)
        mask = np.random.binomial(1, 1 - p, X_cat.shape)
        return X_cat * mask

    def transform(self, X: pd.DataFrame, augment: bool = False) -> np.ndarray:
        """
        Apply learned transformations to data.
        
        Args:
            X: Dataframe containing the same columns as fit data.
            augment: If True, apply tabular augmentations.
        """
        self._check_is_fitted()
        X_clean = X.copy()
        
        # Handle missing columns and fill missing values
        for col in self.categorical_cols:
            if col not in X_clean.columns:
                X_clean[col] = 'Unknown'
            else:
                X_clean[col] = X_clean[col].fillna('Unknown')
                
        for col in self.numeric_cols:
            if col not in X_clean.columns:
                X_clean[col] = 0.0
            else:
                X_clean[col] = X_clean[col].fillna(0.0)
                
        # Numeric processing
        X_num = np.log1p(X_clean[self.numeric_cols].clip(lower=0))
        X_num_scaled = self.scaler.transform(X_num)
        
        # Categorical processing
        X_cat_encoded = self.encoder.transform(X_clean[self.categorical_cols])
        
        # Augmentations (Only during training)
        if augment and np.random.random() < self.augment_p:
            X_num_scaled = self._augment_noise(X_num_scaled)
            X_cat_encoded = self._augment_masking(X_cat_encoded)
            
        # Combine
        return np.hstack((X_num_scaled, X_cat_encoded))

    def resample(self, X_transformed: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to handle class imbalance (e.g., in UNSW-NB15 attack categories).
        Must be called AFTER fit_transform on the training set ONLY.
        """
        self._check_is_fitted()
        X_resampled, y_resampled = self.smote.fit_resample(X_transformed, y)
        return X_resampled, y_resampled
