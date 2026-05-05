import pytest
import pandas as pd
import numpy as np

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.tabular_preprocessor import TabularPreprocessor

def test_text_preprocessor_shapes():
    """Smoke test for text preprocessor shapes and types."""
    # Tiny mock subset
    mock_data = pd.Series([
        "This is a FAKE news article! http://fake.com",
        "A perfectly normal piece of text with <html> tags.",
        "Short text."
    ])
    
    # Use max_length 10 so the URL isn't truncated
    preprocessor = TextPreprocessor(max_length=10, augment_p=0.0)
    
    # Must raise error if transform before fit
    with pytest.raises(RuntimeError):
        preprocessor.transform(mock_data)
        
    out = preprocessor.fit_transform(mock_data)
    
    assert isinstance(out, list)
    assert len(out) == 3
    
    # HTML stripped
    assert "<html>" not in out[1]
    # URL tokenized
    assert "[url]" in out[0] or "[URL]" in out[0]

def test_text_preprocessor_deterministic():
    """Test deterministic output given a seed during augmentation."""
    mock_data = ["This is a test sentence for deterministic augmentation"] * 5
    
    prep1 = TextPreprocessor(seed=42, augment_p=1.0)
    prep1.fit(mock_data)
    out1 = prep1.transform(mock_data, augment=True)
    
    prep2 = TextPreprocessor(seed=42, augment_p=1.0)
    prep2.fit(mock_data)
    out2 = prep2.transform(mock_data, augment=True)
    
    # Due to seed 42, the exact same augmentations (synonym/deletion) should occur
    assert out1 == out2

def test_tabular_preprocessor_shapes():
    """Smoke test for tabular preprocessor."""
    mock_data = pd.DataFrame({
        'dur': [0.1, 0.5, 100.0],
        'sbytes': [100, 500, 1000000],
        'proto': ['tcp', 'udp', 'tcp'],
        'state': ['CON', 'INT', 'FIN']
    })
    y = np.array([0, 1, 1])
    
    num_cols = ['dur', 'sbytes']
    cat_cols = ['proto', 'state']
    
    preprocessor = TabularPreprocessor(numeric_cols=num_cols, categorical_cols=cat_cols)
    
    out = preprocessor.fit_transform(mock_data)
    
    # Output must be numpy array
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 3
    # 2 numeric + (2 proto + 3 state) = 7 columns expected (since OneHotEncoder is sparse_output=False)
    assert out.shape[1] == 7
    
    # Test SMOTE
    # Provide enough samples to satisfy default k_neighbors=5
    mock_data_large = pd.concat([mock_data] * 10, ignore_index=True)
    y_large = np.array([0, 1, 1] * 10)
    
    out_large = preprocessor.fit_transform(mock_data_large)
    X_res, y_res = preprocessor.resample(out_large, y_large)
    
    # Class 0 had 10 samples, Class 1 had 20. Resampled should both have 20.
    assert len(X_res) == 40
    assert len(y_res) == 40
    assert sum(y_res == 0) == 20
    assert sum(y_res == 1) == 20

def test_tabular_leakage_prevention():
    """Ensure transform does not fit new categories (prevents leakage)."""
    train_data = pd.DataFrame({
        'dur': [0.1, 0.5],
        'sbytes': [100, 500],
        'proto': ['tcp', 'udp']
    })
    
    test_data = pd.DataFrame({
        'dur': [0.3],
        'sbytes': [200],
        'proto': ['NEW_UNSEEN_PROTO']
    })
    
    preprocessor = TabularPreprocessor(numeric_cols=['dur', 'sbytes'], categorical_cols=['proto'])
    preprocessor.fit(train_data)
    
    # Transform test data -> the unseen protocol should be ignored due to handle_unknown='ignore'
    out = preprocessor.transform(test_data)
    
    assert out.shape[0] == 1
    # 2 numeric + 2 categorical (tcp, udp)
    assert out.shape[1] == 4
    
    # The one-hot encodings for the unseen protocol should be all zeros
    assert np.all(out[0, 2:] == 0)
