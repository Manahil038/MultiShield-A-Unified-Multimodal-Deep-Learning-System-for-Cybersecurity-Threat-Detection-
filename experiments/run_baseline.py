import os
import sys
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_csv_from_zip, split_data
from src.preprocessing.tabular_preprocessor import TabularPreprocessor
from src.models.baseline_mlp import BaselineMLP
from src.training.train import train_model
from src.utils.helpers import seed_everything

seed_everything(42)

def main():
    print("=== MULTISHIELD: BASELINE MLP EXPERIMENT (UNSW-NB15) ===")
    
    # 1. Load Data
    print("[1/5] Loading Tabular Data (UNSW-NB15)...")
    zip_path = os.path.join("data", "behavioral_anomalies.zip")
    df = load_csv_from_zip(zip_path, "UNSW_NB15_training-set.csv")
    
    # For baseline speed, we can optionally subsample, but UNSW-NB15 training-set isn't insanely huge
    # (around 175k rows). We'll use 50k rows to keep the lab experiment fast but representative.
    df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    
    y = df['label'].values
    X = df.drop(columns=['label', 'attack_cat', 'id'])
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # 2. Strict Train/Val/Test Split
    print("[2/5] Splitting Data (70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, val_size=0.15, test_size=0.15, seed=42, stratify=y
    )

    # 3. Preprocessing (Fit on Train ONLY)
    print("[3/5] Preprocessing (Scaling, Encoding, SMOTE Resampling)...")
    preprocessor = TabularPreprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, augment_p=0.2)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_train_resampled, y_train_resampled = preprocessor.resample(X_train_processed, y_train)
    
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train_resampled), torch.FloatTensor(y_train_resampled))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_processed), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_processed), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    input_dim = X_train_resampled.shape[1]
    print(f"Data shapes: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test ({len(test_dataset)})")
    print(f"Input features dimension after encoding: {input_dim}")

    # 4. Instantiate Model
    print("[4/5] Instantiating Baseline MLP (He Initialized, GELU)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BaselineMLP(
        input_dim=input_dim, 
        hidden_dims=[128, 64], 
        output_dim=1, 
        dropout_p=0.3
    )
    
    # 5. Train & Track
    print(f"[5/5] Starting Training on {device} (with TensorBoard Tracking)...")
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader, # ONLY USED EXACTLY ONCE AT THE END BY train_model!
        task_type='binary',
        epochs=15, # Baseline epochs
        lr=0.001,
        patience=3,
        device=device,
        save_dir=os.path.join("experiments", "checkpoints"),
        log_dir=os.path.join("experiments", "runs", "baseline_unsw_nb15")
    )
    
    print("Baseline Experiment Completed Successfully.")

if __name__ == "__main__":
    main()
