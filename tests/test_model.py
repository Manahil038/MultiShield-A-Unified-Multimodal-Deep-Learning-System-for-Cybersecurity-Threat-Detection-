import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytest

from src.models.baseline_mlp import BaselineMLP
from src.training.train import train_model

def test_mlp_initialization_and_forward():
    """Verify the MLP architecture builds and forward pass outputs correct shape."""
    model = BaselineMLP(input_dim=10, hidden_dims=[32, 16], output_dim=1, dropout_p=0.1)
    
    # 1. Test Architecture Structure
    # Should have Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear
    assert isinstance(model.network[0], nn.Linear)
    assert isinstance(model.network[1], nn.GELU)
    assert isinstance(model.network[2], nn.Dropout)
    assert isinstance(model.network[-1], nn.Linear)
    assert model.network[-1].out_features == 1
    
    # 2. Test Forward Pass Shape
    batch_size = 4
    x_mock = torch.randn(batch_size, 10)
    logits = model(x_mock)
    
    assert logits.shape == (batch_size, 1)

def test_training_loop_binary(tmpdir):
    """Smoke test for the training loop on a dummy binary classification task."""
    # Create simple XOR-like dummy data
    X = torch.randn(20, 5)
    # y = 1 if sum > 0 else 0
    y = (X.sum(dim=1) > 0).long()
    
    dataset = TensorDataset(X, y)
    
    # Tiny splits
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4)
    test_loader = DataLoader(dataset, batch_size=4)
    
    model = BaselineMLP(input_dim=5, hidden_dims=[10], output_dim=1)
    
    # Run loop
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task_type='binary',
        epochs=3, # Tiny epochs just to test loop mechanics
        lr=0.01,
        patience=2,
        device='cpu',
        save_dir=str(tmpdir)
    )
    
    # Ensure it returns expected dict
    assert 'history' in results
    assert 'test_loss' in results
    assert 'test_acc' in results
    
    # Ensure checkpoint was saved
    assert (tmpdir / 'best_model.pth').exists()

def test_training_loop_multiclass(tmpdir):
    """Smoke test for the training loop on a dummy multi-class task."""
    # 3 classes
    X = torch.randn(30, 5)
    y = torch.randint(0, 3, (30,))
    
    dataset = TensorDataset(X, y)
    
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8)
    test_loader = DataLoader(dataset, batch_size=8)
    
    # Output dim is 3
    model = BaselineMLP(input_dim=5, hidden_dims=[10], output_dim=3)
    
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task_type='multiclass',
        epochs=2,
        device='cpu',
        save_dir=str(tmpdir)
    )
    
    assert results['test_acc'] >= 0.0
