import torch
import pytest
from src.models.deepfake_cnn import DeepfakeCNN
from src.models.text_lstm import TextLSTM
from src.models.tabular_deep import DeepTabularMLP

def test_deepfake_cnn_shape():
    """Test Task 2 requirement: dummy input passes through CNN and asserts output shape."""
    model = DeepfakeCNN(num_classes=1, base_channels=8) # Tiny config
    dummy_input = torch.randn(2, 3, 256, 256) # Batch size 2, 3 channels, 256x256
    output = model(dummy_input)
    assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"

def test_text_lstm_shape():
    """Test Task 2 requirement: dummy input passes through LSTM and asserts output shape."""
    vocab_size = 100
    model = TextLSTM(vocab_size=vocab_size, embedding_dim=16, hidden_dim=16, num_classes=2)
    
    # Batch of 2 sequences, max length 5
    dummy_seqs = torch.randint(1, vocab_size, (2, 5)) 
    # Must be sorted descending for basic pack_padded_sequence if enforce_sorted=True, but we set it to False
    seq_lengths = torch.tensor([5, 3]) 
    
    output = model(dummy_seqs, seq_lengths)
    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"

def test_deep_tabular_mlp_shape():
    """Test Task 2 requirement: dummy input passes through Deep MLP and asserts output shape."""
    model = DeepTabularMLP(input_dim=10, hidden_dim=16, num_blocks=2, num_classes=1)
    dummy_input = torch.randn(4, 10) # Batch size 4, 10 features
    output = model(dummy_input)
    assert output.shape == (4, 1), f"Expected shape (4, 1), got {output.shape}"
