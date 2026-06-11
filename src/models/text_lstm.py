import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TextLSTM(nn.Module):
    """
    Lab 10 (Phase 3) - RNN Track Implementation for Text (Fake News / Phishing).
    
    Architecture Justification:
    - Embedding Layer: Maps word indices to dense vectors.
    - Bidirectional LSTM: Captures context from both left and right, essential for 
      understanding sentiment and deceptive patterns in text.
    - Uses packed sequences to avoid wasting compute on padding tokens.
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int = 128, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        num_classes: int = 1,
        dropout_p: float = 0.3
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM per Lab 10 requirements
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p if num_layers > 1 else 0
        )
        
        # Bidirectional means the hidden state is 2 * hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, text_seqs: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_seqs: Tensor of shape (batch_size, max_seq_length)
            seq_lengths: Tensor of shape (batch_size) containing true lengths before padding
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Shape: (batch_size, max_seq_length, embedding_dim)
        embedded = self.embedding(text_seqs)
        
        # Lab 10 Requirement: Pack padded sequences to avoid wasting compute
        packed_embedded = pack_padded_sequence(
            embedded, 
            seq_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # We concatenate the final forward and backward hidden states from the last layer
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1) # Shape: (batch_size, hidden_dim * 2)
        
        x = self.dropout(final_hidden)
        logits = self.fc(x) # Shape: (batch_size, num_classes)
        
        return logits
