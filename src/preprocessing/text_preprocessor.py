import re
import random
import numpy as np
import pandas as pd
from typing import List, Union
import nltk
from nltk.corpus import wordnet

# Download wordnet if not already available
try:
    wordnet.ensure_loaded()
except Exception:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('wordnet', quiet=True)

from .base import BasePreprocessor

class TextPreprocessor(BasePreprocessor):
    """
    Modular preprocessor for textual datasets (Fake News, Phishing).
    Handles cleaning, tokenization-prep (padding/truncation), and augmentation.
    """
    def __init__(self, seed: int = 42, max_length: int = 512, augment_p: float = 0.0):
        super().__init__(seed=seed)
        self.max_length = max_length
        self.augment_p = augment_p
        
    def fit(self, X, y=None):
        """
        Text preprocessing doesn't require fitting statistics for these specific operations,
        but we maintain the interface.
        """
        self.is_fitted = True
        return self
        
    def _clean_text(self, text: str) -> str:
        """Lowercasing, HTML stripping, URL replacement."""
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # URL tokenization
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _truncate_pad(self, text: str) -> str:
        """
        Enforce max word length limit to simulate BERT token limits.
        Actual WordPiece tokenization happens later in the DL pipeline.
        """
        words = text.split()
        if len(words) > self.max_length:
            return ' '.join(words[:self.max_length])
        return text

    def _augment_synonym(self, text: str) -> str:
        """Synonym replacement (Augmentation Strategy 1)"""
        words = text.split()
        if not words:
            return text
            
        n_replacements = max(1, int(len(words) * 0.15))
        replaced_words = words.copy()
        
        # Pick random indices to replace
        random.seed(self.seed)
        replace_idxs = random.sample(range(len(words)), min(n_replacements, len(words)))
        
        for idx in replace_idxs:
            word = words[idx]
            synonyms = set()
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonym = l.name().replace("_", " ").replace("-", " ").lower()
                    if synonym != word and synonym.isalpha():
                        synonyms.add(synonym)
            if synonyms:
                replaced_words[idx] = random.choice(list(synonyms))
                
        return ' '.join(replaced_words)

    def _augment_deletion(self, text: str, p: float = 0.1) -> str:
        """Random word deletion (Augmentation Strategy 2)"""
        words = text.split()
        if len(words) == 1:
            return text
            
        random.seed(self.seed)
        remaining = [w for w in words if random.random() > p]
        
        # Ensure at least one word remains
        if not remaining:
            remaining = [random.choice(words)]
            
        return ' '.join(remaining)

    def transform(self, X: Union[List[str], pd.Series], augment: bool = False) -> List[str]:
        """
        Clean, truncate, and optionally augment the text data.
        
        Args:
            X: List of text strings or pandas Series.
            augment: If True, applies augmentation techniques (should only be True for training).
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.Series):
            X_list = X.tolist()
        else:
            X_list = list(X)
            
        processed = []
        # Ensure reproducible augmentations
        random.seed(self.seed)
        
        for text in X_list:
            cleaned = self._clean_text(text)
            
            if augment and random.random() < self.augment_p:
                if random.random() < 0.5:
                    cleaned = self._augment_synonym(cleaned)
                else:
                    cleaned = self._augment_deletion(cleaned, p=0.1)
                    
            truncated = self._truncate_pad(cleaned)
            processed.append(truncated)
            
        return processed

import collections
import torch

class SimpleTokenizer:
    """
    A basic vocabulary builder and integer encoder for text sequences.
    Required to feed text into an LSTM.
    """
    def __init__(self, vocab_size: int = 10000, unk_token: str = '<UNK>', pad_token: str = '<PAD>'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        self.is_fitted = False
        
    def fit(self, X: List[str]):
        """Build the vocabulary based on term frequency."""
        counter = collections.Counter()
        for text in X:
            counter.update(text.split())
            
        # Keep most common words, leaving space for PAD and UNK
        most_common = counter.most_common(self.vocab_size - 2)
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.is_fitted = True
        
    def transform(self, X: List[str], max_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of strings into padded integer tensors and sequence lengths.
        Returns:
            padded_seqs: Tensor of shape (len(X), max_len)
            lengths: Tensor of shape (len(X),) containing true lengths before padding
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before calling transform.")
            
        seqs = []
        lengths = []
        
        for text in X:
            words = text.split()
            seq = [self.word2idx.get(w, self.word2idx[self.unk_token]) for w in words]
            # Truncate
            seq = seq[:max_len]
            # Length
            length = len(seq) if len(seq) > 0 else 1 # Avoid 0-length
            if len(seq) == 0:
                seq = [self.word2idx[self.unk_token]]
                
            lengths.append(length)
            
            # Pad
            padded_seq = seq + [self.word2idx[self.pad_token]] * (max_len - len(seq))
            seqs.append(padded_seq)
            
        return torch.tensor(seqs, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
