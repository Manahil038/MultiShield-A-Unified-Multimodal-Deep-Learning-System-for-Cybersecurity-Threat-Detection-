import random
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Union

from .base import BasePreprocessor

class ImagePreprocessor(BasePreprocessor):
    """
    Modular preprocessor for image datasets (Deepfake Faces).
    Handles standard resizing (though EDA showed all are 256x256), 
    normalization, and deterministic augmentations.
    """
    def __init__(self, seed: int = 42, target_size: int = 256, augment_p: float = 0.5):
        super().__init__(seed=seed)
        self.target_size = target_size
        self.augment_p = augment_p
        
        # Base transforms: Convert to tensor [0, 1] and Normalize
        self.base_transform = T.Compose([
            T.Resize((self.target_size, self.target_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def fit(self, X, y=None):
        """
        Image preprocessing (standard normalization) doesn't require fitting statistics 
        from the dataset per our EDA, but we maintain the interface.
        """
        self.is_fitted = True
        return self

    def _get_augmentation_transform(self) -> T.Compose:
        """
        Returns a composed transform with 2 deterministic augmentations.
        - RandomHorizontalFlip
        - ColorJitter
        We seed torch manually right before applying to ensure determinism.
        """
        return T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=self.augment_p),
            T.RandomHorizontalFlip(p=self.augment_p)
        ])

    def transform(self, X: Union[List[Image.Image], List[str]], augment: bool = False) -> torch.Tensor:
        """
        Apply base transformations and optional augmentations.
        
        Args:
            X: List of PIL Images or list of file paths.
            augment: If True, applies augmentation techniques (should only be True for training).
            
        Returns:
            A batched torch.Tensor of shape (N, C, H, W).
        """
        self._check_is_fitted()
        
        processed = []
        
        # We set the torch manual seed to ensure deterministic augmentations if augment=True
        if augment:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            aug_transform = self._get_augmentation_transform()
            
        for item in X:
            if isinstance(item, str):
                img = Image.open(item).convert("RGB")
            else:
                img = item.convert("RGB")
                
            if augment:
                # Apply PIL-based augmentations first
                img = aug_transform(img)
                
            # Apply base transforms (Resize -> ToTensor -> Normalize)
            tensor_img = self.base_transform(img)
            processed.append(tensor_img)
            
        return torch.stack(processed)
