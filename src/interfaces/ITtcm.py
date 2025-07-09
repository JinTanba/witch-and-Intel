"""
Interface for Text to Coordinates Mapper.
This interface defines the contract for converting text to 2D coordinates.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class ITextToCoordinatesMapper(ABC):
    """
    Interface for mapping text sentences to 2D coordinates.
    
    Any implementation of this interface must provide a way to convert
    a list of text sentences into 2D spatial coordinates that represent
    their semantic relationships.
    """
    
    @abstractmethod
    def fit(self, sentences: List[str], tag: str = "default") -> None:
        """
        Train the coordinate transformation model.
        
        Args:
            sentences: List of training sentences
            tag: Tag to identify this model version
            
        Raises:
            ValueError: If sentences list is empty
        """
        pass
    
    @abstractmethod
    def transform(self, sentences: List[str], tag: str = "default") -> np.ndarray:
        """
        Apply coordinate transformation to sentences.
        
        Args:
            sentences: List of sentences to transform
            tag: Tag of the model to use
            
        Returns:
            Array of shape (n_sentences, 2) containing 2D coordinates
            
        Raises:
            ValueError: If sentences list is empty or model not found
        """
        pass
    
    @abstractmethod
    def fit_transform(self, sentences: List[str], tag: Optional[str] = None) -> np.ndarray:
        """
        Fit the model and transform sentences in one step.
        
        Args:
            sentences: List of sentences to fit and transform
            tag: Tag for the model (optional)
            
        Returns:
            Array of shape (n_sentences, 2) containing 2D coordinates
            
        Raises:
            ValueError: If sentences list is empty
        """
        pass
