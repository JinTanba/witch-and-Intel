"""
Interface for model caching repository.
This interface defines the contract for storing and retrieving trained models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any


class IModelCache(ABC):
    """
    Interface for caching trained models and their metadata.
    This abstraction allows the core business logic to remain independent
    of the specific caching implementation (Redis, file system, etc.).
    """
    
    @abstractmethod
    def save_model(self, tag: str, model_data: bytes) -> None:
        """
        Save a serialized model to the cache.
        
        Args:
            tag: Unique identifier for the model
            model_data: Serialized model data
        """
        pass
    
    @abstractmethod
    def load_model(self, tag: str) -> Optional[bytes]:
        """
        Load a serialized model from the cache.
        
        Args:
            tag: Unique identifier for the model
            
        Returns:
            Serialized model data or None if not found
        """
        pass
    
    @abstractmethod
    def save_metadata(self, tag: str, metadata: Dict[str, Any]) -> None:
        """
        Save model metadata.
        
        Args:
            tag: Unique identifier for the model
            metadata: Dictionary containing model metadata
        """
        pass
    
    @abstractmethod
    def load_metadata(self, tag: str) -> Optional[Dict[str, Any]]:
        """
        Load model metadata.
        
        Args:
            tag: Unique identifier for the model
            
        Returns:
            Metadata dictionary or None if not found
        """
        pass
    
    @abstractmethod
    def save_training_hashes(self, tag: str, hashes: List[str]) -> None:
        """
        Save training data hashes for refit detection.
        
        Args:
            tag: Unique identifier for the model
            hashes: List of sentence hashes used in training
        """
        pass
    
    @abstractmethod
    def load_training_hashes(self, tag: str) -> Optional[List[str]]:
        """
        Load training data hashes.
        
        Args:
            tag: Unique identifier for the model
            
        Returns:
            List of hashes or None if not found
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """
        List all available model tags.
        
        Returns:
            List of model tags
        """
        pass
    
    @abstractmethod
    def delete_model(self, tag: str) -> None:
        """
        Delete a model and its associated data.
        
        Args:
            tag: Unique identifier for the model
        """
        pass