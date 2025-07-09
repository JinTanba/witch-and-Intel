"""
In-memory implementation of IModelCache for testing and development
"""

from typing import Dict, Any, Optional
from src.application.repositories.IModelCache import IModelCache


class InMemoryModelCache(IModelCache):
    """In-memory cache implementation for testing"""
    
    def __init__(self):
        self._cache: Dict[str, bytes] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def save_model(self, key: str, model_data: bytes) -> None:
        """Save model data to memory"""
        self._cache[key] = model_data
    
    def load_model(self, key: str) -> Optional[bytes]:
        """Load model data from memory"""
        return self._cache.get(key)
    
    def exists(self, key: str) -> bool:
        """Check if model exists in memory"""
        return key in self._cache
    
    def delete(self, key: str) -> None:
        """Delete model from memory"""
        if key in self._cache:
            del self._cache[key]
        if key in self._metadata:
            del self._metadata[key]
    
    def save_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to memory"""
        self._metadata[key] = metadata
    
    def load_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Load metadata from memory"""
        return self._metadata.get(key)
    
    def list_keys(self, pattern: Optional[str] = None) -> list:
        """List all keys, optionally filtered by pattern"""
        keys = list(self._cache.keys())
        if pattern:
            # Simple pattern matching (supports * wildcard)
            import fnmatch
            keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        return keys