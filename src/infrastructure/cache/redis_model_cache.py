"""
Redis implementation of the IModelCache interface.
This is an infrastructure concern that implements the cache abstraction.
"""

import os
import json
import base64
from typing import Optional, Dict, List, Any
import requests

from src.application.repositories.IModelCache import IModelCache


class RedisModelCache(IModelCache):
    """
    Redis-based implementation of model caching.
    Uses Upstash Redis REST API for serverless compatibility.
    """
    
    def __init__(self, redis_url: Optional[str] = None, redis_token: Optional[str] = None):
        """
        Initialize Redis cache connection.
        
        Args:
            redis_url: Upstash Redis REST URL (uses env var if not provided)
            redis_token: Upstash Redis REST token (uses env var if not provided)
        """
        self.redis_url = redis_url or os.environ.get('UPSTASH_REDIS_REST_URL')
        self.redis_token = redis_token or os.environ.get('UPSTASH_REDIS_REST_TOKEN')
        
        if not self.redis_url or not self.redis_token:
            raise ValueError("Redis URL and token must be provided or set in environment variables")
    
    def _redis_request(self, command: List[str]) -> any:
        """Execute a Redis command via REST API."""
        # Upstash expects the command as an array in the body
        response = requests.post(
            self.redis_url,
            headers={
                "Authorization": f"Bearer {self.redis_token}",
            },
            json=command
        )
        response.raise_for_status()
        return response.json().get('result')
    
    def _get_key(self, tag: str, suffix: str) -> str:
        """Generate Redis key for a specific model component."""
        return f"ttcm:{tag}:{suffix}"
    
    def save_model(self, tag: str, model_data: bytes) -> None:
        """Save a serialized model to Redis."""
        key = self._get_key(tag, "model")
        # Encode binary data as base64 for JSON transport
        encoded_data = base64.b64encode(model_data).decode('utf-8')
        self._redis_request(['SET', key, encoded_data])
    
    def load_model(self, tag: str) -> Optional[bytes]:
        """Load a serialized model from Redis."""
        key = self._get_key(tag, "model")
        encoded_data = self._redis_request(['GET', key])
        if encoded_data:
            return base64.b64decode(encoded_data)
        return None
    
    def save_metadata(self, tag: str, metadata: Dict[str, Any]) -> None:
        """Save model metadata to Redis."""
        key = self._get_key(tag, "metadata")
        self._redis_request(['SET', key, json.dumps(metadata)])
        # Add to model set
        self._redis_request(['SADD', 'ttcm:models', tag])
    
    def load_metadata(self, tag: str) -> Optional[Dict[str, Any]]:
        """Load model metadata from Redis."""
        key = self._get_key(tag, "metadata")
        json_data = self._redis_request(['GET', key])
        if json_data:
            return json.loads(json_data)
        return None
    
    def save_training_hashes(self, tag: str, hashes: List[str]) -> None:
        """Save training data hashes to Redis."""
        key = self._get_key(tag, "hashes")
        self._redis_request(['SET', key, json.dumps(hashes)])
    
    def load_training_hashes(self, tag: str) -> Optional[List[str]]:
        """Load training data hashes from Redis."""
        key = self._get_key(tag, "hashes")
        json_data = self._redis_request(['GET', key])
        if json_data:
            return json.loads(json_data)
        return None
    
    def list_models(self) -> List[str]:
        """List all available model tags."""
        tags = self._redis_request(['SMEMBERS', 'ttcm:models'])
        return tags if tags else []
    
    def delete_model(self, tag: str) -> None:
        """Delete a model and its associated data."""
        # Delete all related keys
        for suffix in ['model', 'metadata', 'hashes']:
            key = self._get_key(tag, suffix)
            self._redis_request(['DEL', key])
        
        # Remove from model set
        self._redis_request(['SREM', 'ttcm:models', tag])