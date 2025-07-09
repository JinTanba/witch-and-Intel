"""
Test suite for RedisModelCache implementation.
Tests the Redis-based caching functionality with mocked HTTP requests.
"""

import os
import json
import base64
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.infrastructure.cache.redis_model_cache import RedisModelCache


class TestRedisModelCache:
    """Test suite for RedisModelCache."""
    
    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Mock environment variables for Redis connection."""
        monkeypatch.setenv('UPSTASH_REDIS_REST_URL', 'https://test-redis.upstash.io')
        monkeypatch.setenv('UPSTASH_REDIS_REST_TOKEN', 'test-token-123')
    
    @pytest.fixture
    def redis_cache(self, mock_env_vars):
        """Create a RedisModelCache instance with mocked environment."""
        return RedisModelCache()
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests module for Redis REST API calls."""
        with patch('src.infrastructure.cache.redis_model_cache.requests') as mock:
            yield mock
    
    def test_init_with_parameters(self):
        """Test initialization with explicit parameters."""
        cache = RedisModelCache(
            redis_url='https://custom-redis.upstash.io',
            redis_token='custom-token'
        )
        assert cache.redis_url == 'https://custom-redis.upstash.io'
        assert cache.redis_token == 'custom-token'
    
    def test_init_with_env_vars(self, mock_env_vars):
        """Test initialization using environment variables."""
        cache = RedisModelCache()
        assert cache.redis_url == 'https://test-redis.upstash.io'
        assert cache.redis_token == 'test-token-123'
    
    def test_init_missing_credentials(self):
        """Test initialization fails when credentials are missing."""
        with pytest.raises(ValueError, match="Redis URL and token must be provided"):
            RedisModelCache()
    
    def test_save_model(self, redis_cache, mock_requests):
        """Test saving a model to Redis."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {'result': 'OK'}
        mock_requests.post.return_value = mock_response
        
        # Test data
        model_data = b'test model binary data'
        tag = 'test-model-v1'
        
        # Execute
        redis_cache.save_model(tag, model_data)
        
        # Verify
        expected_data = base64.b64encode(model_data).decode('utf-8')
        mock_requests.post.assert_called_once_with(
            'https://test-redis.upstash.io',
            headers={
                'Authorization': 'Bearer test-token-123',
            },
            json=['SET', 'ttcm:test-model-v1:model', expected_data]
        )
    
    def test_load_model_exists(self, redis_cache, mock_requests):
        """Test loading an existing model from Redis."""
        # Setup mock response
        original_data = b'test model binary data'
        encoded_data = base64.b64encode(original_data).decode('utf-8')
        
        mock_response = Mock()
        mock_response.json.return_value = {'result': encoded_data}
        mock_requests.post.return_value = mock_response
        
        # Execute
        result = redis_cache.load_model('test-model-v1')
        
        # Verify
        assert result == original_data
        mock_requests.post.assert_called_once_with(
            'https://test-redis.upstash.io',
            headers={
                'Authorization': 'Bearer test-token-123',
            },
            json=['GET', 'ttcm:test-model-v1:model']
        )
    
    def test_load_model_not_exists(self, redis_cache, mock_requests):
        """Test loading a non-existent model returns None."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {'result': None}
        mock_requests.post.return_value = mock_response
        
        # Execute
        result = redis_cache.load_model('non-existent-model')
        
        # Verify
        assert result is None
    
    def test_save_metadata(self, redis_cache, mock_requests):
        """Test saving model metadata."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {'result': 'OK'}
        mock_requests.post.return_value = mock_response
        
        # Test data
        metadata = {
            'accuracy': 0.95,
            'training_time': 120.5,
            'parameters': {'learning_rate': 0.01}
        }
        tag = 'test-model-v1'
        
        # Execute
        redis_cache.save_metadata(tag, metadata)
        
        # Verify - should make 2 calls (SET and SADD)
        assert mock_requests.post.call_count == 2
        
        # First call - SET metadata
        first_call = mock_requests.post.call_args_list[0]
        assert first_call[0][0] == 'https://test-redis.upstash.io'
        assert first_call[1]['json'] == ['SET', 'ttcm:test-model-v1:metadata', json.dumps(metadata)]
        
        # Second call - SADD to model set
        second_call = mock_requests.post.call_args_list[1]
        assert second_call[0][0] == 'https://test-redis.upstash.io'
        assert second_call[1]['json'] == ['SADD', 'ttcm:models', 'test-model-v1']
    
    def test_load_metadata(self, redis_cache, mock_requests):
        """Test loading model metadata."""
        # Setup mock response
        metadata = {
            'accuracy': 0.95,
            'training_time': 120.5,
            'parameters': {'learning_rate': 0.01}
        }
        mock_response = Mock()
        mock_response.json.return_value = {'result': json.dumps(metadata)}
        mock_requests.post.return_value = mock_response
        
        # Execute
        result = redis_cache.load_metadata('test-model-v1')
        
        # Verify
        assert result == metadata
    
    def test_save_training_hashes(self, redis_cache, mock_requests):
        """Test saving training data hashes."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {'result': 'OK'}
        mock_requests.post.return_value = mock_response
        
        # Test data
        hashes = ['hash1', 'hash2', 'hash3']
        tag = 'test-model-v1'
        
        # Execute
        redis_cache.save_training_hashes(tag, hashes)
        
        # Verify
        mock_requests.post.assert_called_once_with(
            'https://test-redis.upstash.io',
            headers={
                'Authorization': 'Bearer test-token-123',
            },
            json=['SET', 'ttcm:test-model-v1:hashes', json.dumps(hashes)]
        )
    
    def test_load_training_hashes(self, redis_cache, mock_requests):
        """Test loading training data hashes."""
        # Setup mock response
        hashes = ['hash1', 'hash2', 'hash3']
        mock_response = Mock()
        mock_response.json.return_value = {'result': json.dumps(hashes)}
        mock_requests.post.return_value = mock_response
        
        # Execute
        result = redis_cache.load_training_hashes('test-model-v1')
        
        # Verify
        assert result == hashes
    
    def test_list_models(self, redis_cache, mock_requests):
        """Test listing all available models."""
        # Setup mock response
        models = ['model-v1', 'model-v2', 'model-v3']
        mock_response = Mock()
        mock_response.json.return_value = {'result': models}
        mock_requests.post.return_value = mock_response
        
        # Execute
        result = redis_cache.list_models()
        
        # Verify
        assert result == models
        mock_requests.post.assert_called_once_with(
            'https://test-redis.upstash.io',
            headers={
                'Authorization': 'Bearer test-token-123',
            },
            json=['SMEMBERS', 'ttcm:models']
        )
    
    def test_list_models_empty(self, redis_cache, mock_requests):
        """Test listing models when none exist."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {'result': None}
        mock_requests.post.return_value = mock_response
        
        # Execute
        result = redis_cache.list_models()
        
        # Verify
        assert result == []
    
    def test_delete_model(self, redis_cache, mock_requests):
        """Test deleting a model and all associated data."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {'result': 1}
        mock_requests.post.return_value = mock_response
        
        # Execute
        redis_cache.delete_model('test-model-v1')
        
        # Verify - should make 4 calls (3 DEL + 1 SREM)
        assert mock_requests.post.call_count == 4
        
        # Check DEL calls for each suffix
        expected_keys = [
            'ttcm:test-model-v1:model',
            'ttcm:test-model-v1:metadata',
            'ttcm:test-model-v1:hashes'
        ]
        
        for i, key in enumerate(expected_keys):
            call = mock_requests.post.call_args_list[i]
            assert call[0][0] == 'https://test-redis.upstash.io'
            assert call[1]['json'] == ['DEL', key]
        
        # Check SREM call
        last_call = mock_requests.post.call_args_list[3]
        assert last_call[0][0] == 'https://test-redis.upstash.io'
        assert last_call[1]['json'] == ['SREM', 'ttcm:models', 'test-model-v1']
    
    def test_redis_request_error_handling(self, redis_cache, mock_requests):
        """Test error handling when Redis request fails."""
        # Setup mock to raise an error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("Redis error")
        mock_requests.post.return_value = mock_response
        
        # Execute and verify exception is raised
        with pytest.raises(Exception, match="Redis error"):
            redis_cache.save_model('test-model', b'data')
    
    def test_get_key_format(self, redis_cache):
        """Test the key generation format."""
        assert redis_cache._get_key('test-model', 'model') == 'ttcm:test-model:model'
        assert redis_cache._get_key('prod-v1', 'metadata') == 'ttcm:prod-v1:metadata'
        assert redis_cache._get_key('model-123', 'hashes') == 'ttcm:model-123:hashes'


class TestRedisModelCacheIntegration:
    """Integration tests for RedisModelCache with real-like scenarios."""
    
    @pytest.fixture
    def redis_cache(self):
        """Create a RedisModelCache instance with test credentials."""
        return RedisModelCache(
            redis_url='https://test-redis.upstash.io',
            redis_token='test-token'
        )
    
    @pytest.fixture
    def mock_successful_requests(self):
        """Mock requests for successful operations."""
        with patch('src.infrastructure.cache.redis_model_cache.requests') as mock:
            mock_response = Mock()
            mock_response.json.return_value = {'result': 'OK'}
            mock.post.return_value = mock_response
            yield mock
    
    def test_full_model_lifecycle(self, redis_cache, mock_requests):
        """Test complete model save, load, and delete lifecycle."""
        tag = 'test-model-lifecycle'
        model_data = b'model binary data'
        metadata = {'version': 1, 'accuracy': 0.92}
        hashes = ['hash1', 'hash2']
        
        # Setup mock responses for different operations
        responses = [
            {'result': 'OK'},  # save_model
            {'result': 'OK'},  # save_metadata SET
            {'result': 1},     # save_metadata SADD
            {'result': 'OK'},  # save_training_hashes
            {'result': base64.b64encode(model_data).decode('utf-8')},  # load_model
            {'result': json.dumps(metadata)},  # load_metadata
            {'result': json.dumps(hashes)},  # load_training_hashes
            {'result': 1},  # delete_model DEL 1
            {'result': 1},  # delete_model DEL 2
            {'result': 1},  # delete_model DEL 3
            {'result': 1},  # delete_model SREM
        ]
        
        mock_response = Mock()
        mock_response.json.side_effect = responses
        mock_requests.post.return_value = mock_response
        
        # Save operations
        redis_cache.save_model(tag, model_data)
        redis_cache.save_metadata(tag, metadata)
        redis_cache.save_training_hashes(tag, hashes)
        
        # Load operations
        loaded_model = redis_cache.load_model(tag)
        loaded_metadata = redis_cache.load_metadata(tag)
        loaded_hashes = redis_cache.load_training_hashes(tag)
        
        # Verify loaded data
        assert loaded_model == model_data
        assert loaded_metadata == metadata
        assert loaded_hashes == hashes
        
        # Delete operation
        redis_cache.delete_model(tag)
        
        # Verify all operations were called
        assert mock_requests.post.call_count == 11