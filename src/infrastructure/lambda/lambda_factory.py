"""
Factory for creating Lambda handlers with proper dependency injection
"""

import os
from src.application.handlers.TextProcessingHandler import TextProcessingHandler
from src.core.text_to_coordinates_mapper import TextToCoordinatesMapper
from src.infrastructure.cache.redis_model_cache import RedisModelCache
from src.infrastructure.cache.in_memory_model_cache import InMemoryModelCache


def create_production_handler() -> TextProcessingHandler:
    """
    Create handler with production dependencies
    
    Returns:
        TextProcessingHandler configured for production
    """
    # Determine cache implementation based on environment
    if os.getenv('USE_REDIS_CACHE', 'true').lower() == 'true':
        try:
            model_cache = RedisModelCache()
        except Exception:
            # Fallback to in-memory cache if Redis is not available
            print("Warning: Redis not available, using in-memory cache")
            model_cache = InMemoryModelCache()
    else:
        model_cache = InMemoryModelCache()
    
    # Create text mapper with cache
    text_mapper = TextToCoordinatesMapper(model_cache=model_cache)
    
    # Create and return handler
    return TextProcessingHandler(text_mapper)


def create_test_handler(text_mapper=None) -> TextProcessingHandler:
    """
    Create handler for testing with optional mock dependencies
    
    Args:
        text_mapper: Optional text mapper instance for testing
        
    Returns:
        TextProcessingHandler configured for testing
    """
    if text_mapper is None:
        # Use in-memory cache for testing
        model_cache = InMemoryModelCache()
        text_mapper = TextToCoordinatesMapper(model_cache=model_cache)
    
    return TextProcessingHandler(text_mapper)