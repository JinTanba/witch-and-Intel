"""
Main entry point for TextToCoordinatesMapper with dependency injection.
This is where we wire together the application components.
"""

import os
from typing import List, Optional
import numpy as np

from src.core.text_to_coordinates_mapper import TextToCoordinatesMapper
from src.infrastructure.cache.redis_model_cache import RedisModelCache


class TextToCoordinatesMapperFactory:
    """
    Factory for creating TextToCoordinatesMapper instances with proper dependencies.
    This handles the dependency injection and configuration.
    """
    
    @staticmethod
    def create_with_redis(
        model_name: str = 'all-mpnet-base-v2',
        redis_url: Optional[str] = None,
        redis_token: Optional[str] = None,
        **kwargs
    ) -> TextToCoordinatesMapper:
        """
        Create a TextToCoordinatesMapper with Redis caching.
        
        Args:
            model_name: Name of the sentence transformer model
            redis_url: Redis URL (uses env var if not provided)
            redis_token: Redis token (uses env var if not provided)
            **kwargs: Additional parameters for TextToCoordinatesMapper
            
        Returns:
            Configured TextToCoordinatesMapper instance
        """
        # Create cache implementation
        cache = RedisModelCache(redis_url, redis_token)
        
        # Create mapper with injected dependencies
        return TextToCoordinatesMapper(
            cache=cache,
            model_name=model_name,
            **kwargs
        )


# Convenience function for simple usage
def create_mapper() -> TextToCoordinatesMapper:
    """
    Create a default TextToCoordinatesMapper instance.
    Uses environment variables for Redis configuration.
    """
    return TextToCoordinatesMapperFactory.create_with_redis()


# Example usage
if __name__ == "__main__":
    # Set environment variables (normally these would be set externally)
    os.environ['UPSTASH_REDIS_REST_URL'] = 'https://humorous-adder-10993.upstash.io'
    os.environ['UPSTASH_REDIS_REST_TOKEN'] = 'ASrxAAIjcDE2ZmZjZjU1OGY1ZmQ0MjBiYmEyMWZkODZlODZiYjFkOHAxMA'
    
    # Create mapper instance
    mapper = create_mapper()
    
    # Training sentences
    training_sentences = [
        "Machine learning is a subset of artificial intelligence",
        "AI and machine learning are closely related fields",
        "Deep learning is a type of machine learning algorithm",
        "Neural networks are used in machine learning applications",
        "Supervised learning requires labeled training data",
        "Unsupervised learning finds patterns in unlabeled data",
        "The weather today is sunny and warm",
        "It's a beautiful day outside",
        "I love cooking pasta with fresh tomatoes",
        "Pizza is my favorite food"
    ]
    
    # Fit the model
    mapper.fit(training_sentences, tag="example_v1")
    
    # Transform new sentences
    new_sentences = [
        "Natural language processing uses machine learning",
        "The forecast shows rain tomorrow",
        "I enjoy making homemade bread"
    ]
    
    # Note: transform now requires the tag parameter
    coordinates = mapper.transform(new_sentences, tag="example_v1")
    print("New sentence coordinates:")
    for sentence, coord in zip(new_sentences, coordinates):
        print(f"  {sentence[:50]}: ({coord[0]:.3f}, {coord[1]:.3f})")
    
    # List available models
    print("\nAvailable models:")
    for model in mapper.list_models():
        print(f"  Tag: {model['tag']}, Samples: {model['n_samples']}, Time: {model['timestamp']}")