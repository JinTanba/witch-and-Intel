"""
Example demonstrating Clean Architecture principles with TextToCoordinatesMapper.
Shows proper dependency injection and separation of concerns.
"""

import os
from typing import List
import numpy as np

# Application layer imports
from src.core.text_to_coordinates_mapper import TextToCoordinatesMapper
from src.application.repositories.IModelCache import IModelCache

# Infrastructure layer imports
from src.infrastructure.cache.redis_model_cache import RedisModelCache

# For testing, we can also create an in-memory cache
class InMemoryModelCache(IModelCache):
    """Mock implementation for testing without Redis."""
    
    def __init__(self):
        self.storage = {}
    
    def save_model(self, tag: str, model_data: bytes) -> None:
        self.storage[f"{tag}:model"] = model_data
    
    def load_model(self, tag: str) -> bytes:
        return self.storage.get(f"{tag}:model")
    
    def save_metadata(self, tag: str, metadata: dict) -> None:
        self.storage[f"{tag}:metadata"] = metadata
        if "models" not in self.storage:
            self.storage["models"] = set()
        self.storage["models"].add(tag)
    
    def load_metadata(self, tag: str) -> dict:
        return self.storage.get(f"{tag}:metadata")
    
    def save_training_hashes(self, tag: str, hashes: List[str]) -> None:
        self.storage[f"{tag}:hashes"] = hashes
    
    def load_training_hashes(self, tag: str) -> List[str]:
        return self.storage.get(f"{tag}:hashes")
    
    def list_models(self) -> List[str]:
        return list(self.storage.get("models", []))
    
    def delete_model(self, tag: str) -> None:
        for key in [f"{tag}:model", f"{tag}:metadata", f"{tag}:hashes"]:
            self.storage.pop(key, None)
        if "models" in self.storage:
            self.storage["models"].discard(tag)


def demonstrate_clean_architecture():
    """Show how different cache implementations can be swapped."""
    
    print("=== Clean Architecture Demonstration ===\n")
    
    # Sample data
    training_sentences = [
        "Python is great for machine learning",
        "Machine learning transforms data into insights",
        "Deep learning uses neural networks",
        "AI is changing the world"
    ]
    
    test_sentences = [
        "Artificial intelligence and machine learning",
        "Neural networks power deep learning"
    ]
    
    # 1. Using In-Memory Cache (for testing)
    print("1. Using In-Memory Cache (for testing/development)")
    print("-" * 50)
    
    # Create mapper with in-memory cache
    memory_cache = InMemoryModelCache()
    mapper_memory = TextToCoordinatesMapper(cache=memory_cache)
    
    # Train and use
    mapper_memory.fit(training_sentences, tag="test_model")
    coords = mapper_memory.transform(test_sentences, tag="test_model")
    
    print(f"Transformed {len(test_sentences)} sentences:")
    for sentence, coord in zip(test_sentences, coords):
        print(f"  {sentence[:40]:<40} → ({coord[0]:6.3f}, {coord[1]:6.3f})")
    
    # 2. Using Redis Cache (for production)
    print("\n2. Using Redis Cache (for production)")
    print("-" * 50)
    
    # Set up Redis credentials
    os.environ['UPSTASH_REDIS_REST_URL'] = 'https://humorous-adder-10993.upstash.io'
    os.environ['UPSTASH_REDIS_REST_TOKEN'] = 'ASrxAAIjcDE2ZmZjZjU1OGY1ZmQ0MjBiYmEyMWZkODZlODZiYjFkOHAxMA'
    
    try:
        # Create mapper with Redis cache
        redis_cache = RedisModelCache()
        mapper_redis = TextToCoordinatesMapper(cache=redis_cache)
        
        # Train and use
        mapper_redis.fit(training_sentences, tag="prod_model_v1")
        coords = mapper_redis.transform(test_sentences, tag="prod_model_v1")
        
        print(f"Transformed {len(test_sentences)} sentences:")
        for sentence, coord in zip(test_sentences, coords):
            print(f"  {sentence[:40]:<40} → ({coord[0]:6.3f}, {coord[1]:6.3f})")
        
        # List models
        models = mapper_redis.list_models()
        print(f"\nModels in Redis: {models}")
        
    except Exception as e:
        print(f"Redis example failed (expected without real credentials): {e}")
    
    # 3. Demonstrate the benefit: Same business logic, different storage
    print("\n3. Key Benefits of Clean Architecture")
    print("-" * 50)
    print("✓ Business logic (TextToCoordinatesMapper) is independent of storage")
    print("✓ Can swap between in-memory and Redis without changing core code")
    print("✓ Easy to test with mock implementations")
    print("✓ Can add new storage backends (e.g., S3, DynamoDB) without touching core")
    print("✓ Dependencies flow inward: Infrastructure → Application → Core")


def demonstrate_stateless_operation():
    """Show how the mapper operates statelessly."""
    
    print("\n=== Stateless Operation Demonstration ===\n")
    
    # Create cache and mapper
    cache = InMemoryModelCache()
    
    # Train model with one mapper instance
    mapper1 = TextToCoordinatesMapper(cache=cache)
    training_data = ["Hello world", "Machine learning", "Python programming", "Data science"]
    mapper1.fit(training_data, tag="shared_model")
    print("Model trained with mapper instance 1")
    
    # Use model with a completely different mapper instance
    mapper2 = TextToCoordinatesMapper(cache=cache)  # New instance, same cache
    test_data = ["Hello AI", "Python rocks"]
    coords = mapper2.transform(test_data, tag="shared_model")
    
    print("\nUsing model with mapper instance 2:")
    for sentence, coord in zip(test_data, coords):
        print(f"  {sentence:<20} → ({coord[0]:6.3f}, {coord[1]:6.3f})")
    
    print("\n✓ No state stored in mapper instances")
    print("✓ All state managed through the cache interface")
    print("✓ Perfect for serverless/Lambda environments")


if __name__ == "__main__":
    demonstrate_clean_architecture()
    demonstrate_stateless_operation()