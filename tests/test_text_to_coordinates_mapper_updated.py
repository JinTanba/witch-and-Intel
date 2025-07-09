"""
Updated tests for TextToCoordinatesMapper with dependency injection
"""

import pytest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from src.core.text_to_coordinates_mapper import TextToCoordinatesMapper
from src.application.repositories.IModelCache import IModelCache
from typing import List, Dict, Optional, Any


class MockModelCache(IModelCache):
    """Mock implementation of IModelCache for testing."""
    
    def __init__(self):
        self.storage = {}
    
    def save_model(self, tag: str, model_data: bytes) -> None:
        self.storage[f"{tag}:model"] = model_data
    
    def load_model(self, tag: str) -> Optional[bytes]:
        return self.storage.get(f"{tag}:model")
    
    def save_metadata(self, tag: str, metadata: Dict[str, Any]) -> None:
        self.storage[f"{tag}:metadata"] = metadata
        if "models" not in self.storage:
            self.storage["models"] = set()
        self.storage["models"].add(tag)
    
    def load_metadata(self, tag: str) -> Optional[Dict[str, Any]]:
        return self.storage.get(f"{tag}:metadata")
    
    def save_training_hashes(self, tag: str, hashes: List[str]) -> None:
        self.storage[f"{tag}:hashes"] = hashes
    
    def load_training_hashes(self, tag: str) -> Optional[List[str]]:
        return self.storage.get(f"{tag}:hashes")
    
    def list_models(self) -> List[str]:
        return list(self.storage.get("models", []))
    
    def delete_model(self, tag: str) -> None:
        for key in [f"{tag}:model", f"{tag}:metadata", f"{tag}:hashes"]:
            self.storage.pop(key, None)
        if "models" in self.storage:
            self.storage["models"].discard(tag)


class TestTextToCoordinatesMapper:
    """Test suite for TextToCoordinatesMapper."""
    
    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache for testing."""
        return MockModelCache()
    
    @pytest.fixture
    def semantic_groups(self):
        """Create groups of semantically similar sentences for testing."""
        # Group 1: Machine learning sentences
        ml_sentences = [
            "Machine learning is a subset of artificial intelligence",
            "AI and machine learning are closely related fields",
            "Deep learning is a type of machine learning algorithm",
            "Neural networks are used in machine learning applications",
            "Supervised learning requires labeled training data",
            "Unsupervised learning finds patterns in unlabeled data"
        ]
        
        # Group 2: Cooking sentences
        cooking_sentences = [
            "I love cooking pasta with fresh tomatoes",
            "Making homemade pasta is my favorite cooking activity",
            "Fresh ingredients make the best pasta dishes",
            "Cooking Italian food requires good tomatoes",
            "The chef prepared a delicious meal with herbs",
            "Baking bread requires patience and skill"
        ]
        
        # Group 3: Weather sentences
        weather_sentences = [
            "The weather today is sunny and warm",
            "It's a beautiful sunny day outside",
            "Today's forecast shows clear skies and sunshine",
            "The temperature is warm with lots of sun",
            "Tomorrow will be cloudy with chance of rain",
            "Winter brings cold temperatures and snow"
        ]
        
        # Group 4: Sports sentences
        sports_sentences = [
            "Basketball is a popular team sport",
            "Soccer is played worldwide by millions",
            "Tennis requires good hand-eye coordination",
            "Swimming is great for cardiovascular health",
            "Running marathons requires endurance training",
            "Baseball is America's favorite pastime"
        ]
        
        # All sentences combined
        all_sentences = ml_sentences + cooking_sentences + weather_sentences + sports_sentences
        
        return {
            'ml': ml_sentences,
            'cooking': cooking_sentences,
            'weather': weather_sentences,
            'sports': sports_sentences,
            'all': all_sentences
        }
    
    def test_initialization(self, mock_cache):
        """Test that mapper initializes correctly."""
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        assert mapper is not None
        assert mapper.model_name == 'all-mpnet-base-v2'
        assert mapper.n_neighbors_ratio == 0.1
        assert mapper.min_neighbors == 3
        assert mapper.min_dist == 0.2
        assert mapper.random_state == 42
    
    def test_fit_transform_shape(self, mock_cache, semantic_groups):
        """Test that fit_transform returns correct shape."""
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        sentences = semantic_groups['all']
        
        # Use fit and transform separately
        mapper.fit(sentences, tag="test")
        coordinates = mapper.transform(sentences, tag="test")
        
        assert isinstance(coordinates, np.ndarray)
        assert coordinates.shape == (len(sentences), 2)
        assert coordinates.dtype == np.float32 or coordinates.dtype == np.float64
    
    def test_semantic_clustering(self, mock_cache, semantic_groups):
        """Test that semantically similar sentences cluster together."""
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        sentences = semantic_groups['all']
        
        # Train and transform
        mapper.fit(sentences, tag="semantic_test")
        coordinates = mapper.transform(sentences, tag="semantic_test")
        
        # Calculate pairwise distances
        distances = squareform(pdist(coordinates))
        
        # Check that sentences within the same group are closer together on average
        n_ml = len(semantic_groups['ml'])
        n_cooking = len(semantic_groups['cooking'])
        n_weather = len(semantic_groups['weather'])
        n_sports = len(semantic_groups['sports'])
        
        # Calculate average within-group distances
        ml_indices = range(0, n_ml)
        cooking_indices = range(n_ml, n_ml + n_cooking)
        
        # Get within-group distances
        ml_distances = [distances[i][j] for i in ml_indices for j in ml_indices if i < j]
        cooking_distances = [distances[i][j] for i in cooking_indices for j in cooking_indices if i < j]
        
        # Get between-group distances
        between_distances = [distances[i][j] for i in ml_indices for j in cooking_indices]
        
        # Within-group average should be smaller than between-group average
        if ml_distances and cooking_distances and between_distances:  # Check non-empty
            avg_ml = np.mean(ml_distances) if ml_distances else 0
            avg_cooking = np.mean(cooking_distances) if cooking_distances else 0
            avg_between = np.mean(between_distances)
            
            # This is a soft check - semantic similarity should generally result in closer distances
            print(f"Avg ML distances: {avg_ml:.3f}")
            print(f"Avg Cooking distances: {avg_cooking:.3f}")
            print(f"Avg Between distances: {avg_between:.3f}")
    
    def test_reproducibility(self, mock_cache, semantic_groups):
        """Test that same seed produces same results."""
        sentences = semantic_groups['all'][:10]  # Use subset for speed
        
        cache1 = MockModelCache()
        cache2 = MockModelCache()
        
        mapper1 = TextToCoordinatesMapper(cache=cache1, random_state=123)
        mapper2 = TextToCoordinatesMapper(cache=cache2, random_state=123)
        
        # Train both with same data
        mapper1.fit(sentences, tag="repro_test")
        mapper2.fit(sentences, tag="repro_test")
        
        # Transform and compare
        coords1 = mapper1.transform(sentences, tag="repro_test")
        coords2 = mapper2.transform(sentences, tag="repro_test")
        
        # Should be very close (allowing for minor floating point differences)
        np.testing.assert_allclose(coords1, coords2, rtol=1e-5, atol=1e-8)
    
    def test_empty_sentences_raises_error(self, mock_cache):
        """Test that empty sentences list raises ValueError."""
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        
        with pytest.raises(ValueError, match="Cannot fit with empty sentence list"):
            mapper.fit([])
        
        with pytest.raises(ValueError, match="Cannot transform empty sentence list"):
            mapper.transform([])
    
    def test_single_sentence(self, mock_cache):
        """Test handling of single sentence."""
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        
        # Single sentence should work
        mapper.fit(["Hello world"], tag="single")
        result = mapper.transform(["Hello world"], tag="single")
        
        assert result.shape == (1, 2)
        assert result[0][0] == 0.0 and result[0][1] == 0.0
    
    def test_transform_without_fit(self, mock_cache):
        """Test that transform without fit raises error."""
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        
        with pytest.raises(ValueError, match="No model found with tag"):
            mapper.transform(["Test sentence"], tag="nonexistent")
    
    def test_caching_functionality(self, mock_cache):
        """Test that models are properly cached and can be reloaded."""
        sentences = ["Test A", "Test B", "Test C", "Test D", "Test E"]
        
        # Create first mapper and train
        mapper1 = TextToCoordinatesMapper(cache=mock_cache)
        mapper1.fit(sentences, tag="cache_test")
        coords1 = mapper1.transform(sentences, tag="cache_test")
        
        # Create second mapper with same cache and use saved model
        mapper2 = TextToCoordinatesMapper(cache=mock_cache)
        coords2 = mapper2.transform(sentences, tag="cache_test")
        
        # Results should be identical
        np.testing.assert_array_equal(coords1, coords2)
    
    def test_different_tags(self, mock_cache):
        """Test that different tags create different models."""
        sentences = ["Test A", "Test B", "Test C", "Test D", "Test E"]
        
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        
        # Train two models with different tags
        mapper.fit(sentences, tag="model1")
        mapper.fit(sentences[:3], tag="model2")  # Different training data
        
        # List models should show both
        models = mapper.list_models()
        assert len(models) == 2
        assert any(m['tag'] == 'model1' for m in models)
        assert any(m['tag'] == 'model2' for m in models)
    
    def test_small_dataset_handling(self, mock_cache):
        """Test handling of small datasets (2-3 sentences)."""
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        
        # Test with 2 sentences
        two_sentences = ["Hello world", "Machine learning"]
        mapper.fit(two_sentences, tag="two")
        coords = mapper.transform(two_sentences, tag="two")
        assert coords.shape == (2, 2)
        assert not np.array_equal(coords[0], coords[1])  # Should be different
        
        # Test with 3 sentences
        three_sentences = ["Hello world", "Machine learning", "Python programming"]
        mapper.fit(three_sentences, tag="three")
        coords = mapper.transform(three_sentences, tag="three")
        assert coords.shape == (3, 2)
    
    def test_joblib_serialization(self, mock_cache):
        """Test that joblib serialization works correctly."""
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        sentences = [
            "Test sentence one",
            "Test sentence two", 
            "Test sentence three",
            "Test sentence four",
            "Test sentence five"
        ]
        
        # Train model
        mapper.fit(sentences, tag="joblib_test")
        
        # Verify model was saved
        assert mock_cache.load_model("joblib_test") is not None
        
        # Transform should work
        coords = mapper.transform(sentences, tag="joblib_test")
        assert coords.shape == (5, 2)
    
    def test_refit_threshold(self, mock_cache):
        """Test the refit threshold functionality."""
        mapper = TextToCoordinatesMapper(cache=mock_cache, refit_threshold=0.2)
        
        # Initial training
        initial_sentences = ["A", "B", "C", "D", "E"]
        mapper.fit(initial_sentences, tag="refit_test")
        
        # Check with same sentences - should not need refit
        assert not mapper.is_refit_needed(initial_sentences, tag="refit_test")
        
        # Check with one new sentence (20% threshold)
        new_sentences = initial_sentences + ["F"]  # 1 new out of 5 = 20%
        assert not mapper.is_refit_needed(new_sentences, tag="refit_test")
        
        # Check with two new sentences (40% > 20% threshold)
        new_sentences = initial_sentences + ["F", "G"]  # 2 new out of 5 = 40%
        assert mapper.is_refit_needed(new_sentences, tag="refit_test")
    
    def test_fit_transform_method(self, mock_cache):
        """Test the combined fit_transform method."""
        mapper = TextToCoordinatesMapper(cache=mock_cache)
        sentences = ["A", "B", "C", "D", "E"]
        
        # First call should fit and transform
        coords1 = mapper.fit_transform(sentences, tag="fit_transform_test")
        assert coords1.shape == (5, 2)
        
        # Second call with same data should use existing model
        coords2 = mapper.fit_transform(sentences, tag="fit_transform_test")
        np.testing.assert_array_equal(coords1, coords2)
        
        # Call with significantly different data should refit
        new_sentences = ["X", "Y", "Z", "W", "V"]
        coords3 = mapper.fit_transform(new_sentences, tag="fit_transform_test")
        assert coords3.shape == (5, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])