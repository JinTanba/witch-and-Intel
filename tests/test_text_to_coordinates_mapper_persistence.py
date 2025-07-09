"""
Tests for TextToCoordinatesMapper persistence and efficient operations.
"""

import pytest
import numpy as np
import os
import shutil
import tempfile
from src.core.text_to_coordinates_mapper import TextToCoordinatesMapper


class TestTextToCoordinatesMapperPersistence:
    """Test suite for persistence and efficiency features."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def training_sentences(self):
        """Sample training sentences."""
        return [
            "Machine learning is fascinating",
            "Deep learning uses neural networks",
            "AI is transforming technology",
            "Python is great for data science",
            "Natural language processing is complex",
            "Computer vision recognizes images",
            "Reinforcement learning trains agents",
            "Data science requires statistics",
            "Algorithms solve computational problems",
            "Big data needs efficient processing"
        ]
    
    def test_fit_save_load_transform(self, temp_model_dir, training_sentences):
        """Test the complete fit-save-load-transform workflow."""
        # Create mapper with temp directory
        mapper1 = TextToCoordinatesMapper(model_dir=temp_model_dir)
        
        # Fit the model
        mapper1.fit(training_sentences, tag="test_v1")
        
        # Transform using the fitted model
        coords1 = mapper1.transform(["Machine learning is great"])
        
        # Create new mapper instance
        mapper2 = TextToCoordinatesMapper(model_dir=temp_model_dir)
        
        # Load the saved model
        mapper2.load(tag="test_v1")
        
        # Transform using the loaded model
        coords2 = mapper2.transform(["Machine learning is great"])
        
        # Results should be identical
        np.testing.assert_allclose(coords1, coords2)
    
    def test_multiple_tags(self, temp_model_dir, training_sentences):
        """Test managing multiple model versions with different tags."""
        mapper = TextToCoordinatesMapper(model_dir=temp_model_dir)
        
        # Create models with different tags
        mapper.fit(training_sentences[:5], tag="small_v1")
        mapper.fit(training_sentences, tag="full_v1")
        
        # List models
        models = mapper.list_models()
        assert len(models) == 2
        tags = {m['tag'] for m in models}
        assert "small_v1" in tags
        assert "full_v1" in tags
        
        # Load and use different models with multiple test sentences
        test_sentences = ["Machine learning rocks", "Python programming is fun", "Data science rules"]
        
        mapper.load("small_v1")
        coords1 = mapper.transform(test_sentences)
        
        mapper.load("full_v1")
        coords2 = mapper.transform(test_sentences)
        
        # At least some coordinates should be different (different training data)
        # Check that not all coordinates are exactly the same
        assert not np.allclose(coords1, coords2, rtol=1e-5, atol=1e-8)
    
    def test_transform_without_fit_raises_error(self, temp_model_dir):
        """Test that transform without fit/load raises error."""
        mapper = TextToCoordinatesMapper(model_dir=temp_model_dir)
        
        with pytest.raises(ValueError, match="No model loaded"):
            mapper.transform(["Test sentence"])
    
    def test_refit_threshold(self, temp_model_dir, training_sentences):
        """Test automatic re-fitting based on new data threshold."""
        mapper = TextToCoordinatesMapper(
            model_dir=temp_model_dir,
            refit_threshold=0.3  # Re-fit if new data > 30%
        )
        
        # Initial fit
        mapper.fit(training_sentences, tag="adaptive")
        
        # Small amount of new data (< 30%) - should not refit
        new_sentences = ["New sentence 1", "New sentence 2"]
        assert not mapper.is_refit_needed(new_sentences)
        
        # Large amount of new data (> 30%) - should refit
        many_new = [f"New sentence {i}" for i in range(5)]
        assert mapper.is_refit_needed(many_new)
    
    def test_efficiency_comparison(self, temp_model_dir):
        """Test model loading is faster than fitting."""
        import time
        
        # Create dataset
        training_sentences = [f"Training sentence {i} about topic {i%10}" for i in range(50)]
        test_sentences = [f"Test sentence {i}" for i in range(5)]
        
        mapper = TextToCoordinatesMapper(model_dir=temp_model_dir)
        
        # Time fitting
        start_time = time.time()
        mapper.fit(training_sentences, tag="timing_test")
        fit_time = time.time() - start_time
        
        # Time loading (in new instance)
        mapper2 = TextToCoordinatesMapper(model_dir=temp_model_dir)
        start_time = time.time()
        mapper2.load(tag="timing_test")
        load_time = time.time() - start_time
        
        # Time transform
        start_time = time.time()
        coords = mapper2.transform(test_sentences)
        transform_time = time.time() - start_time
        
        # Loading should be much faster than fitting
        assert load_time < fit_time
        
        print(f"\nTiming comparison:")
        print(f"  Fit time:       {fit_time:.3f}s")
        print(f"  Load time:      {load_time:.3f}s")
        print(f"  Transform time: {transform_time:.3f}s")
        print(f"  Load speedup:   {fit_time/load_time:.1f}x faster than fit")
    
    def test_model_deletion(self, temp_model_dir, training_sentences):
        """Test model deletion functionality."""
        mapper = TextToCoordinatesMapper(model_dir=temp_model_dir)
        
        # Create and verify model exists
        mapper.fit(training_sentences, tag="to_delete")
        models = mapper.list_models()
        assert any(m['tag'] == "to_delete" for m in models)
        
        # Delete model
        mapper.delete_model("to_delete")
        
        # Verify model is gone
        models = mapper.list_models()
        assert not any(m['tag'] == "to_delete" for m in models)
        
        # Loading deleted model should fail
        with pytest.raises(ValueError, match="No model found"):
            mapper.load("to_delete")
    
    def test_edge_cases_with_persistence(self, temp_model_dir):
        """Test edge cases with model persistence."""
        mapper = TextToCoordinatesMapper(model_dir=temp_model_dir)
        
        # Fit with minimal data
        mapper.fit(["One", "Two", "Three"], tag="minimal")
        
        # Load and transform
        mapper2 = TextToCoordinatesMapper(model_dir=temp_model_dir)
        mapper2.load("minimal")
        coords = mapper2.transform(["Test"])
        assert coords.shape == (1, 2)
    
    def test_metadata_persistence(self, temp_model_dir, training_sentences):
        """Test that metadata is correctly saved and loaded."""
        mapper = TextToCoordinatesMapper(
            model_name='all-MiniLM-L6-v2',
            n_neighbors_ratio=0.15,
            min_dist=0.3,
            model_dir=temp_model_dir
        )
        
        mapper.fit(training_sentences, tag="metadata_test")
        
        # Load in new mapper
        mapper2 = TextToCoordinatesMapper(model_dir=temp_model_dir)
        mapper2.load("metadata_test")
        
        # Check metadata
        assert mapper2.metadata['n_training_samples'] == len(training_sentences)
        assert mapper2.metadata['tag'] == "metadata_test"
        assert mapper2.metadata['model_name'] == 'all-MiniLM-L6-v2'
        assert mapper2.metadata['umap_params']['n_neighbors_ratio'] == 0.15
        assert mapper2.metadata['umap_params']['min_dist'] == 0.3