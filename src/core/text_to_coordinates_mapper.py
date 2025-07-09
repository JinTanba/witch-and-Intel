"""
TextToCoordinatesMapper: A single-responsibility class that converts text to 2D coordinates using BERT+UMAP.
Follows Clean Architecture principles with proper dependency injection.
"""

from typing import List, Optional, Dict, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from datetime import datetime
import hashlib
import joblib
import io

from src.interfaces.ITtcm import ITextToCoordinatesMapper
from src.application.repositories.IModelCache import IModelCache


class TextToCoordinatesMapper(ITextToCoordinatesMapper):
    """
    Maps text sentences to 2D coordinates using BERT embeddings and UMAP dimensionality reduction.
    
    This class focuses solely on the text-to-coordinates transformation logic.
    Storage concerns are delegated to the IModelCache interface.
    """
    
    def __init__(
        self,
        cache: IModelCache,
        model_name: str = 'all-mpnet-base-v2',
        n_neighbors_ratio: float = 0.1,
        min_neighbors: int = 3,
        min_dist: float = 0.2,
        random_state: int = 42,
        refit_threshold: float = 0.2  # Re-fit if new data > 20% of training data
    ):
        """
        Initialize the mapper with dependencies and parameters.
        
        Args:
            cache: Model cache implementation (injected dependency)
            model_name: Name of the sentence transformer model (BERT-based)
            n_neighbors_ratio: Ratio of dataset size to use for UMAP n_neighbors
            min_neighbors: Minimum number of neighbors for UMAP
            min_dist: UMAP parameter controlling how tightly points are packed
            random_state: Random seed for reproducibility
            refit_threshold: Threshold for automatic re-fitting (0.2 = 20%)
        """
        self.cache = cache
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.n_neighbors_ratio = n_neighbors_ratio
        self.min_neighbors = min_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.refit_threshold = refit_threshold
    
    def _serialize_model(self, obj: any) -> bytes:
        """Serialize model object to bytes."""
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        return buffer.getvalue()
    
    def _deserialize_model(self, data: bytes) -> any:
        """Deserialize model object from bytes."""
        buffer = io.BytesIO(data)
        return joblib.load(buffer)
    
    def _hash_sentence(self, sentence: str) -> str:
        """Create a hash for a sentence to track uniqueness."""
        return hashlib.md5(sentence.encode()).hexdigest()
    
    def _save_model(self, tag: str, reducer: Optional[umap.UMAP], metadata: Dict, training_hashes: Set[str]) -> None:
        """Internal method to save model components."""
        # Save reducer if it exists
        if reducer is not None:
            self.cache.save_model(tag, self._serialize_model(reducer))
        
        # Save metadata
        self.cache.save_metadata(tag, metadata)
        
        # Save training hashes
        self.cache.save_training_hashes(tag, list(training_hashes))
    
    def _load_reducer(self, tag: str) -> Optional[umap.UMAP]:
        """Internal method to load reducer from cache."""
        model_data = self.cache.load_model(tag)
        if model_data:
            return self._deserialize_model(model_data)
        return None
    
    def fit(self, sentences: List[str], tag: str = "default") -> None:
        """
        Train the coordinate transformation model on the given sentences.
        
        Args:
            sentences: List of training sentences
            tag: Tag to identify this model version
        """
        if not sentences:
            raise ValueError("Cannot fit with empty sentence list")
        
        print(f"Fitting model with tag '{tag}' on {len(sentences)} sentences...")
        
        # Generate embeddings
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        n_samples = len(sentences)
        
        # Create and fit UMAP reducer
        reducer = None
        if n_samples >= 4:  # UMAP needs at least 4 samples
            n_neighbors = max(int(n_samples * self.n_neighbors_ratio), self.min_neighbors)
            n_neighbors = min(n_neighbors, n_samples - 1)
            
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=self.min_dist,
                metric='cosine',
                random_state=self.random_state
            )
            reducer.fit(embeddings)
        
        # Create metadata
        training_hashes = {self._hash_sentence(s) for s in sentences}
        metadata = {
            'tag': tag,
            'n_training_samples': n_samples,
            'model_name': self.model_name,
            'fit_timestamp': datetime.now().isoformat(),
            'umap_params': {
                'n_neighbors_ratio': self.n_neighbors_ratio,
                'min_neighbors': self.min_neighbors,
                'min_dist': self.min_dist,
                'random_state': self.random_state
            }
        }
        
        # Save model components
        self._save_model(tag, reducer, metadata, training_hashes)
        print(f"Model fitted and saved with tag '{tag}'")
    
    def save(self, tag: Optional[str] = None) -> None:
        """
        This method is now internal to fit() for better encapsulation.
        Kept for interface compatibility but does nothing.
        """
        pass
    
    def load(self, tag: str = "default") -> None:
        """
        This method is now internal to transform() for stateless operation.
        Kept for interface compatibility but only validates model existence.
        """
        metadata = self.cache.load_metadata(tag)
        if not metadata:
            raise ValueError(f"No model found with tag '{tag}'")
        print(f"Verified model with tag '{tag}' exists (trained on {metadata['n_training_samples']} samples)")
    
    def transform(self, sentences: List[str], tag: str = "default") -> np.ndarray:
        """
        Apply coordinate transformation to new sentences.
        
        Args:
            sentences: List of sentences to transform
            tag: Tag of the model to use
            
        Returns:
            Array of 2D coordinates
        """
        if not sentences:
            raise ValueError("Cannot transform empty sentence list")
        
        # Load model metadata
        metadata = self.cache.load_metadata(tag)
        if not metadata:
            raise ValueError(f"No model found with tag '{tag}'")
        
        # Generate embeddings
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        n_samples = len(sentences)
        
        # Handle special cases for small datasets
        if n_samples == 1:
            return np.array([[0.0, 0.0]])
        elif n_samples == 2:
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            distance = 1 - similarity
            return np.array([[-distance/2, 0], [distance/2, 0]])
        elif n_samples == 3:
            # Simple triangulation for 3 points
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(embeddings)
            coordinates = np.zeros((3, 2))
            coordinates[1, 0] = distances[0, 1]
            a, b, c = distances[0, 1], distances[0, 2], distances[1, 2]
            cos_angle = (a**2 + b**2 - c**2) / (2 * a * b) if a * b > 0 else 0
            sin_angle = np.sqrt(max(0, 1 - cos_angle**2))
            coordinates[2] = [b * cos_angle, b * sin_angle]
            return coordinates
        else:
            # Load reducer and transform
            reducer = self._load_reducer(tag)
            if reducer is None:
                # Fallback for models trained with < 4 samples
                from sklearn.metrics.pairwise import cosine_distances
                distances = cosine_distances(embeddings)
                coordinates = np.zeros((n_samples, 2))
                # Simple placement based on distances
                for i in range(min(n_samples, 2)):
                    if i == 1:
                        coordinates[i, 0] = distances[0, 1]
                for i in range(2, n_samples):
                    coordinates[i, 0] = distances[0, i]
                    coordinates[i, 1] = distances[1, i] if n_samples > 1 else 0
                return coordinates
            
            return reducer.transform(embeddings)
    
    def is_refit_needed(self, sentences: List[str], tag: str) -> bool:
        """
        Check if re-fitting is needed based on the amount of new data.
        
        Args:
            sentences: New sentences to check
            tag: Tag of the current model
            
        Returns:
            True if re-fitting is recommended
        """
        training_hashes = self.cache.load_training_hashes(tag)
        if not training_hashes:
            return True
        
        training_hashes_set = set(training_hashes)
        
        # Count new unique sentences
        new_hashes = {self._hash_sentence(s) for s in sentences}
        n_new = len(new_hashes - training_hashes_set)
        n_training = len(training_hashes_set)
        
        # Check if new data exceeds threshold
        return n_new / n_training > self.refit_threshold if n_training > 0 else True
    
    def fit_transform(self, sentences: List[str], tag: Optional[str] = None) -> np.ndarray:
        """
        Fit the model and transform sentences in one step.
        
        Args:
            sentences: List of sentences to fit and transform
            tag: Tag for the model
            
        Returns:
            Array of 2D coordinates
        """
        if tag is None:
            tag = "default"
        
        # Check if we should refit
        try:
            if not self.is_refit_needed(sentences, tag):
                print(f"Using existing model with tag '{tag}'")
                return self.transform(sentences, tag)
        except:
            # Model doesn't exist, fit it
            pass
        
        # Fit and transform
        self.fit(sentences, tag)
        return self.transform(sentences, tag)
    
    def list_models(self) -> List[Dict[str, any]]:
        """
        List all available saved models.
        
        Returns:
            List of model metadata
        """
        models = []
        for tag in self.cache.list_models():
            metadata = self.cache.load_metadata(tag)
            if metadata:
                models.append({
                    'tag': tag,
                    'n_samples': metadata['n_training_samples'],
                    'timestamp': metadata['fit_timestamp']
                })
        return models
    
    def delete_model(self, tag: str) -> None:
        """
        Delete a saved model.
        
        Args:
            tag: Tag of the model to delete
        """
        self.cache.delete_model(tag)
        print(f"Deleted model with tag '{tag}'")