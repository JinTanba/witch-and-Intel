import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE, MDS
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple, Optional, Dict, Literal
import warnings
warnings.filterwarnings('ignore')


class SentenceMapper:
    """
    Maps sentences to 2D coordinates based on semantic similarity.
    
    Attributes:
        model_name: Name of the sentence transformer model
        reducer_type: Type of dimensionality reduction ('umap', 'tsne', 'mds')
        embeddings: Cached sentence embeddings
        coordinates: 2D coordinates of sentences
        sentences: List of processed sentences
    """
    
    def __init__(
        self, 
        model_name: str = 'all-mpnet-base-v2',
        reducer_type: Literal['umap', 'tsne', 'mds'] = 'umap',
        n_components: int = 2,
        random_state: int = 42
    ):
        """
        Initialize the SentenceMapper.
        
        Args:
            model_name: Sentence transformer model to use
            reducer_type: Dimensionality reduction method
            n_components: Number of dimensions (2 or 3 recommended)
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.reducer_type = reducer_type
        self.n_components = n_components
        self.random_state = random_state
        self.decide_n_neighbors = lambda x: max(int(x*0.1), 3)
        
        # Load model
        self.model = SentenceTransformer(model_name)
        
       
        
        # Cache for results
        self.embeddings = None
        self.coordinates = None
        self.sentences = None
        
    def _init_reducer(self, sentences_length):

        """Initialize the dimensionality reduction algorithm."""
        if self.reducer_type == 'umap':
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.decide_n_neighbors(sentences_length),
                min_dist=0.2,
                metric='cosine',
                random_state=self.random_state
            )
        elif self.reducer_type == 'tsne':
            self.reducer = TSNE(
                n_components=self.n_components,
                perplexity=4,  # Reduced for smaller datasets
                metric='cosine',
                random_state=self.random_state
            )
        elif self.reducer_type == 'mds':
            self.reducer = MDS(
                n_components=self.n_components,
                dissimilarity='precomputed',
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown reducer type: {self.reducer_type}")
    
    def fit_transform(self, sentences: List[str]) -> np.ndarray:
        """
        Map sentences to 2D coordinates.
        
        Args:
            sentences: List of sentences to map
            
        Returns:
            coordinates: Array of shape (n_sentences, n_components)
        """
        self.sentences = sentences

         # Initialize reducer
        self._init_reducer(self.sentences.length)
        
        # Generate embeddings
        print(f"Generating embeddings using {self.model_name}...")
        self.embeddings = self.model.encode(sentences, show_progress_bar=True)
        
        # Reduce dimensions
        print(f"Reducing dimensions using {self.reducer_type}...")
        if self.reducer_type == 'mds':
            # MDS needs distance matrix
            similarities = cosine_similarity(self.embeddings)
            distances = 1 - similarities
            self.coordinates = self.reducer.fit_transform(distances)
        else:
            self.coordinates = self.reducer.fit_transform(self.embeddings)
        
        return self.coordinates
    
    def visualize(
        self, 
        figsize: Tuple[int, int] = (12, 8),
        show_labels: bool = True,
        label_size: int = 8,
        max_label_length: int = 50,
        color_by_similarity_to: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize the sentence mappings.
        
        Args:
            figsize: Figure size
            show_labels: Whether to show sentence labels
            label_size: Font size for labels
            max_label_length: Maximum characters to show in labels
            color_by_similarity_to: Index of sentence to use for coloring by similarity
            save_path: Path to save the figure (optional)
        """
        if self.coordinates is None:
            raise ValueError("No coordinates to visualize. Run fit_transform first.")
        
        plt.figure(figsize=figsize)
        
        # Color points by similarity to a reference sentence if specified
        if color_by_similarity_to is not None:
            similarities = cosine_similarity(
                [self.embeddings[color_by_similarity_to]], 
                self.embeddings
            )[0]
            scatter = plt.scatter(
                self.coordinates[:, 0], 
                self.coordinates[:, 1],
                c=similarities,
                cmap='viridis',
                s=100,
                alpha=0.7
            )
            plt.colorbar(scatter, label=f'Similarity to: "{self.sentences[color_by_similarity_to][:30]}..."')
        else:
            plt.scatter(
                self.coordinates[:, 0], 
                self.coordinates[:, 1],
                alpha=0.7,
                s=100
            )
        
        # Add labels
        if show_labels:
            for i, sentence in enumerate(self.sentences):
                label = (sentence[:max_label_length] + "...") if len(sentence) > max_label_length else sentence
                plt.annotate(
                    label,
                    (self.coordinates[i, 0], self.coordinates[i, 1]),
                    fontsize=label_size,
                    alpha=0.8,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        plt.title(f"Sentence Semantic Map ({self.reducer_type.upper()})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_preservation(self) -> Dict[str, float]:
        """
        Analyze how well distances/similarities are preserved in 2D.
        
        Returns:
            Dictionary with preservation metrics
        """
        if self.embeddings is None or self.coordinates is None:
            raise ValueError("No data to analyze. Run fit_transform first.")
        
        # Calculate similarities and distances
        high_dim_sim = cosine_similarity(self.embeddings)
        low_dim_dist = squareform(pdist(self.coordinates, metric='euclidean'))
        
        # Normalize distances to similarities
        max_dist = np.max(low_dim_dist)
        low_dim_sim = 1 - (low_dim_dist / max_dist)
        
        # Get upper triangle values (avoid duplicates)
        mask = np.triu(np.ones_like(high_dim_sim, dtype=bool), k=1)
        high_dim_values = high_dim_sim[mask]
        low_dim_values = low_dim_sim[mask]
        
        # Calculate metrics
        pearson_corr, _ = pearsonr(high_dim_values, low_dim_values)
        spearman_corr, _ = spearmanr(high_dim_values, low_dim_values)
        
        # Calculate stress (distortion measure)
        stress = np.sqrt(np.sum((high_dim_values - low_dim_values)**2) / np.sum(high_dim_values**2))
        
        # Calculate neighborhood preservation
        k = min(5, len(self.sentences) - 1)
        preserved = 0
        for i in range(len(self.sentences)):
            high_neighbors = set(np.argsort(high_dim_sim[i])[-k-1:-1])
            low_neighbors = set(np.argsort(low_dim_sim[i])[-k-1:-1])
            preserved += len(high_neighbors & low_neighbors) / k
        neighborhood_preservation = preserved / len(self.sentences)
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'stress': stress,
            'neighborhood_preservation': neighborhood_preservation,
            'mean_absolute_error': np.mean(np.abs(high_dim_values - low_dim_values))
        }
    
    def plot_preservation(self, figsize: Tuple[int, int] = (10, 5)):
        """
        Plot the relationship between high-D and 2D similarities.
        
        Args:
            figsize: Figure size
        """
        if self.embeddings is None or self.coordinates is None:
            raise ValueError("No data to plot. Run fit_transform first.")
        
        # Calculate similarities
        high_dim_sim = cosine_similarity(self.embeddings)
        low_dim_dist = squareform(pdist(self.coordinates, metric='euclidean'))
        max_dist = np.max(low_dim_dist)
        low_dim_sim = 1 - (low_dim_dist / max_dist)
        
        # Get upper triangle values
        mask = np.triu(np.ones_like(high_dim_sim, dtype=bool), k=1)
        high_dim_values = high_dim_sim[mask]
        low_dim_values = low_dim_sim[mask]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot
        ax1.scatter(high_dim_values, low_dim_values, alpha=0.5, s=20)
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect preservation')
        ax1.set_xlabel('High-dimensional similarity')
        ax1.set_ylabel('2D similarity')
        ax1.set_title('Similarity Preservation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        errors = np.abs(high_dim_values - low_dim_values)
        ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
        ax2.set_xlabel('Absolute error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def find_similar(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Find sentences most similar to a query.
        
        Args:
            query: Query sentence
            top_k: Number of similar sentences to return
            
        Returns:
            List of (index, sentence, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Run fit_transform first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                idx,
                self.sentences[idx],
                similarities[idx]
            ))
        
        return results
    
    def compare_sentences(self, idx1: int, idx2: int) -> Dict[str, float]:
        """
        Compare two sentences in both high-D and 2D space.
        
        Args:
            idx1: Index of first sentence
            idx2: Index of second sentence
            
        Returns:
            Dictionary with comparison metrics
        """
        if self.embeddings is None or self.coordinates is None:
            raise ValueError("No data available. Run fit_transform first.")
        
        # High-dimensional similarity
        high_dim_sim = cosine_similarity(
            [self.embeddings[idx1]], 
            [self.embeddings[idx2]]
        )[0, 0]
        
        # 2D distance and normalized similarity
        dist_2d = np.linalg.norm(self.coordinates[idx1] - self.coordinates[idx2])
        max_dist = np.max(pdist(self.coordinates))
        sim_2d = 1 - (dist_2d / max_dist)
        
        return {
            'sentence1': self.sentences[idx1][:50] + '...',
            'sentence2': self.sentences[idx2][:50] + '...',
            'high_dim_similarity': high_dim_sim,
            '2d_similarity': sim_2d,
            '2d_distance': dist_2d,
            'similarity_difference': abs(high_dim_sim - sim_2d)
        }
    
    def export_data(self, filename: str):
        """
        Export coordinates and sentences to a file.
        
        Args:
            filename: Output filename (supports .csv, .json, .npz)
        """
        if self.coordinates is None:
            raise ValueError("No data to export. Run fit_transform first.")
        
        import pandas as pd
        import json
        
        if filename.endswith('.csv'):
            df = pd.DataFrame({
                'sentence': self.sentences,
                'x': self.coordinates[:, 0],
                'y': self.coordinates[:, 1]
            })
            df.to_csv(filename, index=False)
        
        elif filename.endswith('.json'):
            data = {
                'sentences': self.sentences,
                'coordinates': self.coordinates.tolist(),
                'model': self.model_name,
                'reducer': self.reducer_type
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif filename.endswith('.npz'):
            np.savez(
                filename,
                sentences=self.sentences,
                coordinates=self.coordinates,
                embeddings=self.embeddings
            )
        
        else:
            raise ValueError("Unsupported file format. Use .csv, .json, or .npz")

