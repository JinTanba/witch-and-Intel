import numpy as np
from src.playground import SentenceMapper
import matplotlib.pyplot as plt

def test_sentence_coordinates():
    """Test SentenceMapper with similar and dissimilar sentences."""
    
    # Create groups of sentences
    # Group 1: Similar sentences about machine learning
    ml_sentences = [
        "Machine learning is a subset of artificial intelligence",
        "AI and machine learning are closely related fields",
        "Deep learning is a type of machine learning algorithm",
        "Neural networks are used in machine learning applications"
    ]
    
    # Group 2: Similar sentences about cooking
    cooking_sentences = [
        "I love cooking pasta with fresh tomatoes",
        "Making homemade pasta is my favorite cooking activity",
        "Fresh ingredients make the best pasta dishes",
        "Cooking Italian food requires good tomatoes"
    ]
    
    # Group 3: Similar sentences about weather
    weather_sentences = [
        "The weather today is sunny and warm",
        "It's a beautiful sunny day outside",
        "Today's forecast shows clear skies and sunshine",
        "The temperature is warm with lots of sun"
    ]
    
    # Group 4: Dissimilar/random sentences
    random_sentences = [
        "Quantum physics explains particle behavior",
        "The stock market closed higher today",
        "My cat likes to sleep on the windowsill",
        "JavaScript is a programming language"
    ]
    
    # Combine all sentences
    all_sentences = ml_sentences + cooking_sentences + weather_sentences + random_sentences
    
    # Create labels for visualization
    labels = (
        ['ML'] * len(ml_sentences) + 
        ['Cooking'] * len(cooking_sentences) + 
        ['Weather'] * len(weather_sentences) + 
        ['Random'] * len(random_sentences)
    )
    
    # Test with different reducers
    reducers = ['umap', 'tsne', 'mds']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, reducer_type in enumerate(reducers):
        print(f"\n{'='*50}")
        print(f"Testing with {reducer_type.upper()}")
        print('='*50)
        
        # Initialize mapper
        mapper = SentenceMapper(reducer_type=reducer_type)
        
        # Get coordinates
        coordinates = mapper.fit_transform(all_sentences)
        
        # Print coordinates for each group
        print(f"\nCoordinates using {reducer_type}:")
        start_idx = 0
        for group_name, group_sentences in [
            ("Machine Learning", ml_sentences),
            ("Cooking", cooking_sentences),
            ("Weather", weather_sentences),
            ("Random", random_sentences)
        ]:
            print(f"\n{group_name} sentences:")
            for i, sentence in enumerate(group_sentences):
                coord_idx = start_idx + i
                print(f"  '{sentence[:40]}...' -> ({coordinates[coord_idx, 0]:.3f}, {coordinates[coord_idx, 1]:.3f})")
            start_idx += len(group_sentences)
        
        # Calculate and print preservation metrics
        metrics = mapper.analyze_preservation()
        print(f"\nPreservation metrics for {reducer_type}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Plot on subplot
        ax = axes[idx]
        colors = {'ML': 'red', 'Cooking': 'green', 'Weather': 'blue', 'Random': 'purple'}
        
        for i, (coord, label) in enumerate(zip(coordinates, labels)):
            ax.scatter(coord[0], coord[1], c=colors[label], s=100, alpha=0.7, edgecolors='black')
            # Add sentence number for reference
            ax.annotate(str(i), (coord[0], coord[1]), fontsize=8, ha='center', va='center')
        
        ax.set_title(f'{reducer_type.upper()} Reduction')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[label], label=label) for label in colors.keys()]
        ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig('sentence_coordinates_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test similarity calculations
    print("\n" + "="*50)
    print("Testing similarity calculations")
    print("="*50)
    
    # Use UMAP mapper for similarity tests
    mapper = SentenceMapper(reducer_type='umap')
    mapper.fit_transform(all_sentences)
    
    # Compare similar sentences (within same group)
    print("\nComparing similar sentences (same topic):")
    comparison1 = mapper.compare_sentences(0, 1)  # Two ML sentences
    for key, value in comparison1.items():
        print(f"  {key}: {value}")
    
    # Compare dissimilar sentences (different groups)
    print("\nComparing dissimilar sentences (different topics):")
    comparison2 = mapper.compare_sentences(0, 4)  # ML vs Cooking
    for key, value in comparison2.items():
        print(f"  {key}: {value}")
    
    # Find similar sentences to a query
    print("\nFinding similar sentences to 'artificial intelligence and deep learning':")
    similar = mapper.find_similar("artificial intelligence and deep learning", top_k=5)
    for idx, sentence, score in similar:
        print(f"  [{idx}] {sentence[:50]}... (similarity: {score:.3f})")
    
    # Create a detailed visualization with labels
    plt.figure(figsize=(12, 8))
    mapper.visualize(
        show_labels=True,
        label_size=6,
        max_label_length=30,
        color_by_similarity_to=0,  # Color by similarity to first ML sentence
        save_path='sentence_map_detailed.png'
    )
    
    # Plot preservation analysis
    mapper.plot_preservation()
    
    # Export data for further analysis
    mapper.export_data('sentence_coordinates.csv')
    print("\nData exported to sentence_coordinates.csv")

if __name__ == "__main__":
    test_sentence_coordinates()