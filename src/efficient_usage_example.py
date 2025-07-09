"""
Example demonstrating efficient usage of TextToCoordinatesMapper with persistence.
"""

import time
import numpy as np
from src.core.text_to_coordinates_mapper import TextToCoordinatesMapper


def train_model_once():
    """Train a model once on a large dataset and save it."""
    print("=== Training Model Once ===")
    
    # Create mapper
    mapper = TextToCoordinatesMapper(model_name='all-MiniLM-L6-v2')
    
    # Generate large training dataset
    topics = ["technology", "science", "sports", "cooking", "travel", "music", "art", "history"]
    templates = [
        "{} is fascinating to study",
        "I love learning about {}",
        "The world of {} is complex",
        "{} has evolved significantly",
        "Understanding {} requires dedication",
        "Experts in {} work hard",
        "{} impacts our daily lives",
        "The future of {} looks bright",
        "Ancient {} tells us stories",
        "{} connects people globally",
        "Modern {} is innovative",
        "{} requires creativity",
        "The basics of {} are important",
        "Advanced {} is challenging",
        "{} has many applications"
    ]
    
    training_sentences = []
    for topic in topics:
        for template in templates:
            training_sentences.append(template.format(topic))
    
    print(f"Training on {len(training_sentences)} sentences...")
    start_time = time.time()
    
    # Fit and save with tag
    mapper.fit(training_sentences, tag="production_v1")
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Model saved with tag 'production_v1'")
    
    return training_sentences


def efficient_inference():
    """Load pre-trained model and perform fast inference."""
    print("\n=== Efficient Inference ===")
    
    # Create new mapper and load pre-trained model
    mapper = TextToCoordinatesMapper()
    
    print("Loading pre-trained model...")
    start_time = time.time()
    mapper.load(tag="production_v1")
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.3f} seconds")
    
    # New sentences to transform
    new_sentences = [
        "Artificial intelligence is transforming technology",
        "Machine learning algorithms are powerful",
        "Cooking Italian pasta is enjoyable",
        "Basketball is an exciting sport",
        "Classical music is beautiful",
        "Traveling to new places is adventurous"
    ]
    
    print(f"\nTransforming {len(new_sentences)} new sentences...")
    start_time = time.time()
    coordinates = mapper.transform(new_sentences)
    transform_time = time.time() - start_time
    
    print(f"Transform completed in {transform_time:.3f} seconds")
    print("\nResults:")
    for sentence, coord in zip(new_sentences, coordinates):
        print(f"  {sentence[:50]:50} -> ({coord[0]:6.3f}, {coord[1]:6.3f})")
    
    return coordinates


def demonstrate_auto_refit():
    """Demonstrate automatic re-fitting when encountering lots of new data."""
    print("\n=== Automatic Re-fitting Demo ===")
    
    mapper = TextToCoordinatesMapper(refit_threshold=0.25)  # Re-fit if new data > 25%
    
    # Load existing model
    mapper.load(tag="production_v1")
    
    # Create sentences with some overlap and some new
    mixed_sentences = [
        # Some overlap with training
        "Technology is fascinating to study",
        "Science has evolved significantly",
        # Many new sentences
        "Quantum computing revolutionizes calculations",
        "Blockchain enables decentralized systems",
        "Virtual reality creates immersive experiences",
        "Robotics automates complex tasks",
        "Biotechnology advances medicine",
        "Nanotechnology works at molecular scale",
        "Space exploration discovers new worlds",
        "Renewable energy saves the planet",
        "Cybersecurity protects digital assets",
        "5G networks enable faster communication",
        "Gene editing cures diseases",
        "Autonomous vehicles transform transportation",
        "Smart cities optimize urban living",
        "Augmented reality enhances perception"
    ]
    
    print(f"Processing {len(mixed_sentences)} sentences...")
    print(f"Checking if re-fit needed (threshold: {mapper.refit_threshold*100}%)...")
    
    if mapper.is_refit_needed(mixed_sentences):
        print("Re-fit threshold exceeded! Model will be updated.")
    else:
        print("Within threshold, using existing model.")
    
    # This will automatically re-fit if needed
    coordinates = mapper.fit_transform(mixed_sentences, tag="production_v2")
    
    print(f"\nCurrent model tag: {mapper.current_tag}")
    print("First 5 coordinates:")
    for i in range(5):
        print(f"  {mixed_sentences[i][:50]:50} -> ({coordinates[i, 0]:6.3f}, {coordinates[i, 1]:6.3f})")


def demonstrate_tag_management():
    """Demonstrate managing multiple model versions with tags."""
    print("\n=== Tag Management Demo ===")
    
    mapper = TextToCoordinatesMapper()
    
    # List all available models
    print("Available models:")
    for model in mapper.list_models():
        print(f"  - Tag: {model['tag']}")
        print(f"    Samples: {model['n_samples']}")
        print(f"    Created: {model['timestamp']}")
    
    # Demonstrate switching between models
    test_sentence = ["Machine learning is powerful"]
    
    print(f"\nTesting sentence: '{test_sentence[0]}'")
    print("Coordinates with different model versions:")
    
    for model in mapper.list_models()[:2]:  # Test first 2 models
        mapper.load(model['tag'])
        coords = mapper.transform(test_sentence)
        print(f"  {model['tag']:15} -> ({coords[0, 0]:6.3f}, {coords[0, 1]:6.3f})")


def main():
    """Run all demonstrations."""
    # Train model once
    training_sentences = train_model_once()
    
    # Show efficient inference
    efficient_inference()
    
    # Demonstrate automatic re-fitting
    demonstrate_auto_refit()
    
    # Show tag management
    demonstrate_tag_management()
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    mapper1 = TextToCoordinatesMapper()
    mapper2 = TextToCoordinatesMapper()
    
    test_sentences = ["Test sentence " + str(i) for i in range(10)]
    
    # Method 1: Load and transform (efficient)
    start_time = time.time()
    mapper1.load("production_v1")
    coords1 = mapper1.transform(test_sentences)
    method1_time = time.time() - start_time
    
    # Method 2: Fit-transform every time (inefficient)
    start_time = time.time()
    coords2 = mapper2.fit_transform(training_sentences + test_sentences)
    method2_time = time.time() - start_time
    
    print(f"Load + Transform: {method1_time:.3f} seconds")
    print(f"Fit-Transform:    {method2_time:.3f} seconds")
    print(f"Speedup:          {method2_time/method1_time:.1f}x faster")


if __name__ == "__main__":
    main()