"""
Main entry point for Lambda text processing
Demonstrates proper dependency injection and wiring
"""

import asyncio
import json
from src.infrastructure.lambda.lambda_factory import create_production_handler


def main():
    """
    Example of using the Lambda handler locally
    """
    # Create handler with production dependencies
    handler = create_production_handler()
    
    # Example request
    test_event = {
        "sentences": [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks",
            "Python is a popular programming language",
            "Data science involves statistics and programming",
            "The weather is nice today",
            "I enjoy walking in the park"
        ],
        "tag": "example"
    }
    
    # Run the handler
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(handler.handle_fit_transform(test_event))
    
    # Print result
    print(json.dumps(result, indent=2))
    
    if result["statusCode"] == 200:
        coordinates = result["body"]["coordinates"]
        print(f"\nProcessed {len(coordinates)} sentences")
        print("\nCoordinates:")
        for i, (sentence, coord) in enumerate(zip(test_event["sentences"], coordinates)):
            print(f"{i+1}. '{sentence[:50]}...' -> [{coord[0]:.4f}, {coord[1]:.4f}]")


if __name__ == "__main__":
    main()