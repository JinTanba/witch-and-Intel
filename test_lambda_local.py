"""
Test script to run Lambda function locally
"""

import json
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.infrastructure.lambda.lambda_handler import lambda_handler


def test_lambda_handler():
    """Test the Lambda handler with sample data"""
    
    print("Testing Lambda handler locally...")
    print("-" * 60)
    
    # Test 1: Valid request
    print("\nTest 1: Valid request with multiple sentences")
    event = {
        "body": json.dumps({
            "sentences": [
                "Machine learning is a subset of artificial intelligence",
                "Deep learning uses neural networks with multiple layers",
                "Python is a popular programming language for data science",
                "Natural language processing helps computers understand human language",
                "The weather is nice and sunny today",
                "I enjoy walking in the park during autumn"
            ],
            "tag": "test_local"
        })
    }
    
    context = {}  # Mock Lambda context
    
    response = lambda_handler(event, context)
    
    print(f"Status Code: {response['statusCode']}")
    print(f"Headers: {response['headers']}")
    
    body = json.loads(response['body'])
    if response['statusCode'] == 200:
        print(f"Message: {body['message']}")
        print(f"Number of coordinates: {len(body['coordinates'])}")
        print("\nCoordinates:")
        for i, coord in enumerate(body['coordinates']):
            sentence = json.loads(event['body'])['sentences'][i]
            print(f"  {i+1}. '{sentence[:40]}...' -> [{coord[0]:.4f}, {coord[1]:.4f}]")
    else:
        print(f"Error: {body.get('error', 'Unknown error')}")
    
    # Test 2: Direct event format
    print("\n" + "-" * 60)
    print("\nTest 2: Direct event format (no API Gateway wrapper)")
    direct_event = {
        "sentences": ["This is a direct test"],
        "tag": "direct"
    }
    
    response = lambda_handler(direct_event, context)
    print(f"Status Code: {response['statusCode']}")
    
    body = json.loads(response['body'])
    if response['statusCode'] == 200:
        print(f"Coordinates: {body['coordinates']}")
    
    # Test 3: Error case
    print("\n" + "-" * 60)
    print("\nTest 3: Error case - missing sentences")
    error_event = {
        "body": json.dumps({
            "tag": "test"
        })
    }
    
    response = lambda_handler(error_event, context)
    print(f"Status Code: {response['statusCode']}")
    
    body = json.loads(response['body'])
    print(f"Error: {body.get('error', 'Unknown error')}")
    
    print("\n" + "-" * 60)
    print("Lambda handler testing complete!")


if __name__ == "__main__":
    # Set environment to use in-memory cache for local testing
    os.environ['USE_REDIS_CACHE'] = 'false'
    
    test_lambda_handler()