"""
Integration tests for Lambda handler
"""

import json
import pytest
from src.infrastructure.lambda.lambda_handler import lambda_handler


class TestLambdaIntegration:
    """Integration tests for the Lambda handler"""
    
    def test_lambda_handler_with_api_gateway_event(self):
        """Test Lambda handler with API Gateway event format"""
        # API Gateway event format
        event = {
            "body": json.dumps({
                "sentences": ["Hello world", "Test sentence"],
                "tag": "test"
            }),
            "httpMethod": "POST",
            "path": "/text/coordinates"
        }
        
        # Mock context
        context = {}
        
        # Call handler
        response = lambda_handler(event, context)
        
        # Verify response format
        assert response["statusCode"] == 200
        assert "body" in response
        assert "headers" in response
        
        # Verify response body
        body = json.loads(response["body"])
        assert "coordinates" in body
        assert "message" in body
        assert len(body["coordinates"]) == 2
        
        # Verify CORS headers
        headers = response["headers"]
        assert headers["Content-Type"] == "application/json"
        assert headers["Access-Control-Allow-Origin"] == "*"
    
    def test_lambda_handler_with_direct_event(self):
        """Test Lambda handler with direct event format"""
        # Direct event format (no API Gateway wrapper)
        event = {
            "sentences": ["Single sentence"],
            "tag": "direct"
        }
        
        # Mock context
        context = {}
        
        # Call handler
        response = lambda_handler(event, context)
        
        # Verify response
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert len(body["coordinates"]) == 1
    
    def test_lambda_handler_invalid_json(self):
        """Test Lambda handler with invalid JSON"""
        event = {
            "body": "invalid json"
        }
        
        context = {}
        
        response = lambda_handler(event, context)
        
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body
        assert "Invalid JSON" in body["error"]
    
    def test_lambda_handler_missing_sentences(self):
        """Test Lambda handler with missing sentences"""
        event = {
            "body": json.dumps({
                "tag": "test"
            })
        }
        
        context = {}
        
        response = lambda_handler(event, context)
        
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body
        assert "sentences" in body["error"]