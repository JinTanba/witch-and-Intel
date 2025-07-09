"""
AWS Lambda handler for text processing
"""

import json
import asyncio
from typing import Dict, Any

from src.application.handlers.TextProcessingHandler import TextProcessingHandler
from src.core.text_to_coordinates_mapper import TextToCoordinatesMapper
from src.infrastructure.cache.redis_model_cache import RedisModelCache


# Initialize dependencies (this happens once when Lambda container starts)
def _initialize_handler():
    """Initialize handler with production dependencies"""
    # Use Redis cache from environment
    model_cache = RedisModelCache()
    
    # Create text mapper with cache
    text_mapper = TextToCoordinatesMapper(model_cache=model_cache)
    
    # Create handler
    return TextProcessingHandler(text_mapper)


# Global handler instance (reused across Lambda invocations)
_handler = None


def get_handler():
    """Get or create handler instance"""
    global _handler
    if _handler is None:
        _handler = _initialize_handler()
    return _handler


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entry point for text processing
    
    Args:
        event: Lambda event containing:
            - body: JSON string with:
                - sentences: List[str] - sentences to process
                - tag: str (optional) - model tag
        context: Lambda context (unused)
        
    Returns:
        Dict with:
            - statusCode: int
            - body: JSON string with result
            - headers: Dict with CORS headers
    """
    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event
        
        # Get handler instance
        handler = get_handler()
        
        # Run async handler
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(handler.handle_fit_transform(body))
        
        # Format response for API Gateway
        return {
            'statusCode': result['statusCode'],
            'body': json.dumps(result['body']),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            }
        }
        
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'Invalid JSON in request body'
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }