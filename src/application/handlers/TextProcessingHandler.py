from typing import Dict, Any, List
import numpy as np

from src.application.handlers.ITextProcessingHandler import ITextProcessingHandler
from src.interfaces.ITtcm import ITextToCoordinatesMapper


class TextProcessingHandler(ITextProcessingHandler):
    """Handler for text processing requests in Lambda context"""
    
    def __init__(self, text_mapper: ITextToCoordinatesMapper):
        """
        Initialize handler with text mapper dependency
        
        Args:
            text_mapper: Instance of ITextToCoordinatesMapper
        """
        self.text_mapper = text_mapper
    
    async def handle_fit_transform(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle fit_transform request for text to coordinates mapping
        
        Args:
            event: Lambda event containing:
                - sentences: List[str] - sentences to process
                - tag: str (optional) - model tag, defaults to "default"
                
        Returns:
            Dict containing:
                - statusCode: int
                - body: Dict with:
                    - coordinates: List[List[float]] - 2D coordinates
                    - message: str - status message
        """
        try:
            # Validate input
            if "sentences" not in event:
                return {
                    "statusCode": 400,
                    "body": {
                        "error": "Missing required field: sentences"
                    }
                }
            
            sentences = event["sentences"]
            
            # Validate sentences is a list
            if not isinstance(sentences, list):
                return {
                    "statusCode": 400,
                    "body": {
                        "error": "Field 'sentences' must be a list"
                    }
                }
            
            # Validate sentences is not empty
            if len(sentences) == 0:
                return {
                    "statusCode": 400,
                    "body": {
                        "error": "Field 'sentences' cannot be empty"
                    }
                }
            
            # Get optional tag
            tag = event.get("tag", "default")
            
            # Perform fit_transform
            coordinates = self.text_mapper.fit_transform(sentences, tag)
            
            # Convert numpy array to list for JSON serialization
            coordinates_list = coordinates.tolist()
            
            return {
                "statusCode": 200,
                "body": {
                    "coordinates": coordinates_list,
                    "message": f"Successfully processed {len(sentences)} sentences"
                }
            }
            
        except ValueError as e:
            return {
                "statusCode": 400,
                "body": {
                    "error": str(e)
                }
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": {
                    "error": f"Internal server error: {str(e)}"
                }
            }