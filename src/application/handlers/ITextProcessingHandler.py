from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ITextProcessingHandler(ABC):
    """Interface for handling text processing requests in Lambda context"""
    
    @abstractmethod
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
                    
        Raises:
            ValueError: If sentences are missing or invalid
        """
        pass