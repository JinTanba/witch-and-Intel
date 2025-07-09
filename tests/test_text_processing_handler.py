import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock
from typing import List

from src.application.handlers.TextProcessingHandler import TextProcessingHandler
from src.interfaces.ITtcm import ITextToCoordinatesMapper


class MockTextToCoordinatesMapper(ITextToCoordinatesMapper):
    """Mock implementation of ITextToCoordinatesMapper for testing"""
    
    def fit(self, sentences: List[str], tag: str = "default") -> None:
        pass
        
    def transform(self, sentences: List[str], tag: str = "default") -> np.ndarray:
        pass
        
    def fit_transform(self, sentences: List[str], tag: str = None) -> np.ndarray:
        # Return mock 2D coordinates
        n_sentences = len(sentences)
        return np.array([[i * 0.1, i * 0.2] for i in range(n_sentences)])


@pytest.mark.asyncio
async def test_handle_fit_transform_success():
    # Arrange
    mapper = MockTextToCoordinatesMapper()
    handler = TextProcessingHandler(mapper)
    
    event = {
        "sentences": ["Hello world", "Test sentence", "Another example"],
        "tag": "test"
    }
    
    # Act
    result = await handler.handle_fit_transform(event)
    
    # Assert
    assert result["statusCode"] == 200
    assert "body" in result
    body = result["body"]
    assert "coordinates" in body
    assert "message" in body
    assert len(body["coordinates"]) == 3
    assert body["coordinates"][0] == [0.0, 0.0]
    assert body["coordinates"][1] == [0.1, 0.2]
    assert body["coordinates"][2] == [0.2, 0.4]
    assert body["message"] == "Successfully processed 3 sentences"


@pytest.mark.asyncio
async def test_handle_fit_transform_default_tag():
    # Arrange
    mapper = MockTextToCoordinatesMapper()
    handler = TextProcessingHandler(mapper)
    
    event = {
        "sentences": ["Single sentence"]
    }
    
    # Act
    result = await handler.handle_fit_transform(event)
    
    # Assert
    assert result["statusCode"] == 200
    assert len(result["body"]["coordinates"]) == 1


@pytest.mark.asyncio
async def test_handle_fit_transform_missing_sentences():
    # Arrange
    mapper = MockTextToCoordinatesMapper()
    handler = TextProcessingHandler(mapper)
    
    event = {}
    
    # Act
    result = await handler.handle_fit_transform(event)
    
    # Assert
    assert result["statusCode"] == 400
    assert "error" in result["body"]
    assert "sentences" in result["body"]["error"]


@pytest.mark.asyncio
async def test_handle_fit_transform_empty_sentences():
    # Arrange
    mapper = MockTextToCoordinatesMapper()
    handler = TextProcessingHandler(mapper)
    
    event = {
        "sentences": []
    }
    
    # Act
    result = await handler.handle_fit_transform(event)
    
    # Assert
    assert result["statusCode"] == 400
    assert "error" in result["body"]


@pytest.mark.asyncio
async def test_handle_fit_transform_invalid_sentences():
    # Arrange
    mapper = MockTextToCoordinatesMapper()
    handler = TextProcessingHandler(mapper)
    
    event = {
        "sentences": "not a list"
    }
    
    # Act
    result = await handler.handle_fit_transform(event)
    
    # Assert
    assert result["statusCode"] == 400
    assert "error" in result["body"]