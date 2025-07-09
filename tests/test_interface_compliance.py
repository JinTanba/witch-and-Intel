"""
Tests to ensure TextToCoordinatesMapper properly implements ITextToCoordinatesMapper interface.
"""

import pytest
import numpy as np
from typing import List

from src.interfaces.ITtcm import ITextToCoordinatesMapper
from src.text_to_coordinates_mapper import TextToCoordinatesMapper


class TestInterfaceCompliance:
    """Test that TextToCoordinatesMapper complies with its interface."""
    
    def test_implements_interface(self):
        """Test that TextToCoordinatesMapper is a subclass of ITextToCoordinatesMapper."""
        assert issubclass(TextToCoordinatesMapper, ITextToCoordinatesMapper)
    
    def test_instance_of_interface(self):
        """Test that mapper instance is an instance of the interface."""
        mapper = TextToCoordinatesMapper()
        assert isinstance(mapper, ITextToCoordinatesMapper)
    
    def test_interface_method_exists(self):
        """Test that the required interface method exists."""
        mapper = TextToCoordinatesMapper()
        assert hasattr(mapper, 'fit_transform')
        assert callable(mapper.fit_transform)
    
    def test_interface_method_signature(self):
        """Test that fit_transform has the correct signature."""
        mapper = TextToCoordinatesMapper()
        sentences = ["Test sentence"]
        result = mapper.fit_transform(sentences)
        
        # Check input type
        assert isinstance(sentences, List)
        assert all(isinstance(s, str) for s in sentences)
        
        # Check output type
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)
    
    def test_interface_contract_empty_list(self):
        """Test that interface contract is honored for empty list."""
        mapper = TextToCoordinatesMapper()
        
        # According to interface, should raise ValueError for empty list
        with pytest.raises(ValueError):
            mapper.fit_transform([])
    
    def test_can_use_through_interface(self):
        """Test that we can use the mapper through its interface type."""
        def process_with_interface(mapper: ITextToCoordinatesMapper, sentences: List[str]) -> np.ndarray:
            """Function that only knows about the interface."""
            return mapper.fit_transform(sentences)
        
        # Create concrete implementation
        concrete_mapper = TextToCoordinatesMapper()
        
        # Use through interface
        sentences = ["Interface test 1", "Interface test 2"]
        result = process_with_interface(concrete_mapper, sentences)
        
        assert result.shape == (2, 2)
        assert isinstance(result, np.ndarray)