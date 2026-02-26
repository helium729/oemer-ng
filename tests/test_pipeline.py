"""
Unit tests for inference pipeline.
"""

import torch
import pytest
import tempfile
import os
import numpy as np
from oemer_ng.inference.pipeline import OMRPipeline
from oemer_ng.models.omr_model import OMRModel


def test_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = OMRPipeline(num_classes=128)
    assert pipeline.model is not None
    assert pipeline.device is not None


def test_pipeline_save_load():
    """Test model save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model.pth')
        
        # Create and save
        pipeline = OMRPipeline(num_classes=64)
        pipeline.save_model(model_path)
        
        # Load
        pipeline2 = OMRPipeline(model_path=model_path, num_classes=64)
        assert pipeline2.model is not None


def test_predict_tensor():
    """Test prediction with tensor input."""
    pipeline = OMRPipeline(num_classes=128)
    img = torch.randn(1, 3, 512, 512)
    result = pipeline.predict(img, enhance=False, return_probabilities=True)
    assert 'prediction' in result
    assert 'confidence' in result
    assert 'probabilities' in result
    assert 0 <= result['prediction'] < 128


def test_predict_simple():
    """Test simple prediction."""
    pipeline = OMRPipeline(num_classes=128)
    img = torch.randn(1, 3, 512, 512)
    result = pipeline.predict(img, enhance=False, return_probabilities=False)
    assert isinstance(result, int)
    assert 0 <= result < 128


def test_quantization():
    """Test quantization."""
    pipeline = OMRPipeline(num_classes=128, use_quantized=True)
    assert pipeline.model is not None
    # Model should work after quantization
    img = torch.randn(1, 3, 512, 512)
    result = pipeline.predict(img, enhance=False)
    assert isinstance(result, int)


def test_predict_batch_empty():
    """Test predict_batch with empty input list."""
    pipeline = OMRPipeline(num_classes=128)
    result = pipeline.predict_batch([])
    assert result == []


def test_predict_batch_basic():
    """Test basic predict_batch functionality."""
    pipeline = OMRPipeline(num_classes=128)
    # Create dummy images
    images = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(4)]
    results = pipeline.predict_batch(images, batch_size=2, enhance=False)
    
    assert len(results) == 4
    assert all(isinstance(r, (int, np.integer)) for r in results)
    assert all(0 <= r < 128 for r in results)


def test_predict_batch_different_num_workers():
    """Test predict_batch with different num_workers values."""
    pipeline = OMRPipeline(num_classes=128)
    images = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(6)]
    
    # Test with num_workers=1
    results_1 = pipeline.predict_batch(images, batch_size=2, num_workers=1, enhance=False)
    assert len(results_1) == 6
    
    # Test with num_workers=2
    results_2 = pipeline.predict_batch(images, batch_size=2, num_workers=2, enhance=False)
    assert len(results_2) == 6
    
    # Test with num_workers=None (default)
    results_none = pipeline.predict_batch(images, batch_size=2, num_workers=None, enhance=False)
    assert len(results_none) == 6


def test_predict_batch_matches_sequential():
    """Test that parallel predict_batch results match sequential processing."""
    pipeline = OMRPipeline(num_classes=128)
    
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create test images
    images = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
    
    # Get sequential predictions (num_workers=1)
    torch.manual_seed(42)
    sequential_results = pipeline.predict_batch(images, batch_size=1, num_workers=1, enhance=False)
    
    # Get parallel predictions (num_workers=4)
    torch.manual_seed(42)
    parallel_results = pipeline.predict_batch(images, batch_size=1, num_workers=4, enhance=False)
    
    # Results should be identical
    assert sequential_results == parallel_results


def test_predict_batch_preprocessing_error():
    """Test error handling when preprocessing fails."""
    pipeline = OMRPipeline(num_classes=128)
    
    # Mock preprocess_image to raise an exception for the second image
    original_preprocess = pipeline.preprocess_image
    
    class FailingPreprocessor:
        def __init__(self):
            self.call_count = 0
        
        def __call__(self, image, enhance=True):
            self.call_count += 1
            if self.call_count == 2:
                raise ValueError("Simulated preprocessing error")
            return original_preprocess(image, enhance)
    
    pipeline.preprocess_image = FailingPreprocessor()
    
    # Create test images
    images = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
    
    # Should raise an error during preprocessing
    with pytest.raises(ValueError, match="Simulated preprocessing error"):
        pipeline.predict_batch(images, batch_size=2, num_workers=2, enhance=False)


def test_predict_batch_single_image():
    """Test predict_batch with a single image."""
    pipeline = OMRPipeline(num_classes=128)
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    results = pipeline.predict_batch([image], batch_size=1, enhance=False)
    
    assert len(results) == 1
    assert isinstance(results[0], (int, np.integer))
    assert 0 <= results[0] < 128


if __name__ == '__main__':
    pytest.main([__file__])
