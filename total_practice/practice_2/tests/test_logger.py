"""Tests for the logger module."""

import os
import tempfile
import logging
import torch
import torch.nn as nn

from processing_own_phase.logger import setup_logger, TensorBoardLogger


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)


def test_setup_logger():
    """Test standard file and console logger creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.log")
        logger = setup_logger("test_logger", log_file=log_file, level=logging.DEBUG)
        
        logger.info("Info message")
        logger.debug("Debug message")
        
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            lines = f.readlines()
            
        assert len(lines) == 2
        assert "Info message" in lines[0]
        assert "Debug message" in lines[1]


def test_tensorboard_logger():
    """Test TensorBoard Logger creation and writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tb = TensorBoardLogger(log_dir=tmpdir)
        
        # Test Scalars
        tb.log_scalars({"Loss/Train": 0.5, "Accuracy/Train": 90.0}, 1)
        tb.log_learning_rate(0.001, 1)
        
        # Test Histograms & Graph
        model = SimpleModel()
        dummy_input = torch.randn(2, 10)
        
        # Backward pass to create some gradients
        out = model(dummy_input)
        loss = out.sum()
        loss.backward()
        
        tb.log_histograms(model, 1)
        tb.log_model_graph(model, dummy_input)
        
        # Test Images
        dummy_images = torch.rand(4, 3, 32, 32)
        tb.log_images("Samples", dummy_images, 1)
        
        tb.close()
        
        # Check that events file was created
        files = os.listdir(tmpdir)
        events_files = [f for f in files if "events.out.tfevents" in f]
        assert len(events_files) > 0
