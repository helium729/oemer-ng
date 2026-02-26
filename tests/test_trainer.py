"""
Unit tests for training utilities.
"""

import torch
import torch.nn as nn
import pytest
import tempfile
import os
from torch.utils.data import DataLoader, TensorDataset
from oemer_ng.training.trainer import Trainer
from oemer_ng.models.omr_model import OMRModel


def create_dummy_dataloader(num_samples=10, batch_size=2, num_classes=64):
    """Create a dummy dataloader for testing."""
    data = torch.randn(num_samples, 3, 256, 256)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_trainer_initialization():
    """Test trainer initialization."""
    model = OMRModel(num_classes=64)
    train_loader = create_dummy_dataloader()
    trainer = Trainer(model=model, train_loader=train_loader)
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.device is not None


def test_checkpoint_save_load():
    """Test checkpoint save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')
        
        # Create trainer and train for a bit
        model = OMRModel(num_classes=64)
        train_loader = create_dummy_dataloader()
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            checkpoint_dir=tmpdir
        )
        
        # Save initial state
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.best_val_loss = 0.5
        trainer.history['train_loss'] = [1.0, 0.8, 0.6]
        trainer.save_checkpoint('test_checkpoint.pth')
        
        # Create new trainer and load checkpoint
        model2 = OMRModel(num_classes=64)
        train_loader2 = create_dummy_dataloader()
        trainer2 = Trainer(
            model=model2,
            train_loader=train_loader2,
            checkpoint_dir=tmpdir
        )
        trainer2.load_checkpoint(checkpoint_path)
        
        # Verify state was restored
        assert trainer2.current_epoch == 5
        assert trainer2.global_step == 100
        assert trainer2.best_val_loss == 0.5
        assert len(trainer2.history['train_loss']) == 3


def test_safe_checkpoint_loading():
    """Test that checkpoints saved with torch.save can be loaded with weights_only=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_safe_checkpoint.pth')
        
        # Create trainer and save checkpoint
        model = OMRModel(num_classes=64)
        train_loader = create_dummy_dataloader()
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            checkpoint_dir=tmpdir
        )
        
        # Set state and save
        trainer.current_epoch = 3
        trainer.global_step = 50
        trainer.best_val_loss = 0.7
        trainer.history['train_loss'] = [1.0, 0.9]
        trainer.save_checkpoint('test_safe_checkpoint.pth')
        
        # Load using weights_only=True (should succeed)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Verify checkpoint structure
        assert isinstance(checkpoint, dict)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'global_step' in checkpoint
        assert 'best_val_loss' in checkpoint
        assert 'history' in checkpoint
        
        # Verify values
        assert checkpoint['epoch'] == 3
        assert checkpoint['global_step'] == 50
        assert checkpoint['best_val_loss'] == 0.7
        assert len(checkpoint['history']['train_loss']) == 2


if __name__ == '__main__':
    pytest.main([__file__])
