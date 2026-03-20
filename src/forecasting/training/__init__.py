"""
Training module - Responsible for model training only.

This module handles rolling window training with complete separation from validation.
"""

from .rolling_trainer import RollingWindowTrainer, TrainWindow, TrainedModelResult

__all__ = [
    'RollingWindowTrainer',
    'TrainWindow',
    'TrainedModelResult',
]
