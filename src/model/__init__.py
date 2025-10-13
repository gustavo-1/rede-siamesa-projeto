"""
Módulo de modelo que contém a arquitetura da rede siamesa e componentes relacionados.
"""

from .backbone import PretrainedBackbone
from .losses import FocalBCELoss
from .siamese_network import ImprovedSiameseNetwork
from .trainer import train_improved_model
from .optimizer import create_improved_optimizer_and_scheduler

__all__ = [
    'PretrainedBackbone',
    'FocalBCELoss',
    'ImprovedSiameseNetwork',
    'train_improved_model',
    'create_improved_optimizer_and_scheduler'
]
