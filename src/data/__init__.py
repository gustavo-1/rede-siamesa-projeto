"""
Módulo de dados que lida com processamento de áudio, datasets e divisão de dados.
"""

from .audio_processor import ImprovedAudioProcessor
from .dataset import ImprovedSiameseAudioDataset
from .utils_split import (
    create_improved_audio_augmenter,
    combine_datasets,
    split_data_balanced,
    analyze_balanced_split
)

__all__ = [
    'ImprovedAudioProcessor',
    'ImprovedSiameseAudioDataset',
    'create_improved_audio_augmenter',
    'combine_datasets',
    'split_data_balanced',
    'analyze_balanced_split'
]
