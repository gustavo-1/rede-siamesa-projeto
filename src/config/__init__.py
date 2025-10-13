"""
Módulo de configuração que contém constantes e configurações globais do projeto.
"""

from .constants import (
    # Configurações de hardware
    DEVICE,
    
    # Parâmetros de áudio
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, MAX_TIME_STEPS,
    
    # Hiperparâmetros de treinamento
    BATCH_SIZE, EPOCHS, INITIAL_LR, WEIGHT_DECAY,
    
    # Configurações de dados
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    
    # Configurações gerais
    RANDOM_SEED, USE_AUGMENTATION, AUGMENTATION_PROBABILITY,
    
    # Configurações do modelo
    BACKBONE_OPTIONS, SELECTED_BACKBONE, EMBEDDING_DIM
)

__all__ = [
    'SAMPLE_RATE', 'N_MELS', 'HOP_LENGTH', 'N_FFT', 'MAX_TIME_STEPS',
    'BATCH_SIZE', 'EPOCHS', 'INITIAL_LR', 'WEIGHT_DECAY',
    'TRAIN_SPLIT', 'VAL_SPLIT', 'TEST_SPLIT',
    'RANDOM_SEED', 'USE_AUGMENTATION', 'AUGMENTATION_PROBABILITY',
    'BACKBONE_OPTIONS', 'SELECTED_BACKBONE', 'EMBEDDING_DIM'
]
