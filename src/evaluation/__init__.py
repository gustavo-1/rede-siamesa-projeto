"""
Módulo de avaliação que contém funções para classificação e visualização de resultados.
"""

from .classifier import (
    classify_audio_improved,
    evaluate_model_improved
)
from .dashboards import (
    create_training_dashboard,
    create_performance_summary_dashboard
)
from .reports import (
    create_confusion_matrix_enhanced,
    create_detailed_classification_report
)

__all__ = [
    'classify_audio_improved',
    'evaluate_model_improved',
    'create_training_dashboard',
    'create_performance_summary_dashboard',
    'create_confusion_matrix_enhanced',
    'create_detailed_classification_report'
]
