"""
Módulo que contém funções para gerar relatórios detalhados de avaliação do modelo.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px

def create_confusion_matrix_enhanced(y_true, y_pred, labels=['Negativo', 'Positivo'], normalize=True):
    """
    Cria uma matriz de confusão melhorada com visualização e métricas detalhadas.
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Predições do modelo
        labels: Lista com nomes das classes
        normalize: Se True, normaliza os valores da matriz
        
    Returns:
        tuple: (fig, df_metrics) - Figura da matriz de confusão e DataFrame com métricas
    """
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    # Criar heatmap com Plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='RdBu',
        text=cm,
        texttemplate=f'%{{z:{fmt}}}',
        textfont={"size": 16},
        hoverongaps=False,
    ))
    
    # Atualizar layout
    fig.update_layout(
        title='Matriz de Confusão',
        xaxis_title='Predito',
        yaxis_title='Real',
        width=600,
        height=600,
    )
    
    # Calcular métricas adicionais
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'True Negative Rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'False Positive Rate': fp / (tn + fp) if (tn + fp) > 0 else 0,
        'False Negative Rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'True Positive Rate': tp / (fn + tp) if (fn + tp) > 0 else 0,
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'Balanced Accuracy': (
            (tn / (tn + fp) if (tn + fp) > 0 else 0) +
            (tp / (tp + fn) if (tp + fn) > 0 else 0)
        ) / 2,
        'F1 Score': 2 * tp / (2 * tp + fp + fn),
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0
    }
    
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=['Valor'])
    df_metrics = df_metrics.round(4)
    
    return fig, df_metrics

def create_detailed_classification_report(y_true, y_pred, y_scores=None):
    """
    Gera um relatório detalhado de classificação com métricas e visualizações.
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Predições do modelo
        y_scores: Probabilidades/scores do modelo (opcional)
        
    Returns:
        dict: Dicionário contendo o relatório completo e visualizações
    """
    # Relatório básico de classificação
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    # Matriz de confusão e métricas
    cm_fig, metrics_df = create_confusion_matrix_enhanced(y_true, y_pred)
    
    # Análise de distribuição das predições
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    pred_dist_fig = px.bar(
        x=pred_dist.index,
        y=pred_dist.values,
        labels={'x': 'Classe', 'y': 'Contagem'},
        title='Distribuição das Predições'
    )
    
    # ROC Curve se houver scores disponíveis
    roc_fig = None
    if y_scores is not None:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC curve (AUC = {roc_auc:.2f})',
            mode='lines'
        ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Random'
        ))
        roc_fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
    
    # Compilar resultados
    results = {
        'classification_report': report_dict,
        'confusion_matrix': cm_fig,
        'detailed_metrics': metrics_df,
        'prediction_distribution': pred_dist_fig,
        'roc_curve': roc_fig
    }
    
    return results