"""
Módulo que contém funções para criar dashboards interativos de visualização dos resultados.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def create_training_dashboard(training_history, model_name="Siamese Network"):
    """
    Cria um dashboard interativo com métricas de treinamento.
    
    Args:
        training_history (dict): Histórico do treinamento contendo loss e métricas
        model_name (str): Nome do modelo para exibição
        
    Returns:
        go.Figure: Figura do Plotly com o dashboard
    """
    # Criar figura com subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Loss ao Longo do Treinamento',
            'Accuracy ao Longo do Treinamento',
            'Outras Métricas ao Longo do Treinamento',
            'Distribuição dos Valores de Loss'
        )
    )
    
    # Plot de Loss
    fig.add_trace(
        go.Scatter(
            y=training_history['train_loss'],
            name='Train Loss',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    if 'val_loss' in training_history:
        fig.add_trace(
            go.Scatter(
                y=training_history['val_loss'],
                name='Validation Loss',
                line=dict(color='red')
            ),
            row=1, col=1
        )
    
    # Plot de Accuracy
    if 'train_accuracy' in training_history:
        fig.add_trace(
            go.Scatter(
                y=training_history['train_accuracy'],
                name='Train Accuracy',
                line=dict(color='green')
            ),
            row=1, col=2
        )
    if 'val_accuracy' in training_history:
        fig.add_trace(
            go.Scatter(
                y=training_history['val_accuracy'],
                name='Validation Accuracy',
                line=dict(color='orange')
            ),
            row=1, col=2
        )
    
    # Plot de outras métricas (F1, Precision, Recall)
    metrics = ['f1', 'precision', 'recall']
    colors = ['purple', 'brown', 'pink']
    
    for metric, color in zip(metrics, colors):
        if f'train_{metric}' in training_history:
            fig.add_trace(
                go.Scatter(
                    y=training_history[f'train_{metric}'],
                    name=f'Train {metric.capitalize()}',
                    line=dict(color=color)
                ),
                row=2, col=1
            )
    
    # Distribuição dos valores de loss
    fig.add_trace(
        go.Histogram(
            x=training_history['train_loss'],
            name='Train Loss Distribution',
            nbinsx=30,
            marker_color='blue'
        ),
        row=2, col=2
    )
    
    # Atualizar layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Dashboard de Treinamento - {model_name}",
        template='plotly_white'
    )
    
    # Atualizar eixos
    fig.update_xaxes(title_text="Época", row=1, col=1)
    fig.update_xaxes(title_text="Época", row=1, col=2)
    fig.update_xaxes(title_text="Época", row=2, col=1)
    fig.update_xaxes(title_text="Valor de Loss", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="Valor", row=2, col=1)
    fig.update_yaxes(title_text="Frequência", row=2, col=2)
    
    return fig

def create_performance_summary_dashboard(y_true, y_pred, y_scores=None, model_name="Siamese Network"):
    """
    Cria um dashboard de resumo de performance do modelo com métricas detalhadas.
    
    Args:
        y_true (array-like): Labels verdadeiras
        y_pred (array-like): Predições do modelo (classificações)
        y_scores (array-like, optional): Scores/probabilidades do modelo
        model_name (str): Nome do modelo para exibição
    
    Returns:
        go.Figure: Figura do Plotly com o dashboard
    """
    # Criar figura com subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Matriz de Confusão',
            'Distribuição das Predições',
            'Métricas de Performance',
            'ROC Curve' if y_scores is not None else 'Distribuição de Scores'
        ),
        specs=[[{"type": "heatmap"}, {"type": "bar"}],
               [{"type": "table"}, {"type": "scatter"}]]
    )
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=['Negativo', 'Positivo'],
            y=['Negativo', 'Positivo'],
            colorscale='RdBu',
            showscale=True,
            text=cm,
            texttemplate="%{z}",
            textfont={"size": 16},
        ),
        row=1, col=1
    )
    
    # Distribuição das Predições
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=['Negativo', 'Positivo'],
            y=pred_counts,
            name='Predições',
            marker_color=['lightblue', 'lightgreen']
        ),
        row=1, col=2
    )
    
    # Métricas de Performance do Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Métrica', 'Precisão', 'Recall', 'F1-Score', 'Support'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    ['Classe 0', 'Classe 1', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
                    metrics_df['precision'].round(3),
                    metrics_df['recall'].round(3),
                    metrics_df['f1-score'].round(3),
                    metrics_df['support']
                ],
                fill_color='lavender',
                align='left'
            )
        ),
        row=2, col=1
    )
    
    # ROC Curve ou Distribuição de Scores
    if y_scores is not None:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve (AUC = {roc_auc:.2f})',
                mode='lines'
            ),
            row=2, col=2
        )
        
        # Adicionar linha diagonal de referência
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash'),
                name='Random'
            ),
            row=2, col=2
        )
    else:
        # Se não tiver scores, mostra distribuição das predições por classe verdadeira
        for label in [0, 1]:
            mask = y_true == label
            fig.add_trace(
                go.Box(
                    y=y_pred[mask],
                    name=f'Classe {label}',
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=2, col=2
            )
    
    # Atualizar layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text=f"Dashboard de Performance - {model_name}",
        template='plotly_white'
    )
    
    # Atualizar eixos da ROC curve
    if y_scores is not None:
        fig.update_xaxes(title_text='False Positive Rate', row=2, col=2)
        fig.update_yaxes(title_text='True Positive Rate', row=2, col=2)
    else:
        fig.update_xaxes(title_text='Classe Verdadeira', row=2, col=2)
        fig.update_yaxes(title_text='Predições', row=2, col=2)
    
    return fig