# pipeline completo de treino + avaliação

# Importações da biblioteca padrão
import os
import random
import json
from datetime import datetime
from collections import defaultdict, Counter

# Importações de bibliotecas de terceiros
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import audiomentations as A
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
# from tqdm import tqdm

# Importações locais
from src.config.constants import *
from src.data.audio_processor import ImprovedAudioProcessor
from src.data.dataset import ImprovedSiameseAudioDataset
from src.data.utils_split import combine_datasets, split_data_balanced, analyze_balanced_split
from src.model.siamese_network import ImprovedSiameseNetwork
from src.model.losses import FocalBCELoss
from src.model import train_improved_model
from src.model.optimizer import create_improved_optimizer_and_scheduler
from src.utils import EarlyStopping
from src.evaluation.classifier import evaluate_model_improved
from src.evaluation.dashboards import create_training_dashboard
from tqdm import tqdm

# Configuração do modelo
# 📊 CONFIGURAÇÃO DO MODELO:
#    🧠 Backbone: mobilenet_v3_small
#    🔧 Batch Size: 16
#    📈 Épocas: 25
#    ⚡ Learning Rate: 0.001
#    🏅 Melhor época: 9
#    ✅ Melhor F1 validação: 0.9154

# 🎯 MÉTRICAS PRINCIPAIS:
#    🎪 Accuracy:  0.7714 🟢
#    🎯 Precision: 0.6810 🟡
#    📡 Recall:    0.7714 🟢
#    🏆 F1-Score:  0.7110 🟢

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pyo.init_notebook_mode()

# Define semente para reprodutibilidade
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Função de classificação melhorada
def classify_audio_improved(model, audio_file, reference_data, processor):
    model.eval()
    classes = list(reference_data.keys())
   
    waveform = processor.load_audio(audio_file, apply_augmentation=False)
    test_spec = processor.extract_features(waveform)
    test_spec = torch.FloatTensor(test_spec).unsqueeze(0).to(DEVICE)
   
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_spec = transform(test_spec)
   
    with torch.no_grad():
        test_embedding = model.feature_extractor(model.backbone(test_spec))
   
    similarities = {}
    for cls, files in reference_data.items():
        cls_embeddings = []
       
        for file in files:
            try:
                waveform = processor.load_audio(file, apply_augmentation=False)
                ref_spec = processor.extract_features(waveform)
                ref_spec = torch.FloatTensor(ref_spec).unsqueeze(0).to(DEVICE)
                ref_spec = transform(ref_spec)
               
                with torch.no_grad():
                    ref_embedding = model.feature_extractor(model.backbone(ref_spec))
                    cls_embeddings.append(ref_embedding)
            except Exception as e:
                print(f"Erro ao processar {file}: {e}")
                continue
       
        if cls_embeddings:
            cls_embedding = torch.mean(torch.stack(cls_embeddings), dim=0)
           
            # Múltiplas métricas como no treinamento
            abs_diff = torch.abs(test_embedding - cls_embedding)
            sq_diff = torch.pow(test_embedding - cls_embedding, 2)
            element_prod = test_embedding * cls_embedding
            
            # Processa através das camadas de comparação
            comp1 = model.comparison_layers[0](abs_diff)
            comp2 = model.comparison_layers[1](sq_diff)
            comp3 = model.comparison_layers[2](element_prod)
            
            # Fusão final
            fused = torch.cat([comp1, comp2, comp3], dim=1)
            similarity = model.fusion(fused).item()
            similarities[cls] = similarity
   
    if similarities:
        predicted_class = max(similarities, key=similarities.get)
        return predicted_class, similarities
    else:
        return None, {}

# Função para avaliar o modelo completo
def evaluate_model_improved(model, test_data, reference_data, processor):
    """Avalia o modelo melhorado em todo o conjunto de teste"""
    results = {
        'predictions': [],
        'true_labels': [],
        'similarities': []
    }
   
    print("Avaliando modelo MELHORADO no conjunto de teste...")
   
    for true_class, files in tqdm(test_data.items(), desc="Avaliando classes"):
        for file in files:
            try:
                predicted_class, similarities = classify_audio_improved(model, file, reference_data, processor)
               
                if predicted_class is not None:
                    results['predictions'].append(predicted_class)
                    results['true_labels'].append(true_class)
                    results['similarities'].append(similarities)
            except Exception as e:
                print(f"Erro ao avaliar {file}: {e}")
                continue
   
    if results['predictions']:
        accuracy = accuracy_score(results['true_labels'], results['predictions'])
        precision = precision_score(results['true_labels'], results['predictions'], average='weighted', zero_division=0)
        recall = recall_score(results['true_labels'], results['predictions'], average='weighted', zero_division=0)
        f1 = f1_score(results['true_labels'], results['predictions'], average='weighted', zero_division=0)
       
        print(f"\nMétricas de Avaliação MELHORADA ({SELECTED_BACKBONE}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
       
        # Matriz de confusão
        classes = sorted(set(results['true_labels'] + results['predictions']))
        cm = confusion_matrix(results['true_labels'], results['predictions'], labels=classes)
       
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title(f'Matriz de Confusão MELHORADA - {SELECTED_BACKBONE}')
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Predita')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_improved_{SELECTED_BACKBONE}.png', dpi=300, bbox_inches='tight')
        plt.close()
       
        return results
    else:
        print("Nenhuma predição foi feita com sucesso.")
        return None




def create_training_dashboard(train_losses, val_accuracies, checkpoint_info=None):
    """Cria dashboard do treinamento (versão Matplotlib, sem Plotly)"""
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Curva de Loss
    axes[0,0].plot(epochs, train_losses, color='red', linewidth=2, label="Loss Treino")
    axes[0,0].set_title("Curva de Loss")
    axes[0,0].set_xlabel("Época")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Curva de Accuracy
    axes[0,1].plot(epochs, val_accuracies, color='blue', linewidth=2, label="Accuracy Validação")
    axes[0,1].set_title("Curva de Accuracy")
    axes[0,1].set_xlabel("Época")
    axes[0,1].set_ylabel("Accuracy")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Convergência (Loss + Accuracy)
    ax2 = axes[1,0].twinx()
    axes[1,0].plot(epochs, train_losses, color='red', linewidth=2, label="Loss")
    ax2.plot(epochs, val_accuracies, color='blue', linewidth=2, label="Accuracy")
    axes[1,0].set_title("Convergência")
    axes[1,0].set_xlabel("Época")
    axes[1,0].set_ylabel("Loss", color='red')
    ax2.set_ylabel("Accuracy", color='blue')
    axes[1,0].grid(True, alpha=0.3)

    # 4. Métrica final (simples)
    best_acc = max(val_accuracies) if val_accuracies else 0
    axes[1,1].bar(["Melhor Accuracy"], [best_acc], color="green" if best_acc > 0.5 else "red")
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_title("Melhor Accuracy")
    axes[1,1].text(0, best_acc + 0.02, f"{best_acc:.3f}", ha="center", fontsize=12)

    plt.suptitle("Dashboard de Treinamento - Rede Siamesa", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"training_dashboard_{SELECTED_BACKBONE}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fig




def create_confusion_matrix_enhanced(y_true, y_pred, classes):
    """Cria matriz de confusão melhorada com métricas detalhadas"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Matriz de confusão absoluta
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, 
               cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Matriz de Confusão - Valores Absolutos')
    axes[0,0].set_ylabel('Classe Verdadeira')
    axes[0,0].set_xlabel('Classe Predita')
    
    # Matriz de confusão normalizada
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', xticklabels=classes, yticklabels=classes,
               cmap='Blues', ax=axes[0,1])
    axes[0,1].set_title('Matriz de Confusão - Normalizada por Linha')
    axes[0,1].set_ylabel('Classe Verdadeira')
    axes[0,1].set_xlabel('Classe Predita')
    
    # Acurácia por classe
    class_accuracy = cm_norm.diagonal()
    axes[1,0].bar(range(len(classes)), class_accuracy, color='skyblue', edgecolor='navy')
    axes[1,0].set_title('Acurácia por Classe')
    axes[1,0].set_xlabel('Classes')
    axes[1,0].set_ylabel('Acurácia')
    axes[1,0].set_xticks(range(len(classes)))
    axes[1,0].set_xticklabels(classes, rotation=45, ha='right')
    axes[1,0].grid(True, alpha=0.3)
    
    # Adiciona valores nas barras
    for i, v in enumerate(class_accuracy):
        axes[1,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Distribuição de predições
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_true)
    
    x_pos = np.arange(len(classes))
    width = 0.35
    
    true_values = [true_counts.get(cls, 0) for cls in classes]
    pred_values = [pred_counts.get(cls, 0) for cls in classes]
    
    axes[1,1].bar(x_pos - width/2, true_values, width, label='Verdadeiro', 
                  color='lightgreen', edgecolor='darkgreen')
    axes[1,1].bar(x_pos + width/2, pred_values, width, label='Predito',
                  color='lightcoral', edgecolor='darkred')
    
    axes[1,1].set_title('Distribuição: Verdadeiro vs Predito')
    axes[1,1].set_xlabel('Classes')
    axes[1,1].set_ylabel('Quantidade')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_enhanced_{SELECTED_BACKBONE}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_similarity_analysis(evaluation_results):
    """Analisa distribuições de similaridade"""
    if not evaluation_results or not evaluation_results['similarities']:
        print("Dados de similaridade insuficientes para análise")
        return None
    
    # Coleta dados de similaridade
    correct_sims = []
    incorrect_sims = []
    class_similarities = {}
    
    for i, (true_label, pred_label, similarities) in enumerate(zip(
        evaluation_results['true_labels'], 
        evaluation_results['predictions'],
        evaluation_results['similarities']
    )):
        if true_label == pred_label:
            correct_sims.append(similarities[pred_label])
        else:
            incorrect_sims.append(similarities[pred_label])
        
        # Por classe
        if true_label not in class_similarities:
            class_similarities[true_label] = []
        if true_label in similarities:
            class_similarities[true_label].append(similarities[true_label])
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Distribuição geral de similaridades
    all_sims = correct_sims + incorrect_sims
    axes[0,0].hist(all_sims, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribuição Geral de Similaridades')
    axes[0,0].set_xlabel('Similaridade')
    axes[0,0].set_ylabel('Frequência')
    axes[0,0].grid(True, alpha=0.3)
    
    # Comparação: Corretas vs Incorretas
    if correct_sims and incorrect_sims:
        axes[0,1].hist([correct_sims, incorrect_sims], bins=20, alpha=0.7, 
                      label=['Predições Corretas', 'Predições Incorretas'],
                      color=['green', 'red'])
        axes[0,1].set_title('Similaridades: Corretas vs Incorretas')
        axes[0,1].set_xlabel('Similaridade')
        axes[0,1].set_ylabel('Frequência')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Boxplot por classe (top classes)
    if class_similarities:
        # Pega as 10 classes com mais dados
        sorted_classes = sorted(class_similarities.items(), 
                              key=lambda x: len(x[1]), reverse=True)[:10]
        
        if sorted_classes:
            class_names = [cls for cls, _ in sorted_classes]
            class_data = [sims for _, sims in sorted_classes]
            
            axes[1,0].boxplot(class_data, labels=class_names)
            axes[1,0].set_title('Distribuição de Similaridade por Classe (Top 10)')
            axes[1,0].set_xlabel('Classes')
            axes[1,0].set_ylabel('Similaridade')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
    
    # Estatísticas de similaridade
    if correct_sims:
        mean_correct = f"{np.mean(correct_sims):.4f}"
    else:
        mean_correct = "N/A"

    if incorrect_sims:
        mean_incorrect = f"{np.mean(incorrect_sims):.4f}"
    else:
        mean_incorrect = "N/A"

    stats_text = f"""Estatísticas de Similaridade:

    Total de amostras: {len(all_sims)}
    Similaridade média: {np.mean(all_sims):.4f}
    Desvio padrão: {np.std(all_sims):.4f}
    Mínima: {np.min(all_sims):.4f}
    Máxima: {np.max(all_sims):.4f}

    Predições Corretas:
    - Média: {mean_correct}
    - Count: {len(correct_sims)}

    Predições Incorretas:
    - Média: {mean_incorrect}  
    - Count: {len(incorrect_sims)}
    """
    
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    axes[1,1].set_title('Estatísticas Detalhadas')
    
    plt.tight_layout()
    plt.savefig(f'similarity_analysis_{SELECTED_BACKBONE}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_classification_report(y_true, y_pred, classes):
    """Cria relatório de classificação detalhado com visualizações"""
    report = classification_report(y_true, y_pred, labels=classes, 
                                 target_names=classes, output_dict=True)
    
    # Converte para DataFrame para melhor visualização
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.iloc[:-3]  # Remove macro avg, weighted avg, accuracy
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Precision por classe
    precision_values = df_report['precision'].values
    axes[0,0].bar(range(len(classes)), precision_values, color='lightblue', edgecolor='navy')
    axes[0,0].set_title('Precision por Classe')
    axes[0,0].set_xlabel('Classes')
    axes[0,0].set_ylabel('Precision')
    axes[0,0].set_xticks(range(len(classes)))
    axes[0,0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0,0].grid(True, alpha=0.3)
    
    # Adiciona valores
    for i, v in enumerate(precision_values):
        axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Recall por classe
    recall_values = df_report['recall'].values
    axes[0,1].bar(range(len(classes)), recall_values, color='lightgreen', edgecolor='darkgreen')
    axes[0,1].set_title('Recall por Classe')
    axes[0,1].set_xlabel('Classes')
    axes[0,1].set_ylabel('Recall')
    axes[0,1].set_xticks(range(len(classes)))
    axes[0,1].set_xticklabels(classes, rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)
    
    for i, v in enumerate(recall_values):
        axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # F1-Score por classe
    f1_values = df_report['f1-score'].values
    axes[1,0].bar(range(len(classes)), f1_values, color='lightcoral', edgecolor='darkred')
    axes[1,0].set_title('F1-Score por Classe')
    axes[1,0].set_xlabel('Classes')
    axes[1,0].set_ylabel('F1-Score')
    axes[1,0].set_xticks(range(len(classes)))
    axes[1,0].set_xticklabels(classes, rotation=45, ha='right')
    axes[1,0].grid(True, alpha=0.3)
    
    for i, v in enumerate(f1_values):
        axes[1,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Heatmap das métricas
    metrics_data = df_report[['precision', 'recall', 'f1-score']].values
    sns.heatmap(metrics_data, annot=True, fmt='.3f', 
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=classes, cmap='RdYlBu_r', ax=axes[1,1])
    axes[1,1].set_title('Heatmap das Métricas por Classe')
    axes[1,1].set_ylabel('Classes')
    
    plt.tight_layout()
    plt.savefig(f'classification_report_detailed_{SELECTED_BACKBONE}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Imprime tabela detalhada
    print("\n" + "="*80)
    print("RELATÓRIO DETALHADO DE CLASSIFICAÇÃO")
    print("="*80)
    print(f"{'Classe':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-"*70)
    
    for cls in classes:
        if cls in df_report.index:
            row = df_report.loc[cls]
            print(f"{cls:<20} {row['precision']:<10.3f} {row['recall']:<10.3f} "
                  f"{row['f1-score']:<10.3f} {int(row['support']):<10}")
    
    print("-"*70)
    print(f"{'MACRO AVG':<20} {report['macro avg']['precision']:<10.3f} "
          f"{report['macro avg']['recall']:<10.3f} {report['macro avg']['f1-score']:<10.3f}")
    print(f"{'WEIGHTED AVG':<20} {report['weighted avg']['precision']:<10.3f} "
          f"{report['weighted avg']['recall']:<10.3f} {report['weighted avg']['f1-score']:<10.3f}")
    
    return fig, df_report

def create_performance_summary_dashboard(evaluation_results, checkpoint_info=None):
    """Cria dashboard resumo da performance (versão Matplotlib, sem Plotly)"""
    if not evaluation_results:
        return None

    y_true = evaluation_results['true_labels']
    y_pred = evaluation_results['predictions']

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Métricas principais
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    classes = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Métricas principais (texto resumido)
    axes[0,0].axis("off")
    axes[0,0].text(0.1, 0.8, f"Accuracy:  {accuracy:.3f}", fontsize=12)
    axes[0,0].text(0.1, 0.6, f"Precision: {precision:.3f}", fontsize=12)
    axes[0,0].text(0.1, 0.4, f"Recall:    {recall:.3f}", fontsize=12)
    axes[0,0].text(0.1, 0.2, f"F1-Score:  {f1:.3f}", fontsize=12)
    axes[0,0].set_title("Métricas Principais")

    # 2. Distribuição de classes
    from collections import Counter
    counts = Counter(y_true)
    axes[0,1].bar(counts.keys(), counts.values(), color="skyblue", edgecolor="black")
    axes[0,1].set_title("Distribuição de Classes")
    axes[0,1].tick_params(axis='x', rotation=45)

    # 3. F1 por classe
    f1_scores = []
    for cls in classes:
        y_true_bin = [1 if y == cls else 0 for y in y_true]
        y_pred_bin = [1 if y == cls else 0 for y in y_pred]
        f1_scores.append(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    axes[0,2].bar(classes, f1_scores, color="lightcoral", edgecolor="darkred")
    axes[0,2].set_title("F1-Score por Classe")
    axes[0,2].tick_params(axis='x', rotation=45)

    # 4. Matriz de confusão
    im = axes[1,0].imshow(cm, cmap="Blues")
    axes[1,0].set_xticks(np.arange(len(classes)))
    axes[1,0].set_yticks(np.arange(len(classes)))
    axes[1,0].set_xticklabels(classes, rotation=45)
    axes[1,0].set_yticklabels(classes)
    axes[1,0].set_title("Matriz de Confusão")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[1,0].text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[1,0])

    # 5. Evolução (se houver dados de treino no checkpoint)
    if checkpoint_info and "train_losses" in checkpoint_info and "val_accuracies" in checkpoint_info:
        epochs = range(1, len(checkpoint_info["train_losses"]) + 1)
        axes[1,1].plot(epochs, checkpoint_info["train_losses"], color="red", label="Loss")
        axes[1,1].plot(epochs, checkpoint_info["val_accuracies"], color="blue", label="Accuracy")
        axes[1,1].set_title("Evolução do Treinamento")
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].axis("off")

    # 6. Resumo Final
    axes[1,2].axis("off")
    axes[1,2].text(0.1, 0.5, f"F1 Final: {f1:.3f}", fontsize=14, fontweight="bold")
    axes[1,2].set_title("Resumo Final")

    plt.suptitle(f"Dashboard de Performance - {SELECTED_BACKBONE}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"performance_dashboard_{SELECTED_BACKBONE}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fig

def print_final_summary(evaluation_results, checkpoint_info=None):
    """Imprime resumo final detalhado e colorido"""
    print("\n" + "🎯" + "="*78 + "🎯")
    print("🏆                 RELATÓRIO FINAL DE PERFORMANCE                 🏆")
    print("🎯" + "="*78 + "🎯")
    
    if not evaluation_results:
        print("❌ Nenhum resultado de avaliação disponível!")
        return
    
    y_true = evaluation_results['true_labels']
    y_pred = evaluation_results['predictions']
    
    # Métricas principais
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Informações do modelo
    print(f"\n📊 CONFIGURAÇÃO DO MODELO:")
    print(f"   🧠 Backbone: {SELECTED_BACKBONE}")
    print(f"   🔧 Batch Size: {BATCH_SIZE}")
    print(f"   📈 Épocas: {EPOCHS}")
    print(f"   ⚡ Learning Rate: {INITIAL_LR}")
    if checkpoint_info:
        print(f"   🏅 Melhor época: {checkpoint_info.get('epoch', 'N/A')}")
        print(f"   ✅ Melhor F1 validação: {checkpoint_info.get('val_f1', 'N/A'):.4f}")
    
    # Métricas principais com cores
    print(f"\n🎯 MÉTRICAS PRINCIPAIS:")
    print(f"   🎪 Accuracy:  {accuracy:.4f} {'🟢' if accuracy > 0.7 else '🟡' if accuracy > 0.5 else '🔴'}")
    print(f"   🎯 Precision: {precision:.4f} {'🟢' if precision > 0.7 else '🟡' if precision > 0.5 else '🔴'}")
    print(f"   📡 Recall:    {recall:.4f} {'🟢' if recall > 0.7 else '🟡' if recall > 0.5 else '🔴'}")
    print(f"   🏆 F1-Score:  {f1:.4f} {'🟢' if f1 > 0.7 else '🟡' if f1 > 0.5 else '🔴'}")
    
    # Análise de performance
    random_baseline = 1.0 / len(set(y_true))
    improvement = (accuracy - random_baseline) / random_baseline * 100
    
    print(f"\n📈 ANÁLISE DE PERFORMANCE:")
    print(f"   📊 Total de amostras testadas: {len(y_true)}")
    print(f"   🎲 Baseline aleatório: {random_baseline:.4f}")
    print(f"   🚀 Melhoria sobre baseline: {improvement:.1f}%")
    print(f"   🎯 Classes únicas: {len(set(y_true))}")
    
    # Status final
    print(f"\n🏁 STATUS FINAL:")
    if accuracy > 0.8:
        print("   🌟 EXCELENTE! Performance excepcional!")
        print("   ✨ Modelo pronto para produção!")
    elif accuracy > 0.7:
        print("   🎉 MUITO BOM! Performance sólida!")
        print("   ✅ Modelo funciona bem para a maioria dos casos!")
    elif accuracy > 0.5:
        print("   ⚡ BOM! Performance acima do baseline!")
        print("   🔧 Considere ajustes finos para melhorar!")
    else:
        print("   ⚠️ ATENÇÃO! Performance ainda pode melhorar!")
        print("   🛠️ Revisar dados e estratégia de treinamento!")
    
    # Top e bottom classes
    classes_unique = sorted(set(y_true))
    class_f1_scores = {}
    
    for cls in classes_unique:
        y_true_binary = [1 if label == cls else 0 for label in y_true]
        y_pred_binary = [1 if pred == cls else 0 for pred in y_pred]
        f1_cls = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        class_f1_scores[cls] = f1_cls
    
    # Ordenar por F1-Score
    sorted_classes = sorted(class_f1_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 TOP 5 CLASSES (F1-Score):")
    for i, (cls, score) in enumerate(sorted_classes[:5]):
        print(f"   {i+1}. {cls}: {score:.4f} {'🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else '🏅'}")
    
    if len(sorted_classes) > 5:
        print(f"\n⚠️ BOTTOM 5 CLASSES:")
        for i, (cls, score) in enumerate(sorted_classes[-5:]):
            print(f"   {len(sorted_classes)-4+i}. {cls}: {score:.4f} {'❌' if score < 0.3 else '⚠️'}")
    
    print("\n" + "🎯" + "="*78 + "🎯")
    print("📁 Arquivos salvos:")
    print(f"   📊 confusion_matrix_enhanced_{SELECTED_BACKBONE}.png")
    print(f"   📈 similarity_analysis_{SELECTED_BACKBONE}.png") 
    print(f"   📋 classification_report_detailed_{SELECTED_BACKBONE}.png")
    print(f"   💾 best_improved_siamese_{SELECTED_BACKBONE}.pth")
    print("🎯" + "="*78 + "🎯\n")

# MAIN MELHORADA
if __name__ == "__main__":
 
    print("=" * 80)
    print("🚀 TREINAMENTO SIAMÊS MELHORADO - VERSÃO APRIMORADA")
    print("=" * 80)
    
    print(f"Melhorias implementadas:")
    print(f"  - Backbone: {SELECTED_BACKBONE}")
    print(f"  - Batch Size: {BATCH_SIZE} (aumentado)")
    print(f"  - Épocas: {EPOCHS} (aumentado)")
    print(f"  - Learning Rate: {INITIAL_LR} (otimizado)")
    print(f"  - Arquitetura: Múltiplas métricas de comparação")
    print(f"  - Features: Mel + MFCC + Spectral Contrast")
    print(f"  - Otimizador: LRs diferenciados + warm-up")
    print(f"  - Loss: Focal BCE Loss")
    print(f"  - Dispositivo: {DEVICE}")
    print(f"  - Augmentação: {'Habilitado' if USE_AUGMENTATION else 'Desabilitado'}")
    
    if USE_AUGMENTATION:
        print(f"  - Probabilidade de augmentação: {AUGMENTATION_PROBABILITY} (aumentada)")
    
    # DIVISÃO MELHORADA DOS DADOS
    print("\n" + "=" * 80)
    print("📊 DIVISÃO MELHORADA DOS DADOS (60/25/15)")
    print("=" * 80)
    
    # Descomente e adapte estas linhas com seus dados reais:
    combined_data = combine_datasets(native_audio, my_audio)
    
    train_data, val_data, test_data = split_data_balanced(
        combined_data,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        random_seed=RANDOM_SEED
    )
    
    analyze_balanced_split(train_data, val_data, test_data)
    
    # CRIAÇÃO DOS DATASETS MELHORADOS
    print("\n" + "=" * 80)
    print("🔄 CRIANDO DATASETS MELHORADOS")
    print("=" * 80)
    
    processor = ImprovedAudioProcessor(use_augmentation=False)
    
    train_dataset = ImprovedSiameseAudioDataset(
        train_data,
        mode='train',
        use_augmentation=USE_AUGMENTATION
    )
    
    val_dataset = ImprovedSiameseAudioDataset(
        val_data,
        mode='val',
        use_augmentation=False
    )
    
    # DataLoaders otimizados
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    print(f"Dataset de treino: {len(train_dataset)} pares")
    print(f"Dataset de validação: {len(val_dataset)} pares")
    
    # INICIALIZAÇÃO DO MODELO MELHORADO
    print("\n" + "=" * 80)
    print("🧠 INICIALIZANDO MODELO MELHORADO")
    print("=" * 80)
    
    model = ImprovedSiameseNetwork(SELECTED_BACKBONE).to(DEVICE)
    
    # Focal BCE Loss
    criterion = FocalBCELoss(alpha=1, gamma=1.5)
    
    # Otimizador melhorado com LRs diferenciados
    optimizer, scheduler = create_improved_optimizer_and_scheduler(model)
    
    # Early stopping mais paciente
    early_stopping = EarlyStopping(
        patience=7,
        min_delta=0.001,
        restore_best_weights=True
    )
    
    # TREINAMENTO MELHORADO
    print("\n" + "=" * 80)
    print("🚀 INICIANDO TREINAMENTO MELHORADO")
    print("=" * 80)
    
    # Descomente para treinar com dados reais:
    train_losses, val_accuracies = train_improved_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        epochs=EPOCHS,
        log_dir="runs"
    )
    
    # DASHBOARD DE TREINAMENTO INTERATIVO

    print("\n" + "=" * 80)
    print("📈 GERANDO DASHBOARD DE TREINAMENTO")
    print("=" * 80)
    
    create_training_dashboard(train_losses, val_accuracies)
    
    # CARREGAMENTO DO MELHOR MODELO
    checkpoint_info = None
    try:
        checkpoint = torch.load(f"best_improved_siamese_{SELECTED_BACKBONE}.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint_info = checkpoint
        print(f"✅ Modelo carregado!")
        print(f"   - Melhor F1 de validação: {checkpoint.get('val_f1', 'N/A'):.4f}")
        print(f"   - Melhor Accuracy de validação: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"   - Época: {checkpoint.get('epoch', 'N/A')}")
    except FileNotFoundError:
        print("⚠️  Arquivo do melhor modelo não encontrado. Usando modelo atual.")
    
    # AVALIAÇÃO FINAL
    print("\n" + "=" * 80)
    print("🎯 INICIANDO AVALIAÇÃO FINAL")
    print("=" * 80)
    
    evaluation_results = evaluate_model_improved(model, test_data, train_data, processor)
    
    if evaluation_results:
        y_true = evaluation_results['true_labels'] 
        y_pred = evaluation_results['predictions']
        classes = sorted(set(y_true + y_pred))
        
        # VISUALIZAÇÕES DETALHADAS
        print("\n" + "=" * 80)
        print("📊 GERANDO VISUALIZAÇÕES DETALHADAS")
        print("=" * 80)
        
        # 1. Matriz de confusão melhorada
        print("📋 Criando matriz de confusão detalhada...")
        create_confusion_matrix_enhanced(y_true, y_pred, classes)
        
        # 2. Relatório de classificação visual
        print("📈 Gerando relatório de classificação...")
        create_detailed_classification_report(y_true, y_pred, classes)
        
        # 3. Análise de similaridades
        print("🔍 Analisando distribuições de similaridade...")
        create_similarity_analysis(evaluation_results)
        
        # 4. Dashboard de performance geral
        print("🎯 Criando dashboard de performance...")
        create_performance_summary_dashboard(evaluation_results, checkpoint_info)
        
        # RESUMO FINAL DETALHADO E VISUAL
        print("\n" + "=" * 80)
        print("🎊 RELATÓRIO FINAL COMPLETO")
        print("=" * 80)
        
        print_final_summary(evaluation_results, checkpoint_info)
        
        # Métricas adicionais para notebook
        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        random_baseline = 1.0 / len(set(y_true))
        improvement = (test_acc - random_baseline) / random_baseline * 100
        
        # Resultado final para análise programática
        print(f"\n📊 MÉTRICAS PARA ANÁLISE PROGRAMÁTICA:")
        print(f"test_accuracy = {test_acc:.4f}")
        print(f"test_f1_score = {test_f1:.4f}") 
        print(f"improvement_over_random = {improvement:.1f}%")
        print(f"total_classes = {len(set(y_true))}")
        print(f"total_test_samples = {len(y_true)}")
        
        # Salva resultados em formato legível para análise posterior
        results_summary = {
            'model_config': {
                'backbone': SELECTED_BACKBONE,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'learning_rate': INITIAL_LR,
                'augmentation': USE_AUGMENTATION
            },
            'performance': {
                'accuracy': test_acc,
                'f1_score': test_f1,
                'improvement': improvement,
                'baseline': random_baseline
            },
            'dataset_info': {
                'total_classes': len(set(y_true)),
                'test_samples': len(y_true),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset)
            }
        }
        
        # Salva em arquivo JSON para análise posterior
        import json
        with open(f'results_summary_{SELECTED_BACKBONE}.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n💾 Resultados salvos em: results_summary_{SELECTED_BACKBONE}.json")
        
    else:
        print("❌ Nenhuma predição foi feita com sucesso.")
        print("🔧 Verifique os dados de teste e referência.")
    
    # RESUMO DAS MELHORIAS IMPLEMENTADAS  
    print("\n" + "=" * 80)
    print("🎉 RESUMO DAS MELHORIAS E VISUALIZAÇÕES")
    print("=" * 80)
    
    print("📈 Visualizações criadas:")
    print("  1. 📊 Dashboard interativo de treinamento (Plotly)")
    print("  2. 🔍 Matriz de confusão detalhada (4 visualizações)")
    print("  3. 📋 Relatório de classificação visual (métricas por classe)")
    print("  4. 🎯 Análise de similaridades (distribuições e estatísticas)")
    print("  5. 🏆 Dashboard de performance geral")
    print("  6. 📄 Resumo final colorido e detalhado")
    
    print("\n📁 Arquivos gerados:")
    print(f"  - confusion_matrix_enhanced_{SELECTED_BACKBONE}.png")
    print(f"  - similarity_analysis_{SELECTED_BACKBONE}.png")
    print(f"  - classification_report_detailed_{SELECTED_BACKBONE}.png")
    print(f"  - results_summary_{SELECTED_BACKBONE}.json")
    print(f"  - best_improved_siamese_{SELECTED_BACKBONE}.pth")
    
    print("\n🎯 Para análise interativa adicional:")
    print("  - Use tensorboard --logdir runs para ver métricas de treinamento")
    print("  - Dashboards Plotly são interativos no notebook")
    print("  - Arquivo JSON contém métricas para análise programática")
    
    print("\n" + "=" * 80)