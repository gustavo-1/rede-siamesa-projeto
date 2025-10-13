"""
Módulo que contém funções para classificação e avaliação do modelo.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

def classify_audio_improved(model, audio_tensor, processor, device):
    """
    Classifica um áudio usando o modelo Siamese melhorado.
    
    Args:
        model: Modelo Siamese treinado
        audio_tensor: Tensor do áudio a ser classificado
        processor: Processador de áudio para preparar os dados
        device: Dispositivo onde o modelo está (CPU/GPU)
    
    Returns:
        float: Probabilidade de similaridade
    """
    model.eval()
    with torch.no_grad():
        audio_tensor = audio_tensor.to(device)
        output = model(audio_tensor, audio_tensor)
        probability = torch.sigmoid(output).item()
    return probability

def evaluate_model_improved(model, test_loader, criterion, device):
    """
    Avalia o modelo usando o conjunto de teste.
    
    Args:
        model: Modelo Siamese treinado
        test_loader: DataLoader com dados de teste
        criterion: Função de perda
        device: Dispositivo onde o modelo está (CPU/GPU)
    
    Returns:
        dict: Métricas de avaliação (loss, accuracy, precision, recall, f1)
    """
    model.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Avaliando"):
            audio1, audio2, labels = batch
            audio1, audio2, labels = audio1.to(device), audio2.to(device), labels.to(device)
            
            outputs = model(audio1, audio2)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = {
        'loss': test_loss,
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions),
        'f1': f1_score(all_labels, all_predictions),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions)
    }
    
    return metrics