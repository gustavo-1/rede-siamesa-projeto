"""
Módulo responsável pelo treinamento da rede siamesa.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from tqdm import tqdm
from src.config.constants import DEVICE

def train_improved_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                         early_stopping, epochs, log_dir="runs"):
    """Treina o modelo siamês com monitoramento detalhado e early stopping.
    
    Args:
        model (ImprovedSiameseNetwork): O modelo a ser treinado
        train_loader (DataLoader): DataLoader para dados de treino
        val_loader (DataLoader): DataLoader para dados de validação
        criterion: Função de perda (loss)
        optimizer: Otimizador configurado
        scheduler: Scheduler para learning rate
        early_stopping (EarlyStopping): Monitor de early stopping
        epochs (int): Número máximo de épocas
        log_dir (str): Diretório para logs do TensorBoard
        
    Returns:
        tuple: (train_losses, val_accuracies) histórico de métricas
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{log_dir}/improved_siamese_{model.backbone.backbone_name}_{timestamp}")
   
    train_losses = []
    val_accuracies = []
   
    print(f"Iniciando treinamento MELHORADO por {epochs} épocas...")
    print("-" * 80)
   
    for epoch in range(epochs):
        # TREINAMENTO
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
       
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Treino]")
        for batch_idx, (specs1, specs2, labels) in enumerate(train_pbar):
            specs1 = specs1.to(DEVICE)
            specs2 = specs2.to(DEVICE)
            labels = labels.to(DEVICE)
           
            optimizer.zero_grad()
           
            outputs, emb1, emb2 = model(specs1, specs2)
            loss = criterion(outputs, labels)
           
            loss.backward()
           
            # Gradient clipping para estabilidade
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
           
            optimizer.step()
           
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
           
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
       
        # Métricas de treino
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds, average='binary', zero_division=0)
        train_recall = recall_score(train_labels, train_preds, average='binary', zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, average='binary', zero_division=0)
       
        # VALIDAÇÃO
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_model_detailed(
            model, val_loader, criterion
        )
       
        # Step do scheduler
        scheduler.step()
       
        # Log no TensorBoard
        writer.add_scalars('Loss', {'Treino': train_loss, 'Validação': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Treino': train_acc, 'Validação': val_acc}, epoch)
        writer.add_scalars('F1-Score', {'Treino': train_f1, 'Validação': val_f1}, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
       
        # Armazena métricas
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
       
        # Print das métricas
        current_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else current_lr
        print(f"Época {epoch+1}/{epochs} - LR: {current_lr:.6f}/{head_lr:.6f}")
        print(f"Treino - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Valid  - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
       
        # Early stopping (usa F1 como métrica principal)
        if early_stopping(val_f1, model):
            print(f"Early stopping acionado na época {epoch+1}")
            break
       
        # Salva melhor modelo baseado em F1
        if len(val_accuracies) == 1 or val_f1 >= max([val_f1] + val_accuracies[:-1]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'backbone': model.backbone.backbone_name,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }, f"best_improved_siamese_{model.backbone.backbone_name}.pth")
       
        print("-" * 80)
   
    # Restaura melhores pesos se early stopping
    early_stopping.restore_weights(model)
   
    writer.close()
    print(f"Treinamento concluído!")
    print(f"TensorBoard: {log_dir}/improved_siamese_{model.backbone.backbone_name}_{timestamp}")
   
    return train_losses, val_accuracies

def validate_model_detailed(model, val_loader, criterion):
    """Valida o modelo e retorna métricas detalhadas.
    
    Args:
        model (ImprovedSiameseNetwork): Modelo a ser validado
        val_loader (DataLoader): DataLoader para dados de validação
        criterion: Função de perda (loss)
        
    Returns:
        tuple: (val_loss, val_acc, val_precision, val_recall, val_f1)
    """
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    val_similarities = []
   
    with torch.no_grad():
        for specs1, specs2, labels in val_loader:
            specs1 = specs1.to(DEVICE)
            specs2 = specs2.to(DEVICE)
            labels = labels.to(DEVICE)
           
            outputs, _, _ = model(specs1, specs2)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
           
            val_similarities.extend(outputs.cpu().numpy().flatten())
            val_preds.extend((outputs > 0.5).float().cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())
   
    val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, average='binary', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='binary', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='binary', zero_division=0)
   
    # Análise das similaridades
    similarities_array = torch.tensor(val_similarities)
    pos_similarities = [val_similarities[i] for i, label in enumerate(val_labels) if label == 1]
    neg_similarities = [val_similarities[i] for i, label in enumerate(val_labels) if label == 0]
   
    print(f"    Similaridades - Geral: [{similarities_array.min():.3f}, {similarities_array.max():.3f}], Média: {similarities_array.mean():.3f}")
    if pos_similarities:
        print(f"    Positivos: Média: {sum(pos_similarities)/len(pos_similarities):.3f}")
    if neg_similarities:
        print(f"    Negativos: Média: {sum(neg_similarities)/len(neg_similarities):.3f}")
   
    return val_loss, val_acc, val_precision, val_recall, val_f1
