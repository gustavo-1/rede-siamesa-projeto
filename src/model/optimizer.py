import torch
"""
Este módulo implementa o otimizador e scheduler personalizados para o modelo siamês.
"""

import torch
from src.config.constants import INITIAL_LR, WEIGHT_DECAY

def create_improved_optimizer_and_scheduler(model):
    """Cria otimizador e scheduler otimizados para o modelo siamês.
    
    O otimizador usa learning rates diferentes para o backbone (pré-treinado) 
    e as camadas de cabeça (treinadas do zero). O scheduler implementa warm-up
    seguido de decay exponencial.
    
    Args:
        model (nn.Module): O modelo siamês para otimizar
        
    Returns:
        tuple: (optimizer, scheduler)
            - optimizer (torch.optim.AdamW): Otimizador configurado
            - scheduler (torch.optim.lr_scheduler.LambdaLR): Scheduler configurado
    """
    
    # Diferentes learning rates para diferentes partes
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': INITIAL_LR * 0.1},  # LR menor para backbone pré-treinado
        {'params': head_params, 'lr': INITIAL_LR}  # LR normal para cabeçalho
    ], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    
    # Scheduler com warm-up nos primeiros 3 epochs
    def lr_lambda(epoch):
        if epoch < 3:  # Warm-up
            return (epoch + 1) / 3
        else:
            return 0.95 ** (epoch - 2)  # Decay exponencial após warm-up
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler