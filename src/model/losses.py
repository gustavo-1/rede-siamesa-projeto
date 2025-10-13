# FocalBCELoss
import torch
import torch.nn as nn

class FocalBCELoss(nn.Module):
    """Focal Loss para lidar com desbalanceamento de classes.
    
    Args:
        alpha (float): Fator de balanceamento para amostras positivas. Default: 1.
        gamma (float): Fator de modulação da perda para amostras fáceis vs difíceis. Default: 2.
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Previsões do modelo (probabilidades após sigmoid)
            targets (torch.Tensor): Alvos verdadeiros (0 ou 1)
            
        Returns:
            torch.Tensor: Valor da perda média por batch
        """
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # Probabilidade do alvo verdadeiro
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()