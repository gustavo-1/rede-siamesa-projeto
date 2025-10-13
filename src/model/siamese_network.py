
"""
Módulo que contém a implementação da rede siamesa melhorada.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import PretrainedBackbone, BACKBONE_OPTIONS
class ImprovedSiameseNetwork(nn.Module):
    def __init__(self, backbone_name='mobilenet_v3_small'):
        super(ImprovedSiameseNetwork, self).__init__()
        self.backbone = PretrainedBackbone(backbone_name)
        
        embedding_dim = BACKBONE_OPTIONS[backbone_name]['embedding_dim']
        
        # Extrator de features melhorado
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )
        
        # Camadas para comparação com múltiplas métricas
        self.comparison_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(512, 256), 
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True), 
                nn.BatchNorm1d(256),
                nn.Dropout(0.2)
            )
        ])
        
        # Fusão final
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x1, x2):
        # Extrai embeddings
        raw_emb1 = self.backbone(x1)
        raw_emb2 = self.backbone(x2)
        
        # Processa embeddings
        emb1 = self.feature_extractor(raw_emb1)
        emb2 = self.feature_extractor(raw_emb2)
        
        # Múltiplas métricas de comparação
        comparisons = []
        
        # 1. Diferença absoluta
        abs_diff = torch.abs(emb1 - emb2)
        comparisons.append(self.comparison_layers[0](abs_diff))
        
        # 2. Diferença quadrática
        sq_diff = torch.pow(emb1 - emb2, 2)
        comparisons.append(self.comparison_layers[1](sq_diff))
        
        # 3. Produto elemento a elemento
        element_prod = emb1 * emb2
        comparisons.append(self.comparison_layers[2](element_prod))
        
        # Fusão das comparações
        fused = torch.cat(comparisons, dim=1)
        output = self.fusion(fused)
        
        return output, emb1, emb2