"""
Módulo que contém a implementação do backbone pré-treinado para a rede siamesa.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Opções de backbone
BACKBONE_OPTIONS = {
    'efficientnet_b0': {'model': 'efficientnet_b0', 'embedding_dim': 1280},
    'efficientnet_b1': {'model': 'efficientnet_b1', 'embedding_dim': 1280},
    'efficientnet_b2': {'model': 'efficientnet_b2', 'embedding_dim': 1408},
    'mobilenet_v3_small': {'model': 'mobilenet_v3_small', 'embedding_dim': 576},
    'mobilenet_v3_large': {'model': 'mobilenet_v3_large', 'embedding_dim': 960}
}

class PretrainedBackbone(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0'):
        super(PretrainedBackbone, self).__init__()
       
        self.backbone_name = backbone_name
       
        if 'efficientnet' in backbone_name:
            if backbone_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif backbone_name == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            elif backbone_name == 'efficientnet_b2':
                self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
           
            self.features = self.backbone.features
            self.avgpool = self.backbone.avgpool
           
        elif 'mobilenet_v3' in backbone_name:
            if backbone_name == 'mobilenet_v3_small':
                self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            elif backbone_name == 'mobilenet_v3_large':
                self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
           
            self.features = self.backbone.features
            self.avgpool = self.backbone.avgpool
       
        print(f"Backbone {backbone_name} carregado com pesos pré-treinados do ImageNet!")
   
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
