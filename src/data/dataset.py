import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.data.audio_processor import ImprovedAudioProcessor
from src.config.constants import N_MELS, MAX_TIME_STEPS, AUGMENTATION_PROBABILITY

class ImprovedSiameseAudioDataset(Dataset):
    """Dataset para treinar a rede siamesa com pares de áudios.
    
    Args:
        data (dict): Dicionário com classes como chaves e listas de arquivos como valores
        mode (str): Modo de operação ('train', 'val', 'test'). Default: 'train'
        pairs_per_class (int, optional): Número de pares por classe. Se None, é calculado
        use_augmentation (bool): Se True, aplica augmentação em pares de treino. Default: False
    """
    def __init__(self, data, mode='train', pairs_per_class=None, use_augmentation=False):
        self.data = data
        self.mode = mode
        self.classes = list(data.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.use_augmentation = use_augmentation and mode == 'train'
        
        # Mais pares para melhor aprendizado
        if pairs_per_class is None:
            avg_files_per_class = np.mean([len(files) for files in data.values()])
            if avg_files_per_class < 3:
                self.pairs_per_class = 40  # Aumentado de 30
            else:
                self.pairs_per_class = min(120, int(avg_files_per_class * 20))  # Aumentado
        else:
            self.pairs_per_class = pairs_per_class
            
        self.pairs = self._generate_balanced_pairs()
        self.processor = ImprovedAudioProcessor(use_augmentation=self.use_augmentation)
        
        # Normalização padrão ImageNet (para compatibilidade com backbones pré-treinados)
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _generate_balanced_pairs(self):
        """Gera pares balanceados de amostras positivas e negativas.
        
        Returns:
            list: Lista de tuplas (arquivo1, arquivo2, label), onde label é 1 para pares
                 da mesma classe e 0 para pares de classes diferentes.
        """
        pairs = []
        positive_pairs = []
        
        # Gera mais pares positivos por classe
        for cls, files in self.data.items():
            n_files = len(files)
            if n_files >= 2:
                # Todos os pares possíveis
                for i in range(n_files):
                    for j in range(i + 1, n_files):
                        positive_pairs.append((files[i], files[j], 1))
                
                # Se poucos pares, aumenta com repetições (augmentação dupla)
                current_pairs = [p for p in positive_pairs if p[0] in files or p[1] in files]
                if len(current_pairs) < self.pairs_per_class // 2:
                    needed = (self.pairs_per_class // 2) - len(current_pairs)
                    for _ in range(needed):
                        if n_files >= 2:
                            file1, file2 = random.sample(files, 2)
                            positive_pairs.append((file1, file2, 1))
        
        pairs.extend(positive_pairs)
        
        # Gera pares negativos balanceados
        target_negative = len(positive_pairs)
        negative_pairs = []
        
        for _ in range(target_negative):
            cls1, cls2 = random.sample(self.classes, 2)
            if cls1 != cls2 and len(self.data[cls1]) > 0 and len(self.data[cls2]) > 0:
                file1 = random.choice(self.data[cls1])
                file2 = random.choice(self.data[cls2])
                negative_pairs.append((file1, file2, 0))
        
        pairs.extend(negative_pairs)
        random.shuffle(pairs)
        
        positive_count = sum(1 for p in pairs if p[2] == 1)
        print(f"Dataset {self.mode}: {positive_count} pos, {len(pairs)-positive_count} neg, {len(pairs)} total")
        
        return pairs
    
    def __len__(self):
        """Retorna o número total de pares no dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Retorna um par de espectrogramas e seu rótulo para o índice dado.
        
        Args:
            idx (int): Índice do par desejado
            
        Returns:
            tuple: (spec1, spec2, label) onde:
                spec1, spec2 (torch.Tensor): Espectrogramas normalizados [3, N_MELS, MAX_TIME_STEPS]
                label (torch.Tensor): Rótulo binário [1] (1 para mesmo locutor, 0 caso contrário)
        """
        file1, file2, label = self.pairs[idx]
        
        try:
            # Augmentação mais agressiva
            apply_aug1 = self.use_augmentation and random.random() < AUGMENTATION_PROBABILITY
            apply_aug2 = self.use_augmentation and random.random() < AUGMENTATION_PROBABILITY
            
            waveform1 = self.processor.load_audio(file1, apply_augmentation=apply_aug1)
            waveform2 = self.processor.load_audio(file2, apply_augmentation=apply_aug2)
            
            spec1 = self.processor.extract_features(waveform1)
            spec2 = self.processor.extract_features(waveform2)
            
            spec1 = torch.FloatTensor(spec1)
            spec2 = torch.FloatTensor(spec2)
            
            spec1 = self.transform(spec1)
            spec2 = self.transform(spec2)
            
            label = torch.FloatTensor([label])
            
            return spec1, spec2, label
            
        except Exception as e:
            print(f"Erro ao processar {file1}, {file2}: {e}")
            # Retorna tensores zerados em caso de erro
            spec1 = torch.zeros(3, N_MELS, MAX_TIME_STEPS)
            spec2 = torch.zeros(3, N_MELS, MAX_TIME_STEPS)
            spec1 = self.transform(spec1)
            spec2 = self.transform(spec2)
            label = torch.FloatTensor([0])
            return spec1, spec2, label
