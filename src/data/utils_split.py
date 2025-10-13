"""
Módulo que contém funções utilitárias para divisão e balanceamento dos dados.
"""

from collections import defaultdict
import random
import numpy as np
import audiomentations as A

def create_improved_audio_augmenter():
    """Cria o pipeline de aumento de dados para áudio - versão menos agressiva"""
    augmenter = A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.4),
        A.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),     # Menos agressivo
        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.4), # Menos agressivo
        A.TimeMask(min_band_part=0.0, max_band_part=0.05, p=0.2), # Máscaras menores
        A.Normalize(p=0.2),
    ])
    return augmenter

# Combina todos os dados em um dataset único
def combine_datasets(reference_data, test_data):
    """Combina reference_data e test_data em um único dataset"""
    combined_data = defaultdict(list)
   
    for word, files in reference_data.items():
        combined_data[word].extend(files)
   
    for word, files in test_data.items():
        if word in combined_data:
            combined_data[word].extend(files)
        else:
            combined_data[word] = files
   
    return dict(combined_data)

# Divisão balanceada dos dados
def split_data_balanced(combined_data, train_split=0.6, val_split=0.25, test_split=0.15,
                        random_seed=42):
    """
    Divisão balanceada com mais dados para treino
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
   
    train_data = {}
    val_data = {}
    test_data = {}
   
    print("Divisão balanceada melhorada por classe:")
    print("-" * 60)
    print(f"{'Classe':<15} {'Total':<6} {'Treino':<6} {'Val':<6} {'Teste':<6}")
    print("-" * 60)
   
    classes_with_few_files = 0
    total_val_pairs_possible = 0
   
    for word, files in combined_data.items():
        n_files = len(files)
       
        if n_files <= 3:
            if n_files == 3:
                train_size, val_size, test_size = 2, 1, 0  # Mais para treino
            elif n_files == 2:
                train_size, val_size, test_size = 1, 1, 0
                classes_with_few_files += 1
            else:
                train_size, val_size, test_size = 1, 0, 0
                classes_with_few_files += 1
        elif n_files == 4:
            train_size, val_size, test_size = 2, 1, 1  # Mais para treino
        elif n_files == 5:
            train_size, val_size, test_size = 3, 1, 1  # Mais para treino
        elif n_files == 6:
            train_size, val_size, test_size = 4, 1, 1  # Mais para treino
        else:  # 7 ou mais
            train_size = max(3, int(n_files * train_split))
            test_size = max(1, int(n_files * test_split))
            val_size = n_files - train_size - test_size
           
            # Garante pelo menos 1 para validação
            if val_size < 1:
                val_size = 1
                train_size = n_files - val_size - test_size
       
        # Embaralha os arquivos
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)
       
        # Faz as divisões
        train_data[word] = shuffled_files[:train_size]
        val_data[word] = shuffled_files[train_size:train_size + val_size]
        test_data[word] = shuffled_files[train_size + val_size:]
       
        # Conta pares positivos possíveis na validação
        if val_size >= 2:
            total_val_pairs_possible += val_size * (val_size - 1) // 2
       
        print(f"{word:<15} {n_files:<6} {train_size:<6} {val_size:<6} {test_size:<6}")
   
    print("-" * 60)
    print(f"Classes com poucos dados: {classes_with_few_files}")
    print(f"Pares positivos possíveis na validação: {total_val_pairs_possible}")
   
    return train_data, val_data, test_data

def analyze_balanced_split(train_data, val_data, test_data):
    """Analisa a qualidade da divisão balanceada"""
   
    total_train = sum(len(files) for files in train_data.values())
    total_val = sum(len(files) for files in val_data.values())
    total_test = sum(len(files) for files in test_data.values())
    total_files = total_train + total_val + total_test
   
    print(f"\nAnálise da Divisão Melhorada:")
    print(f"Total de arquivos: {total_files}")
    print(f"  - Treino: {total_train} ({total_train/total_files*100:.1f}%)")
    print(f"  - Validação: {total_val} ({total_val/total_files*100:.1f}%)")
    print(f"  - Teste: {total_test} ({total_test/total_files*100:.1f}%)")
   
    # Verifica classes com dados suficientes para validação
    classes_with_val_pairs = sum(1 for files in val_data.values() if len(files) >= 2)
    print(f"Classes com pares positivos na validação: {classes_with_val_pairs}/{len(val_data)}")
