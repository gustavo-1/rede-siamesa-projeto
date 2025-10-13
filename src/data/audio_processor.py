import numpy as np
import librosa
from src.config.constants import (
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, MAX_TIME_STEPS
)
from src.data.utils_split import create_improved_audio_augmenter

class ImprovedAudioProcessor:
    """Processador de áudio melhorado com múltiplas representações espectrais.
    
    Esta classe implementa um pipeline de processamento que:
    1. Carrega e normaliza o áudio
    2. Remove silêncios
    3. Opcionalmente aplica data augmentation
    4. Extrai características usando mel-spectrogram, MFCC e contraste espectral
    
    Args:
        use_augmentation (bool): Se True, inicializa o pipeline de augmentação. Default: False
    """
    def __init__(self, use_augmentation=False):
        self.use_augmentation = use_augmentation
        self.augmenter = create_improved_audio_augmenter() if use_augmentation else None
    
    def load_audio(self, file_path, apply_augmentation=False):
        """Carrega áudio com pré-processamento melhorado.
        
        Args:
            file_path (str): Caminho para o arquivo de áudio
            apply_augmentation (bool): Se True, aplica augmentação. Default: False
            
        Returns:
            numpy.ndarray: Forma de onda do áudio processada
        """
        waveform, sample_rate = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )
        
        # Normalização inicial
        waveform = librosa.util.normalize(waveform)
        
        # Remove silêncios nas bordas
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
        
        if apply_augmentation and self.augmenter is not None:
            try:
                waveform = self.augmenter(samples=waveform, sample_rate=SAMPLE_RATE)
            except Exception as e:
                print(f"Erro ao aplicar augmentação: {e}")
        
        return waveform
    
    @staticmethod
    def extract_features(waveform):
        """Extração de features melhorada com múltiplas representações espectrais.
        
        Args:
            waveform (numpy.ndarray): Forma de onda do áudio
            
        Returns:
            numpy.ndarray: Stack de características [3, N_MELS, MAX_TIME_STEPS]
                Canal 0: Mel-spectrogram
                Canal 1: MFCC
                Canal 2: Contraste espectral
        """
        # Mel-spectrogram principal
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            window='hann',
            fmin=0,
            fmax=SAMPLE_RATE//2
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC como canal adicional
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=SAMPLE_RATE,
            n_mfcc=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        
        # Spectral contrast como terceiro canal
        try:
            contrast = librosa.feature.spectral_contrast(
                y=waveform,
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
                n_bands=min(6, N_MELS//16)
            )
            # Repete para ter o mesmo tamanho
            if contrast.shape[0] < N_MELS:
                contrast = np.repeat(contrast, N_MELS//contrast.shape[0] + 1, axis=0)[:N_MELS]
        except:
            # Fallback se spectral contrast falhar
            contrast = np.zeros((N_MELS, mel_spec_db.shape[1]))
        
        # Normalização individual
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
        contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)
        
        # Ajusta dimensões temporais
        features = [mel_spec_db, mfcc, contrast]
        processed_features = []
        
        for feature in features:
            if feature.shape[1] < MAX_TIME_STEPS:
                pad_width = MAX_TIME_STEPS - feature.shape[1]
                feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                feature = feature[:, :MAX_TIME_STEPS]
            processed_features.append(feature)
        
        # Stack como canais RGB
        feature_stack = np.stack(processed_features, axis=0)
        
        return feature_stack
