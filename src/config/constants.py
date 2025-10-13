# Hiperparâmetros e configurações globais
import torch

# Configurações de hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parâmetros de processamento de áudio
SAMPLE_RATE = 32000  # Taxa de amostragem em Hz
N_MELS = 128  # Número de bandas mel - reduzido de 224 para menos features redundantes
HOP_LENGTH = 256  # Tamanho do hop - reduzido de 512 para mais resolução temporal
N_FFT = 1024  # Tamanho da FFT - reduzido proporcionalmente
MAX_TIME_STEPS = 128  # Duração máxima em frames - consistente com N_MELS

# Hiperparâmetros de treinamento
BATCH_SIZE = 16  # Tamanho do batch - aumentado de 8 para melhor gradiente
EPOCHS = 25  # Número de épocas - aumentado de 15 para melhor convergência
INITIAL_LR = 0.001  # Learning rate inicial - aumentado de 0.0003 para convergência mais rápida
WEIGHT_DECAY = 5e-4  # Regularização L2 - reduzido de 1e-3

# Divisão dos dados
TRAIN_SPLIT = 0.6  # Proporção para treino - aumentado de 0.5
VAL_SPLIT = 0.25  # Proporção para validação - reduzido de 0.3
TEST_SPLIT = 0.15  # Proporção para teste - reduzido de 0.2

# Configurações de reprodutibilidade
RANDOM_SEED = 42

# Configurações de data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.7  # Probabilidade de aplicar augmentation - aumentado de 0.5

# Configurações do modelo
# Opções de backbone disponíveis com suas dimensões de embedding
BACKBONE_OPTIONS = {
    'efficientnet_b0': {'model': 'efficientnet_b0', 'embedding_dim': 1280},
    'efficientnet_b1': {'model': 'efficientnet_b1', 'embedding_dim': 1280},
    'efficientnet_b2': {'model': 'efficientnet_b2', 'embedding_dim': 1408},
    'mobilenet_v3_small': {'model': 'mobilenet_v3_small', 'embedding_dim': 576},
    'mobilenet_v3_large': {'model': 'mobilenet_v3_large', 'embedding_dim': 960}
}

SELECTED_BACKBONE = 'mobilenet_v3_small'
EMBEDDING_DIM = BACKBONE_OPTIONS[SELECTED_BACKBONE]['embedding_dim']



native_audio = {
    # Yanomami
    'hãroaa':   ['audiotratado\\hãroaa1.wav', 'audiotratado\\hãroaa2.wav', 'audiotratado\\hãroaa3.wav'],
    'kõkaraxi': ['audiotratado\\kõkaraxi1.wav', 'audiotratado\\kõkaraxi2.wav', 'audiotratado\\kõkaraxi3.wav'],
    'oruheã':   ['audiotratado\\oruheã1.wav', 'audiotratado\\oruheã2.wav', 'audiotratado\\oruheã3.wav'],
    'pakokomaa':['audiotratado\\pakokomaa1.wav', 'audiotratado\\pakokomaa2.wav', 'audiotratado\\pakokomaa3.wav'],
    'wirimamo' : ['audiotratado\\wirimamo1.wav', 'audiotratado\\wirimamo2.wav', 'audiotratado\\wirimamo3.wav'],
    'tatãia'    : ['audiotratado\\tatãia1.wav', 'audiotratado\\tatãia2.wav', 'audiotratado\\tatãia3.wav'],
    'poromotëko': ['audiotratado\\poromotëko1.wav', 'audiotratado\\poromotëko2.wav', 'audiotratado\\poromotëko3.wav'],
    'ỹorimatë': ['audiotratado\\ỹorimatë1.wav', 'audiotratado\\ỹorimatë3.wav','audiotratado\\ỹorimatë4.wav'],
    'xamaa': ['audiotratado\\xamaa1.wav', 'audiotratado\\xamaa2.wav', 'audiotratado\\xamaa3.wav','audiotratado\\xamaa4.wav','audiotratado\\xamaa5.wav'],
    'hõwahõwamaa': ['audiotratado\\hõwahõwamaa1.wav', 'audiotratado\\hõwahõwamaa2.wav', 'audiotratado\\hõwahõwamaa3.wav','audiotratado\\hõwahõwamaa4.wav'],
    'kaỹaa': ['audiotratado\\kaỹaa1.wav', 'audiotratado\\kaỹaa2.wav', 'audiotratado\\kaỹaa3.wav','audiotratado\\kaỹaa4.wav'],
    'matõrõa': ['audiotratado\\matõrõa1.wav', 'audiotratado\\matõrõa2.wav', 'audiotratado\\matõrõa3.wav','audiotratado\\matõrõa4.wav'],
    'puhuruki': ['audiotratado\\puhuruki1.wav', 'audiotratado\\puhuruki2.wav', 'audiotratado\\puhuruki3.wav','audiotratado\\puhuruki4.wav'],
    'morõa': ['audiotratado\\morõa1.wav', 'audiotratado\\morõa2.wav', 'audiotratado\\morõa3.wav'],
    'waxarotëna': ['audiotratado\\waxarotëna1.wav', 'audiotratado\\waxarotëna2.wav', 'audiotratado\\waxarotëna3.wav', 'audiotratado\\waxarotëna4.wav'],
    'kaxuina': ['audiotratado\\kaxuina1.wav', 'audiotratado\\kaxuina2.wav', 'audiotratado\\kaxuina3.wav', 'audiotratado\\kaxuina4.wav'],
    'naroa': ['audiotratado\\naroa1.wav', 'audiotratado\\naroa2.wav', 'audiotratado\\naroa3.wav'],
    'nëxia': ['audiotratado\\nëxia1.wav', 'audiotratado\\nëxia2.wav', 'audiotratado\\nëxia3.wav', 'audiotratado\\nëxia4.wav'],
    'warapana': ['audiotratado\\warapana1.wav', 'audiotratado\\warapana2.wav', 'audiotratado\\warapana3.wav'],
    'iroa': ['audiotratado\\iroa1.wav', 'audiotratado\\iroa2.wav', 'audiotratado\\iroa3.wav'],
    'oaria': ['audiotratado\\oaria1.wav', 'audiotratado\\oaria2.wav', 'audiotratado\\oaria3.wav', 'audiotratado\\oaria4.wav'],
    'yaonaxi': ['audiotratado\\yaonaxi1.wav', 'audiotratado\\yaonaxi2.wav', 'audiotratado\\yaonaxi3.wav'],
    'ỹokoxitë': ['audiotratado\\ỹokoxitë1.wav', 'audiotratado\\ỹokoxitë2.wav', 'audiotratado\\ỹokoxitë3.wav', 'audiotratado\\ỹokoxitë4.wav'],
    'heraa': ['audiotratado\\heraa1.wav', 'audiotratado\\heraa2.wav', 'audiotratado\\heraa3.wav', 'audiotratado\\heraa4.wav'],
    'orixipërëhërimaki': ['audiotratado\\orixipërëhërimaki1.wav', 'audiotratado\\orixipërëhërimaki2.wav', 'audiotratado\\orixipërëhërimaki3.wav'],
    'kanaa': ['audiotratado\\kanaa1.wav', 'audiotratado\\kanaa2.wav', 'audiotratado\\kanaa3.wav' ],
    'paxoa': ['audiotratado\\paxoa1.wav', 'audiotratado\\paxoa2.wav', 'audiotratado\\paxoa3.wav', 'audiotratado\\paxoa4.wav'],
    'wakakuamo': ['audiotratado\\wakakuamo1.wav', 'audiotratado\\wakakuamo2.wav', 'audiotratado\\wakakuamo3.wav' ],
    'koxixi': ['audiotratado\\koxixi1.wav', 'audiotratado\\koxixi2.wav', 'audiotratado\\koxixi3.wav', 'audiotratado\\koxixi4.wav'],
    'ewëa': ['audiotratado\\ewëa1.wav', 'audiotratado\\ewëa2.wav', 'audiotratado\\ewëa3.wav', 'audiotratado\\ewëa4.wav'],
    'ewëpara': ['audiotratado\\ewëpara1.wav', 'audiotratado\\ewëpara2.wav', 'audiotratado\\ewëpara3.wav'],
    'riia': ['audiotratado\\riia1.wav', 'audiotratado\\riia2.wav', 'audiotratado\\riia3.wav','audiotratado\\riia4.wav' ],
    'riiuxirimaa': ['audiotratado\\riiuxirimaa1.wav', 'audiotratado\\riiuxirimaa2.wav', 'audiotratado\\riiuxirimaa3.wav'],
    'hopoa': ['audiotratado\\hopoa1.wav', 'audiotratado\\hopoa2.wav', 'audiotratado\\hopoa3.wav'],
    'amotaa' : ['audiotratado\\amotaa1.wav', 'audiotratado\\amotaa2.wav', 'audiotratado\\amotaa3.wav'],
    'poxihi': ['audiotratado\\poxihi1.wav', 'audiotratado\\poxihi2.wav', 'audiotratado\\poxihi3.wav' ],
    'yawerea': ['audiotratado\\yawerea1.wav', 'audiotratado\\yawerea2.wav', 'audiotratado\\yawerea3.wav','audiotratado\\yawerea4.wav' ],
    'yaruxea': ['audiotratado\\yaruxea1.wav', 'audiotratado\\yaruxea2.wav', 'audiotratado\\yaruxea3.wav' ,'audiotratado\\yaruxea4.wav' ],
    'wayapaxi': ['audiotratado\\wayapaxi1.wav', 'audiotratado\\wayapaxi2.wav', 'audiotratado\\wayapaxi3.wav' ],
    'warëa': ['audiotratado\\warëa1.wav', 'audiotratado\\warëa2.wav', 'audiotratado\\warëa3.wav','audiotratado\\warëa4.wav' ],
    'orokoxomaa': ['audiotratado\\orokoxomaa1.wav', 'audiotratado\\orokoxomaa2.wav', 'audiotratado\\orokoxomaa3.wav' ],
    'pãhomixipërimaa': ['audiotratado\\pãhomixipërimaa1.wav', 'audiotratado\\pãhomixipërimaa2.wav', 'audiotratado\\pãhomixipërimaa3.wav' ],
    'pãhoa': ['audiotratado\\pãhoa1.wav', 'audiotratado\\pãhoa2.wav', 'audiotratado\\pãhoa3.wav' ],
    'kõkaratëxi': ['audiotratado\\kõkaratëxi1.wav', 'audiotratado\\kõkaratëxi2.wav', 'audiotratado\\kõkaratëxi3.wav' ],
    'xuwëaỹokomaa': ['audiotratado\\xuwëaỹokomaa1.wav', 'audiotratado\\xuwëaỹokomaa2.wav', 'audiotratado\\xuwëaỹokomaa3.wav' ,'audiotratado\\xuwëaỹokomaa4.wav' ,   'audiotratado\\xuwëaỹokomaa5.wav',  'audiotratado\\xuwëaỹokomaa6.wav' ],
    'mokaa': ['audiotratado\\mokaa1.wav', 'audiotratado\\mokaa2.wav', 'audiotratado\\mokaa3.wav' ],
    'uweimaa': ['audiotratado\\uweimaa1.wav', 'audiotratado\\uweimaa2.wav', 'audiotratado\\uweimaa3.wav' ],
    'warëkiheã': ['audiotratado\\warëkiheã1.wav', 'audiotratado\\warëkiheã2.wav', 'audiotratado\\warëkiheã3.wav'  ],
    # Sanöma
    'kanakökö': ['audiotratado\\kanakökö1.wav', 'audiotratado\\kanakökö2.wav' ],
    'hãki': ['audiotratado\\hãki1.wav', 'audiotratado\\hãki2.wav' ],
    'ĩsããĩ': ['audiotratado\\ĩsããĩ1.wav', 'audiotratado\\ĩsããĩ2.wav' ],
    'ĩsããĩnoki': ['audiotratado\\ĩsããĩnoki1.wav', 'audiotratado\\ĩsããĩnoki2.wav' ],
    'humoösö': ['audiotratado\\humoösö1.wav', 'audiotratado\\humoösö2.wav' ],
    'õkomakö': ['audiotratado\\õkomakö1.wav', 'audiotratado\\õkomakö2.wav' ],
    'ãte': ['audiotratado\\ãte1.wav', 'audiotratado\\ãte2.wav' ],
    'pilitiso': ['audiotratado\\pilitiso1.wav', 'audiotratado\\pilitiso2.wav' ],
    'kutiataösö': ['audiotratado\\kutiataösö1.wav', 'audiotratado\\kutiataösö2.wav' ],
    'maopoö': ['audiotratado\\maopoö1.wav', 'audiotratado\\maopoö2.wav'],
    'makamitoto': ['audiotratado\\makamitoto1.wav', 'audiotratado\\makamitoto2.wav' ],
    'samatoto': ['audiotratado\\samatoto1.wav', 'audiotratado\\samatoto2.wav' ],
    'halahala': ['audiotratado\\halahala1.wav', 'audiotratado\\halahala2.wav' ],
    'mokotaa': ['audiotratado\\mokotaa1.wav', 'audiotratado\\mokotaa2.wav' ],
    'anepoko': ['audiotratado\\anepoko1.wav', 'audiotratado\\anepoko2.wav' ],
    'makaösö': ['audiotratado\\makaösö1.wav', 'audiotratado\\makaösö2.wav' ],
    'alaa': ['audiotratado\\alaa1.wav', 'audiotratado\\alaa2.wav' ],
    'makekö': ['audiotratado\\makekö1.wav', 'audiotratado\\makekö2.wav' ],
    'paa': ['audiotratado\\paa1.wav', 'audiotratado\\paa2.wav' ],
    'öpaa': ['audiotratado\\öpaa1.wav', 'audiotratado\\öpaa2.wav' ],
    'nomöhöki': ['audiotratado\\nomöhöki1.wav', 'audiotratado\\nomöhöki2.wav' ],
}

my_audio = {
    'hãroaa': [
        'audiotratado\\hãroaateste1.wav', 'audiotratado\\hãroaateste2.wav', 'audiotratado\\hãroaateste3.wav',
        'audiotratado\\hãroaatesteAutomatico1.wav', 'audiotratado\\hãroaatesteAutomatico2.wav', 'audiotratado\\hãroaatesteAutomatico3.wav', 'audiotratado\\hãroaatesteAutomatico4.wav', 'audiotratado\\hãroaatesteAutomatico5.wav'
    ],
    'kõkaraxi': [
        'audiotratado\\kõkaraxiteste1.wav', 'audiotratado\\kõkaraxiteste2.wav', 'audiotratado\\kõkaraxiteste3.wav',
        'audiotratado\\kõkaraxitesteAutomatico1.wav', 'audiotratado\\kõkaraxitesteAutomatico2.wav', 'audiotratado\\kõkaraxitesteAutomatico3.wav', 'audiotratado\\kõkaraxitesteAutomatico4.wav', 'audiotratado\\kõkaraxitesteAutomatico5.wav'
    ],
    'oruheã': [
        'audiotratado\\oruheãteste1.wav', 'audiotratado\\oruheãteste2.wav', 'audiotratado\\oruheãteste3.wav',
        'audiotratado\\oruheãtesteAutomatico1.wav', 'audiotratado\\oruheãtesteAutomatico2.wav', 'audiotratado\\oruheãtesteAutomatico3.wav', 'audiotratado\\oruheãtesteAutomatico4.wav', 'audiotratado\\oruheãtesteAutomatico5.wav'
    ],
    'pakokomaa': [
        'audiotratado\\pakokomaateste1.wav', 'audiotratado\\pakokomaateste2.wav', 'audiotratado\\pakokomaateste3.wav',
        'audiotratado\\pakokomaatesteAutomatico1.wav', 'audiotratado\\pakokomaatesteAutomatico2.wav', 'audiotratado\\pakokomaatesteAutomatico3.wav', 'audiotratado\\pakokomaatesteAutomatico4.wav', 'audiotratado\\pakokomaatesteAutomatico5.wav'
    ],
    'wirimamo': [
        'audiotratado\\wirimamoteste1.wav', 'audiotratado\\wirimamoteste2.wav', 'audiotratado\\wirimamoteste3.wav',
        'audiotratado\\wirimamotesteAutomatico1.wav', 'audiotratado\\wirimamotesteAutomatico2.wav', 'audiotratado\\wirimamotesteAutomatico3.wav', 'audiotratado\\wirimamotesteAutomatico4.wav', 'audiotratado\\wirimamotesteAutomatico5.wav'
    ],
    'tatãia': [
        'audiotratado\\tatãiateste1.wav', 'audiotratado\\tatãiateste2.wav', 'audiotratado\\tatãiateste3.wav',
        'audiotratado\\tatãiatesteAutomatico1.wav', 'audiotratado\\tatãiatesteAutomatico2.wav', 'audiotratado\\tatãiatesteAutomatico3.wav', 'audiotratado\\tatãiatesteAutomatico4.wav', 'audiotratado\\tatãiatesteAutomatico5.wav'
    ],
    'poromotëko': [
        'audiotratado\\poromotëkoteste1.wav', 'audiotratado\\poromotëkoteste2.wav', 'audiotratado\\poromotëkoteste3.wav',
        'audiotratado\\poromotëkotesteAutomatico1.wav', 'audiotratado\\poromotëkotesteAutomatico2.wav', 'audiotratado\\poromotëkotesteAutomatico3.wav', 'audiotratado\\poromotëkotesteAutomatico4.wav', 'audiotratado\\poromotëkotesteAutomatico5.wav'
    ],
    'ỹorimatë': [
        'audiotratado\\ỹorimatëteste1.wav', 'audiotratado\\ỹorimatëteste2.wav', 'audiotratado\\ỹorimatëteste3.wav',
        'audiotratado\\ỹorimatëtesteAutomatico1.wav', 'audiotratado\\ỹorimatëtesteAutomatico2.wav', 'audiotratado\\ỹorimatëtesteAutomatico3.wav', 'audiotratado\\ỹorimatëtesteAutomatico4.wav', 'audiotratado\\ỹorimatëtesteAutomatico5.wav'
    ],
    'xamaa': [
        'audiotratado\\xamaateste1.wav', 'audiotratado\\xamaateste2.wav', 'audiotratado\\xamaateste3.wav',
        'audiotratado\\xamaatesteAutomatico1.wav', 'audiotratado\\xamaatesteAutomatico2.wav', 'audiotratado\\xamaatesteAutomatico3.wav', 'audiotratado\\xamaatesteAutomatico4.wav', 'audiotratado\\xamaatesteAutomatico5.wav'
    ],
    'hõwahõwamaa': [
        'audiotratado\\hõwahõwamaateste1.wav', 'audiotratado\\hõwahõwamaateste2.wav', 'audiotratado\\hõwahõwamaateste3.wav',
        'audiotratado\\hõwahõwamaatesteAutomatico1.wav', 'audiotratado\\hõwahõwamaatesteAutomatico2.wav', 'audiotratado\\hõwahõwamaatesteAutomatico3.wav', 'audiotratado\\hõwahõwamaatesteAutomatico4.wav', 'audiotratado\\hõwahõwamaatesteAutomatico5.wav'
    ],
    'kaỹaa': [
        'audiotratado\\kaỹaateste1.wav', 'audiotratado\\kaỹaateste2.wav', 'audiotratado\\kaỹaateste3.wav',
        'audiotratado\\kaỹaatesteAutomatico1.wav', 'audiotratado\\kaỹaatesteAutomatico2.wav', 'audiotratado\\kaỹaatesteAutomatico3.wav', 'audiotratado\\kaỹaatesteAutomatico4.wav', 'audiotratado\\kaỹaatesteAutomatico5.wav'
    ],
    'matõrõa': [
        'audiotratado\\matõrõateste1.wav', 'audiotratado\\matõrõateste2.wav', 'audiotratado\\matõrõateste3.wav',
        'audiotratado\\matõrõatesteAutomatico1.wav', 'audiotratado\\matõrõatesteAutomatico2.wav', 'audiotratado\\matõrõatesteAutomatico3.wav', 'audiotratado\\matõrõatesteAutomatico4.wav', 'audiotratado\\matõrõatesteAutomatico5.wav'
    ],
    'puhuruki': [
        'audiotratado\\puhurukiteste1.wav', 'audiotratado\\puhurukiteste2.wav', 'audiotratado\\puhurukiteste3.wav',
        'audiotratado\\puhurukitesteAutomatico1.wav', 'audiotratado\\puhurukitesteAutomatico2.wav', 'audiotratado\\puhurukitesteAutomatico3.wav', 'audiotratado\\puhurukitesteAutomatico4.wav', 'audiotratado\\puhurukitesteAutomatico5.wav'
    ],
    'morõa': [
        'audiotratado\\morõateste1.wav', 'audiotratado\\morõateste2.wav', 'audiotratado\\morõateste3.wav',
        'audiotratado\\morõatesteAutomatico1.wav', 'audiotratado\\morõatesteAutomatico2.wav', 'audiotratado\\morõatesteAutomatico3.wav', 'audiotratado\\morõatesteAutomatico4.wav', 'audiotratado\\morõatesteAutomatico5.wav'
    ],
    'waxarotëna': [
        'audiotratado\\waxarotënateste1.wav', 'audiotratado\\waxarotënateste2.wav', 'audiotratado\\waxarotënateste3.wav',
        'audiotratado\\waxarotënatesteAutomatico1.wav', 'audiotratado\\waxarotënatesteAutomatico2.wav', 'audiotratado\\waxarotënatesteAutomatico3.wav', 'audiotratado\\waxarotënatesteAutomatico4.wav', 'audiotratado\\waxarotënatesteAutomatico5.wav'
    ],
    'kaxuina': [
        'audiotratado\\kaxuinateste1.wav', 'audiotratado\\kaxuinateste2.wav', 'audiotratado\\kaxuinateste3.wav',
        'audiotratado\\kaxuinatesteAutomatico1.wav', 'audiotratado\\kaxuinatesteAutomatico2.wav', 'audiotratado\\kaxuinatesteAutomatico3.wav', 'audiotratado\\kaxuinatesteAutomatico4.wav', 'audiotratado\\kaxuinatesteAutomatico5.wav'
    ],
    'naroa': [
        'audiotratado\\naroateste1.wav', 'audiotratado\\naroateste2.wav', 'audiotratado\\naroateste3.wav',
        'audiotratado\\naroatesteAutomatico1.wav', 'audiotratado\\naroatesteAutomatico2.wav', 'audiotratado\\naroatesteAutomatico3.wav', 'audiotratado\\naroatesteAutomatico4.wav', 'audiotratado\\naroatesteAutomatico5.wav'
    ],
    'nëxia': [
        'audiotratado\\nëxiateste1.wav', 'audiotratado\\nëxiateste2.wav', 'audiotratado\\nëxiateste3.wav',
        'audiotratado\\nëxiatesteAutomatico1.wav', 'audiotratado\\nëxiatesteAutomatico2.wav', 'audiotratado\\nëxiatesteAutomatico3.wav', 'audiotratado\\nëxiatesteAutomatico4.wav', 'audiotratado\\nëxiatesteAutomatico5.wav'
    ],
    'warapana': [
        'audiotratado\\warapanateste1.wav', 'audiotratado\\warapanateste2.wav', 'audiotratado\\warapanateste3.wav',
        'audiotratado\\warapanatesteAutomatico1.wav', 'audiotratado\\warapanatesteAutomatico2.wav', 'audiotratado\\warapanatesteAutomatico3.wav', 'audiotratado\\warapanatesteAutomatico4.wav', 'audiotratado\\warapanatesteAutomatico5.wav'
    ],
    'iroa': [
        'audiotratado\\iroateste1.wav', 'audiotratado\\iroateste2.wav', 'audiotratado\\iroateste3.wav',
        'audiotratado\\iroatesteAutomatico1.wav', 'audiotratado\\iroatesteAutomatico2.wav', 'audiotratado\\iroatesteAutomatico3.wav', 'audiotratado\\iroatesteAutomatico4.wav', 'audiotratado\\iroatesteAutomatico5.wav'
    ],
    'oaria': [
        'audiotratado\\oariateste1.wav', 'audiotratado\\oariateste2.wav', 'audiotratado\\oariateste3.wav',
        'audiotratado\\oariatesteAutomatico1.wav', 'audiotratado\\oariatesteAutomatico2.wav', 'audiotratado\\oariatesteAutomatico3.wav', 'audiotratado\\oariatesteAutomatico4.wav', 'audiotratado\\oariatesteAutomatico5.wav'
    ],
    'yaonaxi': [
        'audiotratado\\yaonaxiteste1.wav', 'audiotratado\\yaonaxiteste2.wav', 'audiotratado\\yaonaxiteste3.wav',
        'audiotratado\\yaonaxitesteAutomatico1.wav', 'audiotratado\\yaonaxitesteAutomatico2.wav', 'audiotratado\\yaonaxitesteAutomatico3.wav', 'audiotratado\\yaonaxitesteAutomatico4.wav', 'audiotratado\\yaonaxitesteAutomatico5.wav'
    ],
    'ỹokoxitë': [
        'audiotratado\\ỹokoxitëteste1.wav', 'audiotratado\\ỹokoxitëteste2.wav', 'audiotratado\\ỹokoxitëteste3.wav',
        'audiotratado\\ỹokoxitëtesteAutomatico1.wav', 'audiotratado\\ỹokoxitëtesteAutomatico2.wav', 'audiotratado\\ỹokoxitëtesteAutomatico3.wav', 'audiotratado\\ỹokoxitëtesteAutomatico4.wav', 'audiotratado\\ỹokoxitëtesteAutomatico5.wav'
    ],
    'heraa': [
        'audiotratado\\heraateste1.wav', 'audiotratado\\heraateste2.wav', 'audiotratado\\heraateste3.wav',
        'audiotratado\\heraatesteAutomatico1.wav', 'audiotratado\\heraatesteAutomatico2.wav', 'audiotratado\\heraatesteAutomatico3.wav', 'audiotratado\\heraatesteAutomatico4.wav', 'audiotratado\\heraatesteAutomatico5.wav', 'audiotratado\\heraatesteAutomatico6.wav'
    ],
    'orixipërëhërimaki': [
        'audiotratado\\orixipërëhërimakiteste1.wav', 'audiotratado\\orixipërëhërimakiteste2.wav', 'audiotratado\\orixipërëhërimakiteste3.wav',
        'audiotratado\\orixipërëhërimakitesteAutomatico1.wav', 'audiotratado\\orixipërëhërimakitesteAutomatico2.wav', 'audiotratado\\orixipërëhërimakitesteAutomatico3.wav', 'audiotratado\\orixipërëhërimakitesteAutomatico4.wav', 'audiotratado\\orixipërëhërimakitesteAutomatico5.wav'
    ],
    'kanaa': [
        'audiotratado\\kanaateste1.wav', 'audiotratado\\kanaateste2.wav', 'audiotratado\\kanaateste3.wav',
        'audiotratado\\kanaatesteAutomatico1.wav', 'audiotratado\\kanaatesteAutomatico2.wav', 'audiotratado\\kanaatesteAutomatico3.wav', 'audiotratado\\kanaatesteAutomatico4.wav', 'audiotratado\\kanaatesteAutomatico5.wav'
    ],
    'paxoa': [
        'audiotratado\\paxoateste1.wav', 'audiotratado\\paxoateste2.wav', 'audiotratado\\paxoateste3.wav',
        'audiotratado\\paxoatesteAutomatico1.wav', 'audiotratado\\paxoatesteAutomatico2.wav', 'audiotratado\\paxoatesteAutomatico3.wav', 'audiotratado\\paxoatesteAutomatico4.wav', 'audiotratado\\paxoatesteAutomatico5.wav'
    ],
    'wakakuamo': [
        'audiotratado\\wakakuamoteste1.wav', 'audiotratado\\wakakuamoteste2.wav', 'audiotratado\\wakakuamoteste3.wav',
        'audiotratado\\wakakuamotesteAutomatico1.wav', 'audiotratado\\wakakuamotesteAutomatico2.wav', 'audiotratado\\wakakuamotesteAutomatico3.wav', 'audiotratado\\wakakuamotesteAutomatico4.wav', 'audiotratado\\wakakuamotesteAutomatico5.wav'
    ],
    'koxixi': [
        'audiotratado\\koxixiteste1.wav', 'audiotratado\\koxixiteste2.wav', 'audiotratado\\koxixiteste3.wav',
        'audiotratado\\koxixitesteAutomatico1.wav', 'audiotratado\\koxixitesteAutomatico2.wav', 'audiotratado\\koxixitesteAutomatico3.wav', 'audiotratado\\koxixitesteAutomatico4.wav', 'audiotratado\\koxixitesteAutomatico5.wav'
    ],
    'ewëa': [
        'audiotratado\\ewëateste1.wav', 'audiotratado\\ewëateste2.wav', 'audiotratado\\ewëateste3.wav',
        'audiotratado\\ewëatesteAutomatico1.wav', 'audiotratado\\ewëatesteAutomatico2.wav', 'audiotratado\\ewëatesteAutomatico3.wav', 'audiotratado\\ewëatesteAutomatico4.wav', 'audiotratado\\ewëatesteAutomatico5.wav'
    ],
    'ewëpara': [
        'audiotratado\\ewëparateste1.wav', 'audiotratado\\ewëparateste2.wav', 'audiotratado\\ewëparateste3.wav',
        'audiotratado\\ewëparatesteAutomatico1.wav', 'audiotratado\\ewëparatesteAutomatico2.wav', 'audiotratado\\ewëparatesteAutomatico3.wav', 'audiotratado\\ewëparatesteAutomatico4.wav', 'audiotratado\\ewëparatesteAutomatico5.wav'
    ],
    'riia': [
        'audiotratado\\riiateste1.wav', 'audiotratado\\riiateste2.wav', 'audiotratado\\riiateste3.wav',
        'audiotratado\\riiatesteAutomatico1.wav', 'audiotratado\\riiatesteAutomatico2.wav', 'audiotratado\\riiatesteAutomatico3.wav', 'audiotratado\\riiatesteAutomatico4.wav', 'audiotratado\\riiatesteAutomatico5.wav'
    ],
    'riiuxirimaa': [
        'audiotratado\\riiuxirimaateste1.wav', 'audiotratado\\riiuxirimaateste2.wav', 'audiotratado\\riiuxirimaateste3.wav',
        'audiotratado\\riiuxirimaatesteAutomatico1.wav', 'audiotratado\\riiuxirimaatesteAutomatico2.wav', 'audiotratado\\riiuxirimaatesteAutomatico3.wav', 'audiotratado\\riiuxirimaatesteAutomatico4.wav', 'audiotratado\\riiuxirimaatesteAutomatico5.wav'
    ],
    'hopoa': [
        'audiotratado\\hopoateste1.wav', 'audiotratado\\hopoateste2.wav', 'audiotratado\\hopoateste3.wav',
        'audiotratado\\hopoatesteAutomatico1.wav', 'audiotratado\\hopoatesteAutomatico2.wav', 'audiotratado\\hopoatesteAutomatico3.wav', 'audiotratado\\hopoatesteAutomatico4.wav', 'audiotratado\\hopoatesteAutomatico5.wav'
    ],
    'amotaa': [
        'audiotratado\\amotaateste1.wav', 'audiotratado\\amotaateste2.wav', 'audiotratado\\amotaateste3.wav',
        'audiotratado\\amotaatesteAutomatico1.wav', 'audiotratado\\amotaatesteAutomatico2.wav', 'audiotratado\\amotaatesteAutomatico3.wav', 'audiotratado\\amotaatesteAutomatico4.wav', 'audiotratado\\amotaatesteAutomatico5.wav'
    ],
    'poxihi': [
        'audiotratado\\poxihiteste1.wav', 'audiotratado\\poxihiteste2.wav', 'audiotratado\\poxihiteste3.wav',
        'audiotratado\\poxihitesteAutomatico1.wav', 'audiotratado\\poxihitesteAutomatico2.wav', 'audiotratado\\poxihitesteAutomatico3.wav', 'audiotratado\\poxihitesteAutomatico4.wav', 'audiotratado\\poxihitesteAutomatico5.wav'
    ],
    'yawerea': [
        'audiotratado\\yawereateste1.wav', 'audiotratado\\yawereateste2.wav', 'audiotratado\\yawereateste3.wav',
        'audiotratado\\yawereatesteAutomatico1.wav', 'audiotratado\\yawereatesteAutomatico2.wav', 'audiotratado\\yawereatesteAutomatico3.wav', 'audiotratado\\yawereatesteAutomatico4.wav', 'audiotratado\\yawereatesteAutomatico5.wav'
    ],
    'yaruxea': [
        'audiotratado\\yaruxeateste1.wav', 'audiotratado\\yaruxeateste2.wav', 'audiotratado\\yaruxeateste3.wav',
        'audiotratado\\yaruxeatesteAutomatico1.wav', 'audiotratado\\yaruxeatesteAutomatico2.wav', 'audiotratado\\yaruxeatesteAutomatico3.wav', 'audiotratado\\yaruxeatesteAutomatico4.wav', 'audiotratado\\yaruxeatesteAutomatico5.wav'
    ],
    'wayapaxi': [
        'audiotratado\\wayapaxiteste1.wav', 'audiotratado\\wayapaxiteste2.wav', 'audiotratado\\wayapaxiteste3.wav',
        'audiotratado\\wayapaxitesteAutomatico1.wav', 'audiotratado\\wayapaxitesteAutomatico2.wav', 'audiotratado\\wayapaxitesteAutomatico3.wav', 'audiotratado\\wayapaxitesteAutomatico4.wav', 'audiotratado\\wayapaxitesteAutomatico5.wav'
    ],
    'warëa': [
        'audiotratado\\warëateste1.wav', 'audiotratado\\warëateste2.wav', 'audiotratado\\warëateste3.wav',
        'audiotratado\\warëatesteAutomatico1.wav', 'audiotratado\\warëatesteAutomatico2.wav', 'audiotratado\\warëatesteAutomatico3.wav', 'audiotratado\\warëatesteAutomatico4.wav', 'audiotratado\\warëatesteAutomatico5.wav'
    ],
    'orokoxomaa': [
        'audiotratado\\orokoxomaateste1.wav', 'audiotratado\\orokoxomaateste2.wav', 'audiotratado\\orokoxomaateste3.wav',
        'audiotratado\\orokoxomaatesteAutomatico1.wav', 'audiotratado\\orokoxomaatesteAutomatico2.wav', 'audiotratado\\orokoxomaatesteAutomatico3.wav', 'audiotratado\\orokoxomaatesteAutomatico4.wav', 'audiotratado\\orokoxomaatesteAutomatico5.wav'
    ],
    'pãhomixipërimaa': [
        'audiotratado\\pãhomixipërimaateste1.wav', 'audiotratado\\pãhomixipërimaateste2.wav', 'audiotratado\\pãhomixipërimaateste3.wav',
        'audiotratado\\pãhomixipërimaatesteAutomatico1.wav', 'audiotratado\\pãhomixipërimaatesteAutomatico2.wav', 'audiotratado\\pãhomixipërimaatesteAutomatico3.wav', 'audiotratado\\pãhomixipërimaatesteAutomatico4.wav', 'audiotratado\\pãhomixipërimaatesteAutomatico5.wav'
    ],
    'pãhoa': [
        'audiotratado\\pãhoateste1.wav', 'audiotratado\\pãhoateste2.wav', 'audiotratado\\pãhoateste3.wav',
        'audiotratado\\pãhoatesteAutomatico1.wav', 'audiotratado\\pãhoatesteAutomatico2.wav', 'audiotratado\\pãhoatesteAutomatico3.wav', 'audiotratado\\pãhoatesteAutomatico4.wav', 'audiotratado\\pãhoatesteAutomatico5.wav'
    ],
    'kõkaratëxi': [
        'audiotratado\\kõkaratëxiteste1.wav', 'audiotratado\\kõkaratëxiteste2.wav', 'audiotratado\\kõkaratëxiteste3.wav',
        'audiotratado\\kõkaratëxitesteAutomatico1.wav', 'audiotratado\\kõkaratëxitesteAutomatico2.wav', 'audiotratado\\kõkaratëxitesteAutomatico3.wav', 'audiotratado\\kõkaratëxitesteAutomatico4.wav', 'audiotratado\\kõkaratëxitesteAutomatico5.wav'
    ],
    'xuwëaỹokomaa': [
        'audiotratado\\xuwëaỹokomaateste1.wav', 'audiotratado\\xuwëaỹokomaateste2.wav', 'audiotratado\\xuwëaỹokomaateste3.wav',
        'audiotratado\\xuwëaỹokomaatesteAutomatico1.wav', 'audiotratado\\xuwëaỹokomaatesteAutomatico2.wav', 'audiotratado\\xuwëaỹokomaatesteAutomatico3.wav', 'audiotratado\\xuwëaỹokomaatesteAutomatico4.wav', 'audiotratado\\xuwëaỹokomaatesteAutomatico5.wav'
    ],
    'mokaa': [
        'audiotratado\\mokaateste1.wav', 'audiotratado\\mokaateste2.wav', 'audiotratado\\mokaateste3.wav',
        'audiotratado\\mokaatesteAutomatico1.wav', 'audiotratado\\mokaatesteAutomatico2.wav', 'audiotratado\\mokaatesteAutomatico3.wav', 'audiotratado\\mokaatesteAutomatico4.wav', 'audiotratado\\mokaatesteAutomatico5.wav'
    ],
    'uweimaa': [
        'audiotratado\\uweimaateste1.wav', 'audiotratado\\uweimaateste2.wav',
        'audiotratado\\uweimaatesteAutomatico1.wav', 'audiotratado\\uweimaatesteAutomatico2.wav', 'audiotratado\\uweimaatesteAutomatico3.wav', 'audiotratado\\uweimaatesteAutomatico4.wav', 'audiotratado\\uweimaatesteAutomatico5.wav', 'audiotratado\\uweimaatesteAutomatico6.wav'
    ],
    'warëkiheã': [
        'audiotratado\\warëkiheãteste1.wav', 'audiotratado\\warëkiheãteste2.wav', 'audiotratado\\warëkiheãteste3.wav',
        'audiotratado\\warëkiheãtesteAutomatico1.wav', 'audiotratado\\warëkiheãtesteAutomatico2.wav', 'audiotratado\\warëkiheãtesteAutomatico3.wav', 'audiotratado\\warëkiheãtesteAutomatico4.wav', 'audiotratado\\warëkiheãtesteAutomatico5.wav'
    ],
    'kanakökö': [
        'audiotratado\\kanakököteste1.wav', 'audiotratado\\kanakököteste2.wav', 'audiotratado\\kanakököteste3.wav',
        'audiotratado\\kanakökötesteAutomatico1.wav', 'audiotratado\\kanakökötesteAutomatico2.wav', 'audiotratado\\kanakökötesteAutomatico3.wav', 'audiotratado\\kanakökötesteAutomatico4.wav', 'audiotratado\\kanakökötesteAutomatico5.wav'
    ],
    'hãki': [
        'audiotratado\\hãkiteste1.wav', 'audiotratado\\hãkiteste2.wav', 'audiotratado\\hãkiteste3.wav',
        'audiotratado\\hãkitesteAutomatico1.wav', 'audiotratado\\hãkitesteAutomatico2.wav', 'audiotratado\\hãkitesteAutomatico3.wav', 'audiotratado\\hãkitesteAutomatico4.wav', 'audiotratado\\hãkitesteAutomatico5.wav'
    ],
    'ĩsããĩ': [
        'audiotratado\\ĩsããĩteste1.wav', 'audiotratado\\ĩsããĩteste2.wav', 'audiotratado\\ĩsããĩteste3.wav',
        'audiotratado\\ĩsããĩtesteAutomatico1.wav', 'audiotratado\\ĩsããĩtesteAutomatico2.wav', 'audiotratado\\ĩsããĩtesteAutomatico3.wav', 'audiotratado\\ĩsããĩtesteAutomatico4.wav', 'audiotratado\\ĩsããĩtesteAutomatico5.wav'
    ],
    'ĩsããĩnoki': [
        'audiotratado\\ĩsããĩnokiteste1.wav', 'audiotratado\\ĩsããĩnokiteste2.wav', 'audiotratado\\ĩsããĩnokiteste3.wav',
        'audiotratado\\ĩsããĩnokitesteAutomatico1.wav', 'audiotratado\\ĩsããĩnokitesteAutomatico2.wav', 'audiotratado\\ĩsããĩnokitesteAutomatico3.wav', 'audiotratado\\ĩsããĩnokitesteAutomatico4.wav', 'audiotratado\\ĩsããĩnokitesteAutomatico5.wav'
    ],
    'humoösö': [
        'audiotratado\\humoösöteste1.wav', 'audiotratado\\humoösöteste2.wav', 'audiotratado\\humoösöteste3.wav',
        'audiotratado\\humoösötesteAutomatico1.wav', 'audiotratado\\humoösötesteAutomatico2.wav', 'audiotratado\\humoösötesteAutomatico3.wav', 'audiotratado\\humoösötesteAutomatico4.wav', 'audiotratado\\humoösötesteAutomatico5.wav'
    ],
    'õkomakö': [
        'audiotratado\\õkomaköteste1.wav', 'audiotratado\\õkomaköteste2.wav', 'audiotratado\\õkomaköteste3.wav',
        'audiotratado\\õkomakötesteAutomatico1.wav', 'audiotratado\\õkomakötesteAutomatico2.wav', 'audiotratado\\õkomakötesteAutomatico3.wav', 'audiotratado\\õkomakötesteAutomatico4.wav', 'audiotratado\\õkomakötesteAutomatico5.wav'
    ],
    'ãte': [
        'audiotratado\\ãteteste1.wav', 'audiotratado\\ãteteste2.wav', 'audiotratado\\ãteteste3.wav',
        'audiotratado\\ãtetesteAutomatico1.wav', 'audiotratado\\ãtetesteAutomatico2.wav', 'audiotratado\\ãtetesteAutomatico3.wav', 'audiotratado\\ãtetesteAutomatico4.wav', 'audiotratado\\ãtetesteAutomatico5.wav'
    ],
    'pilitiso': [
        'audiotratado\\pilitisoteste1.wav', 'audiotratado\\pilitisoteste2.wav', 'audiotratado\\pilitisoteste3.wav',
        'audiotratado\\pilitisotesteAutomatico1.wav', 'audiotratado\\pilitisotesteAutomatico2.wav', 'audiotratado\\pilitisotesteAutomatico3.wav', 'audiotratado\\pilitisotesteAutomatico4.wav', 'audiotratado\\pilitisotesteAutomatico5.wav'
    ],
    'kutiataösö': [
        'audiotratado\\kutiataösöteste1.wav', 'audiotratado\\kutiataösöteste2.wav', 'audiotratado\\kutiataösöteste3.wav',
        'audiotratado\\kutiataösötesteAutomatico1.wav', 'audiotratado\\kutiataösötesteAutomatico2.wav', 'audiotratado\\kutiataösötesteAutomatico3.wav', 'audiotratado\\kutiataösötesteAutomatico4.wav', 'audiotratado\\kutiataösötesteAutomatico5.wav'
    ],
    'maopoö': [
        'audiotratado\\maopoöteste1.wav', 'audiotratado\\maopoöteste2.wav', 'audiotratado\\maopoöteste3.wav',
        'audiotratado\\maopoötesteAutomatico1.wav', 'audiotratado\\maopoötesteAutomatico2.wav', 'audiotratado\\maopoötesteAutomatico3.wav', 'audiotratado\\maopoötesteAutomatico4.wav', 'audiotratado\\maopoötesteAutomatico5.wav'
    ],
    'makamitoto': [
        'audiotratado\\makamitototeste1.wav', 'audiotratado\\makamitototeste2.wav', 'audiotratado\\makamitototeste3.wav',
        'audiotratado\\makamitototesteAutomatico1.wav', 'audiotratado\\makamitototesteAutomatico2.wav', 'audiotratado\\makamitototesteAutomatico3.wav', 'audiotratado\\makamitototesteAutomatico4.wav', 'audiotratado\\makamitototesteAutomatico5.wav'
    ],
    'samatoto': [
        'audiotratado\\samatototeste1.wav', 'audiotratado\\samatototeste2.wav', 'audiotratado\\samatototeste3.wav',
        'audiotratado\\samatototesteAutomatico1.wav', 'audiotratado\\samatototesteAutomatico2.wav', 'audiotratado\\samatototesteAutomatico3.wav', 'audiotratado\\samatototesteAutomatico4.wav', 'audiotratado\\samatototesteAutomatico5.wav'
    ],
    'halahala': [
        'audiotratado\\halahalateste1.wav', 'audiotratado\\halahalateste2.wav', 'audiotratado\\halahalateste3.wav',
        'audiotratado\\halahalatesteAutomatico1.wav', 'audiotratado\\halahalatesteAutomatico2.wav', 'audiotratado\\halahalatesteAutomatico3.wav', 'audiotratado\\halahalatesteAutomatico4.wav', 'audiotratado\\halahalatesteAutomatico5.wav'
    ],
    'mokotaa': [
        'audiotratado\\mokotaateste1.wav', 'audiotratado\\mokotaateste2.wav', 'audiotratado\\mokotaateste3.wav',
        'audiotratado\\mokotaatesteAutomatico1.wav', 'audiotratado\\mokotaatesteAutomatico2.wav', 'audiotratado\\mokotaatesteAutomatico3.wav', 'audiotratado\\mokotaatesteAutomatico4.wav', 'audiotratado\\mokotaatesteAutomatico5.wav'
    ],
    'anepoko': [
        'audiotratado\\anepokoteste1.wav', 'audiotratado\\anepokoteste2.wav', 'audiotratado\\anepokoteste3.wav',
        'audiotratado\\anepokotesteAutomatico1.wav', 'audiotratado\\anepokotesteAutomatico2.wav', 'audiotratado\\anepokotesteAutomatico3.wav', 'audiotratado\\anepokotesteAutomatico4.wav', 'audiotratado\\anepokotesteAutomatico5.wav'
    ],
    'makaösö': [
        'audiotratado\\makaösöteste1.wav', 'audiotratado\\makaösöteste2.wav', 'audiotratado\\makaösöteste3.wav',
        'audiotratado\\makaösötesteAutomatico1.wav', 'audiotratado\\makaösötesteAutomatico2.wav', 'audiotratado\\makaösötesteAutomatico3.wav', 'audiotratado\\makaösötesteAutomatico4.wav', 'audiotratado\\makaösötesteAutomatico5.wav'
    ],
    'alaa': [
        'audiotratado\\alaateste1.wav', 'audiotratado\\alaateste2.wav', 'audiotratado\\alaateste3.wav',
        'audiotratado\\alaatesteAutomatico1.wav', 'audiotratado\\alaatesteAutomatico2.wav', 'audiotratado\\alaatesteAutomatico3.wav', 'audiotratado\\alaatesteAutomatico4.wav', 'audiotratado\\alaatesteAutomatico5.wav'
    ],
    'makekö': [
        'audiotratado\\makeköteste1.wav', 'audiotratado\\makeköteste2.wav', 'audiotratado\\makeköteste3.wav',
        'audiotratado\\makekötesteAutomatico1.wav', 'audiotratado\\makekötesteAutomatico2.wav', 'audiotratado\\makekötesteAutomatico3.wav', 'audiotratado\\makekötesteAutomatico4.wav', 'audiotratado\\makekötesteAutomatico5.wav'
    ],
    'paa': [
        'audiotratado\\paateste1.wav', 'audiotratado\\paateste2.wav', 'audiotratado\\paateste3.wav',
        'audiotratado\\paatesteAutomatico1.wav', 'audiotratado\\paatesteAutomatico2.wav', 'audiotratado\\paatesteAutomatico3.wav', 'audiotratado\\paatesteAutomatico4.wav', 'audiotratado\\paatesteAutomatico5.wav'
    ],
    'öpaa': [
        'audiotratado\\öpaateste1.wav', 'audiotratado\\öpaateste2.wav', 'audiotratado\\öpaateste3.wav',
        'audiotratado\\öpaatesteAutomatico1.wav', 'audiotratado\\öpaatesteAutomatico2.wav', 'audiotratado\\öpaatesteAutomatico3.wav', 'audiotratado\\öpaatesteAutomatico4.wav', 'audiotratado\\öpaatesteAutomatico5.wav'
    ],
    'nomöhöki': [
        'audiotratado\\nomöhökiteste1.wav', 'audiotratado\\nomöhökiteste2.wav', 'audiotratado\\nomöhökiteste3.wav',
        'audiotratado\\nomöhökitesteAutomatico1.wav', 'audiotratado\\nomöhökitesteAutomatico2.wav', 'audiotratado\\nomöhökitesteAutomatico3.wav', 'audiotratado\\nomöhökitesteAutomatico4.wav', 'audiotratado\\nomöhökitesteAutomatico5.wav', 'audiotratado\\nomöhökitesteAutomatico6.wav'
    ]
}