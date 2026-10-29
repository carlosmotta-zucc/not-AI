"""Valores impressos no livro, com a página de cada um.

Fonte única para tests/test_book_checkpoints.py (que afirma que o projeto
reproduz cada número) e para experimentos/sprint04/livro_exemplos.py (que
publica a comparação lado a lado). Só dados: nenhuma conta aqui.

Referência: RASCHKA, S. Build a Large Language Model (From Scratch). Manning,
2024. Capítulo 4 inteiro, seção 5.3 e apêndice C (exercícios 4.1, 4.2 e 5.1).

Os logits do livro são impressos truncados (três primeiros e três últimos
valores de cada linha, quatro casas). É o que se compara aqui, com tolerância
de meia unidade na quarta casa.
"""

# Seed usada em todos os exemplos do capítulo 4 e da seção 5.3.
BOOK_SEED = 123

# Tolerância para valores impressos com 4 casas decimais.
PRINT_TOLERANCE = 5e-5

# --- Seção 4.1, p. 97-99: o lote de entrada e o esqueleto (DummyGPTModel) ---

BATCH_TEXTS = ("Every effort moves you", "Every day holds a")
BATCH_IDS = [[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]

# (três primeiros, três últimos) de cada uma das 2 x 4 linhas de logits.
DUMMY_LOGITS = [
    [
        ([-1.2034, 0.3201, -0.7130], [-1.5548, -0.2390, -0.4667]),
        ([-0.1192, 0.4539, -0.4432], [0.2392, 1.3469, 1.2430]),
        ([0.5307, 1.6720, -0.4695], [1.1966, 0.0111, 0.5835]),
        ([0.0139, 1.6755, -0.3388], [1.1586, -0.0435, -1.0400]),
    ],
    [
        ([-1.0908, 0.1798, -0.9484], [-1.6047, 0.2439, -0.4530]),
        ([-0.7860, 0.5581, -0.0610], [0.4835, -0.0077, 1.6621]),
        ([0.3567, 1.2698, -0.6398], [-0.0162, -0.1296, 0.3717]),
        ([-0.2407, -0.7349, -0.5102], [2.0057, -0.3694, 0.1814]),
    ],
]

# --- Seção 4.2, p. 100-104: LayerNorm passo a passo ---

# Saída de nn.Sequential(nn.Linear(5, 6), nn.ReLU()) sobre torch.randn(2, 5).
LAYER_OUTPUT = [
    [0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
    [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000],
]
# Média e variância por linha. A variância do livro aqui é a padrão do
# PyTorch (n - 1), não a enviesada da classe LayerNorm.
LAYER_OUTPUT_MEAN = [0.1324, 0.2170]
LAYER_OUTPUT_VAR = [0.0231, 0.0398]
# (out - mean) / sqrt(var), com aquela mesma variância.
LAYER_OUTPUT_NORMALIZED = [
    [0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719],
    [-0.0189, 0.1121, -1.0876, 1.5173, 0.5647, -1.0876],
]

# --- Seção 4.3, p. 106: o mínimo da GELU ---

# "nonzero gradient for almost all negative values (except at approximately
# x = -0.75)": o ponto em que a curva para de descer.
GELU_MINIMUM_X = -0.75

# --- Seção 4.4, p. 112-113: gradiente médio absoluto por camada ---

# ExampleDeepNeuralNetwork, layer_sizes [3, 3, 3, 3, 3, 1], entrada
# [[1., 0., -1.]], MSE contra 0. Da camada 0 (primeira) à 4 (última).
SHORTCUT_LAYER_SIZES = [3, 3, 3, 3, 3, 1]
SHORTCUT_INPUT = [[1.0, 0.0, -1.0]]
GRADIENTS_WITHOUT_SHORTCUT = [
    0.00020173587836325169,
    0.0001201116101583466,
    0.0007152041653171182,
    0.001398873864673078,
    0.005049646366387606,
]
GRADIENTS_WITH_SHORTCUT = [
    0.22169792652130127,
    0.20694105327129364,
    0.32896995544433594,
    0.2665732502937317,
    1.3258541822433472,
]

# --- Seção 4.5, p. 116 e exercício 4.1 (apêndice C, p. 302) ---

BLOCK_INPUT_SHAPE = (2, 4, 768)
FEED_FORWARD_PARAMS = 4_722_432
ATTENTION_PARAMS = 2_360_064

# --- Seção 4.6, p. 120-122: o GPTModel de 124M ---

GPT_LOGITS = [
    [
        ([0.3613, 0.4222, -0.0711], [0.3483, 0.4661, -0.2838]),
        ([-0.1792, -0.5660, -0.9485], [0.0477, 0.5181, -0.3168]),
        ([0.7120, 0.0332, 0.1085], [0.1018, -0.4327, -0.2553]),
        ([-1.0076, 0.3418, -0.1190], [0.7195, 0.4023, 0.0532]),
    ],
    [
        ([-0.2564, 0.0900, 0.0335], [0.2659, 0.4454, -0.6806]),
        ([0.1230, 0.3653, -0.2074], [0.7705, 0.2710, 0.2246]),
        ([1.0558, 1.0318, -0.2800], [0.6936, 0.3205, -0.3178]),
        ([-0.1565, 0.3926, 0.3288], [1.2630, -0.1858, 0.0388]),
    ],
]
GPT_124M_PARAMS = 163_009_536
GPT_124M_PARAMS_TIED = 124_412_160
GPT_124M_MEGABYTES = "621.83"
EMBEDDING_SHAPE = (50_257, 768)

# --- Exercício 4.2 (apêndice C, p. 302): o livro só imprime o XL ---

GPT2_XL_PARAMS = 1_637_792_000
GPT2_XL_PARAMS_TIED = 1_557_380_800
GPT2_XL_MEGABYTES = "6247.68"

# --- Seção 4.7, p. 125-127: geração gulosa sem treino ---

GENERATION_PROMPT = "Hello, I am"
GENERATION_PROMPT_IDS = [15496, 11, 314, 716]
GENERATION_NEW_TOKENS = 6
GENERATION_OUTPUT_IDS = [15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267]
GENERATION_OUTPUT_TEXT = "Hello, I am Featureiman Byeswickattribute argue"

# --- Seção 5.3, p. 152-157 e exercício 5.1 (p. 304): o vocabulário de 9 palavras ---

TOY_VOCAB = ["closer", "every", "effort", "forward", "inches", "moves", "pizza", "toward", "you"]
TOY_LOGITS = [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
TOY_GREEDY_WORD = "forward"
TOY_SAMPLES = 1_000
# Frequências de 1 000 sorteios com seed 123 (p. 153-154). "you" não aparece
# na listagem do livro porque o bincount para no maior ID sorteado (7).
TOY_SAMPLE_COUNTS = {
    "closer": 73,
    "every": 0,
    "effort": 0,
    "forward": 582,
    "inches": 2,
    "moves": 0,
    "pizza": 0,
    "toward": 343,
}
TOY_TEMPERATURES = (1.0, 0.1, 5.0)
# Exercício 5.1: "pizza" sorteado 32 vezes com T = 5; probabilidade real 4,3%.
TOY_PIZZA_COUNT_T5 = 32
TOY_PIZZA_PROBABILITY_T5 = 0.043
TOY_TOP_K = 3
TOY_TOP3_LOGITS = [6.75, 6.28, 4.51]
TOY_TOP3_POSITIONS = [3, 7, 0]
TOY_TOP3_PROBABILITIES = [0.0615, 0.0, 0.0, 0.5775, 0.0, 0.0, 0.0, 0.3610, 0.0]
