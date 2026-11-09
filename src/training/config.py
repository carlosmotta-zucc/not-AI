"""Hiperparâmetros do treino num lugar só (Sprint 5, seções 5.1.3 e 5.2).

O Entregável 3 pede registrar os parâmetros de cada treino: épocas, tamanho do
lote, taxa de aprendizado, otimizador. Eles vivem numa dataclass congelada, no
mesmo molde do GPTConfig da Sprint 4:

- Os nomes seguem o livro (`num_epochs`, `eval_freq`, `eval_iter`,
  `start_context`), para o capítulo e o código continuarem legíveis lado a lado.
- Validação na construção: uma taxa de aprendizado negativa falha aqui, e não
  depois de dez minutos de treino.
- `dataclasses.replace(BASE_TRAINING_CONFIG, learning_rate=1e-3)` é como os
  experimentos variam um hiperparâmetro de cada vez.
- `dataclasses.asdict(cfg)` vira JSON ou tabela. GPTConfig + TrainingConfig
  identificam uma execução inteira: é o que precisa ir junto com os pesos
  salvos para alguém reproduzir o treino.

Os valores padrão são os do livro, como no GPTConfig: quem não mexe em nada
treina exatamente como o capítulo.
"""

from dataclasses import dataclass, replace

from src.model import GPT_CONFIG_124M

# O texto de partida das amostras do livro (seção 5.1.1).
BOOK_START_CONTEXT = "Every effort moves you"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuração completa de um treino.

    Attributes:
        num_epochs: passadas completas pelo conjunto de treino.
        batch_size: sequências por lote (o B dos shapes).
        learning_rate: passo do AdamW.
        weight_decay: decaimento de peso do AdamW (desacoplado do gradiente).
        eval_freq: a cada quantos passos de otimização as perdas são avaliadas.
        eval_iter: quantos lotes entram em cada avaliação; None = o conjunto
            inteiro. O livro usa 5: a perda de treino avaliada é uma estimativa
            sobre 5 lotes, não a perda do conjunto de treino.
        train_ratio: fração do texto, em caracteres, que vai para o treino; o
            resto é validação.
        stride: avanço da janela deslizante no TREINO; None = context_length
            (janelas sem sobreposição, como no livro). A validação usa sempre
            context_length (ver src/training/data.py).
        seed: semente global, aplicada antes de criar o modelo. Decide os pesos
            iniciais, a ordem dos lotes e as máscaras de dropout.
        start_context: texto de partida da amostra gerada no fim de cada época.
        sample_tokens: tokens gerados nessa amostra; 0 desliga a amostra.
    """

    num_epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 4e-4
    weight_decay: float = 0.1
    eval_freq: int = 5
    eval_iter: int | None = 5
    train_ratio: float = 0.9
    stride: int | None = None
    seed: int = 123
    start_context: str = BOOK_START_CONTEXT
    sample_tokens: int = 50

    def __post_init__(self) -> None:
        """Rejeita valores que só falhariam no meio do treino.

        Raises:
            ValueError: contagem não positiva, taxa fora do domínio ou fração de
                treino fora de (0, 1).
        """
        positivos = {
            "batch_size": self.batch_size,
            "eval_freq": self.eval_freq,
        }
        for nome, valor in positivos.items():
            if valor < 1:
                raise ValueError(f"{nome}={valor} precisa ser positivo")

        opcionais = {"eval_iter": self.eval_iter, "stride": self.stride}
        for nome, valor in opcionais.items():
            if valor is not None and valor < 1:
                raise ValueError(f"{nome}={valor} precisa ser positivo ou None")

        if self.num_epochs < 0:
            raise ValueError(f"num_epochs={self.num_epochs} não pode ser negativo")
        if self.sample_tokens < 0:
            raise ValueError(f"sample_tokens={self.sample_tokens} não pode ser negativo")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate={self.learning_rate} precisa ser positiva")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay={self.weight_decay} não pode ser negativo")
        if not 0.0 < self.train_ratio < 1.0:
            # 0 ou 1 deixariam um dos dois conjuntos vazio.
            raise ValueError(f"train_ratio={self.train_ratio} fora de (0, 1)")


# O treino da seção 5.2 (p. 143 e 149): AdamW com lr 0,0004 e weight decay 0,1,
# 10 épocas, lotes de 2, avaliação a cada 5 passos sobre 5 lotes.
BOOK_TRAINING_CONFIG = TrainingConfig()

# O modelo que o livro treina no capítulo 5: o GPT-2 small com o contexto
# reduzido de 1 024 para 256 tokens (p. 130), para caber num laptop.
BOOK_MODEL_CONFIG = replace(GPT_CONFIG_124M, context_length=256)

# Treino de referência do projeto, sobre o BASE_CONFIG da Sprint 4: os
# hiperparâmetros do livro, a seed dos experimentos do projeto (42) e
# avaliação sobre os conjuntos inteiros. Com o BASE_CONFIG, cada avaliação
# completa custa uma fração de segundo, e a perda de treino passa a ser a do
# conjunto de treino, não uma estimativa sobre 5 lotes sorteados -- o que a
# análise de sobreajuste precisa.
BASE_TRAINING_CONFIG = replace(BOOK_TRAINING_CONFIG, seed=42, eval_iter=None)
