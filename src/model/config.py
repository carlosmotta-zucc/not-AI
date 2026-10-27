"""Hiperparâmetros do GPT num lugar só (Sprint 4, capítulo 4).

O livro passa um dicionário `cfg` para cada classe. Aqui ele vira uma
dataclass congelada, por três motivos:

- Os nomes dos campos são os do `GPT_CONFIG_124M` do livro (`vocab_size`,
  `context_length`, `emb_dim`, `n_heads`, `n_layers`, `drop_rate`,
  `qkv_bias`), então o capítulo e o glossário continuam legíveis lado a lado
  com o código.
- Validação na construção. `emb_dim` indivisível por `n_heads` falha aqui, onde
  o hiperparâmetro foi escolhido, e não no meio do forward.
- `dataclasses.replace(BASE_CONFIG, n_layers=8)` é como os experimentos variam
  um hiperparâmetro de cada vez, sem copiar dicionário à mão nem esquecer chave.

Três acréscimos ao dicionário do livro, todos com padrão que reproduz o livro:

- `ff_hidden_dim`: o livro fixa `4 * emb_dim` dentro da FeedForward. Aqui é
  hiperparâmetro, porque o Entregável 5 pede variar a dimensão da FeedForward.
- `drop_rate_emb`, `drop_rate_attn`, `drop_rate_shortcut`: o exercício 4.3.
  `None` herda o `drop_rate` global, então quem não mexe neles tem o
  comportamento do livro.
- `tie_weights`: weight tying (seção 4.6). `False`, como no `GPTModel` do livro.
"""

from dataclasses import dataclass, replace

from src.attention import head_dimension
from src.transformer import EXPANSION_FACTOR

# Vocabulário do BPE do GPT-2. O projeto não escreve vocab_size como literal
# (ver src/embeddings/token_embeddings.py); a constante existe porque a
# configuração não pode depender de carregar o tiktoken na importação, e o
# teste tests/test_gpt_model.py confere que ela bate com len(BPETokenizer()).
GPT2_VOCAB_SIZE = 50_257


@dataclass(frozen=True)
class GPTConfig:
    """Configuração completa de um GPTModel.

    Attributes:
        vocab_size: linhas da tabela de tokens e colunas dos logits.
        context_length: maior sequência aceita; define a tabela de posições e
            a máscara causal.
        emb_dim: dimensão do fluxo residual, mantida do primeiro ao último bloco.
        n_heads: cabeças de atenção por bloco.
        n_layers: quantos TransformerBlocks empilhados. Zero é aceito e dá o
            esqueleto do livro (DummyGPTModel): embeddings direto na saída.
        drop_rate: taxa de dropout global.
        qkv_bias: bias nas projeções Q, K e V. False como no GPT-2.
        ff_hidden_dim: camada escondida da FeedForward; None = 4 * emb_dim.
        drop_rate_emb: dropout depois dos embeddings; None = drop_rate.
        drop_rate_attn: dropout nos pesos de atenção; None = drop_rate.
        drop_rate_shortcut: dropout antes de cada soma residual; None = drop_rate.
        tie_weights: reaproveita a tabela de tokens como cabeça de saída.
    """

    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float = 0.0
    qkv_bias: bool = False
    ff_hidden_dim: int | None = None
    drop_rate_emb: float | None = None
    drop_rate_attn: float | None = None
    drop_rate_shortcut: float | None = None
    tie_weights: bool = False

    def __post_init__(self) -> None:
        """Rejeita combinações que só falhariam mais tarde, longe da causa.

        Raises:
            ValueError: dimensão não positiva, emb_dim indivisível por n_heads
                ou taxa de dropout fora de [0, 1).
        """
        positivos = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "emb_dim": self.emb_dim,
            "n_heads": self.n_heads,
            "feed_forward_dim": self.feed_forward_dim,
        }
        for nome, valor in positivos.items():
            if valor < 1:
                raise ValueError(f"{nome}={valor} precisa ser positivo")

        if self.n_layers < 0:
            raise ValueError(f"n_layers={self.n_layers} não pode ser negativo")

        # Mesma mensagem da Multi-Head da Sprint 3, escrita uma vez só lá.
        head_dimension(self.emb_dim, self.n_heads)

        taxas = {
            "drop_rate": self.drop_rate,
            "drop_rate_emb": self.emb_drop_rate,
            "drop_rate_attn": self.attn_drop_rate,
            "drop_rate_shortcut": self.shortcut_drop_rate,
        }
        for nome, taxa in taxas.items():
            # p = 1 zeraria tudo e o fator 1/(1-p) dividiria por zero.
            if not 0.0 <= taxa < 1.0:
                raise ValueError(f"{nome}={taxa} fora de [0, 1)")

    @property
    def head_dim(self) -> int:
        """Dimensão de cada cabeça: emb_dim // n_heads."""
        return self.emb_dim // self.n_heads

    @property
    def feed_forward_dim(self) -> int:
        """Camada escondida da FeedForward, já resolvido o padrão de 4x."""
        if self.ff_hidden_dim is None:
            return EXPANSION_FACTOR * self.emb_dim
        return self.ff_hidden_dim

    @property
    def emb_drop_rate(self) -> float:
        return self.drop_rate if self.drop_rate_emb is None else self.drop_rate_emb

    @property
    def attn_drop_rate(self) -> float:
        return self.drop_rate if self.drop_rate_attn is None else self.drop_rate_attn

    @property
    def shortcut_drop_rate(self) -> float:
        if self.drop_rate_shortcut is None:
            return self.drop_rate
        return self.drop_rate_shortcut


# O GPT-2 small, com os valores da seção 4.1 do livro.
GPT_CONFIG_124M = GPTConfig(
    vocab_size=GPT2_VOCAB_SIZE,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=False,
)

# Exercício 4.2: os quatro tamanhos do GPT-2 diferem só nestes três campos. O
# nome é o da OpenAI; o número de parâmetros real sai do modelo, não daqui.
GPT2_CONFIGS = {
    "gpt2-small": GPT_CONFIG_124M,
    "gpt2-medium": replace(GPT_CONFIG_124M, emb_dim=1024, n_layers=24, n_heads=16),
    "gpt2-large": replace(GPT_CONFIG_124M, emb_dim=1280, n_layers=36, n_heads=20),
    "gpt2-xl": replace(GPT_CONFIG_124M, emb_dim=1600, n_layers=48, n_heads=25),
}

# Configuração base do projeto: a das Sprints 2 e 3 (emb_dim 256, context_length
# 128, 4 cabeças de 64, dropout 0.1) mais 4 blocos. É o modelo que a Sprint 5
# vai treinar em CPU; o GPT_CONFIG_124M fica para reproduzir o livro.
BASE_CONFIG = GPTConfig(
    vocab_size=GPT2_VOCAB_SIZE,
    context_length=128,
    emb_dim=256,
    n_heads=4,
    n_layers=4,
    drop_rate=0.1,
    qkv_bias=False,
)
