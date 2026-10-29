"""GPTModel (seção 4.6): do Token ID aos logits do próximo token.

    (B, T) Token IDs
      -> InputEmbedding (Sprint 2)      tokens + posições       (B, T, emb_dim)
      -> Dropout                                                (B, T, emb_dim)
      -> n_layers x TransformerBlock    fluxo residual          (B, T, emb_dim)
      -> LayerNorm final                                        (B, T, emb_dim)
      -> cabeça de saída (Linear)       um logit por token      (B, T, vocab_size)

A posição t dos logits é a previsão do token t+1: com a máscara causal, ela foi
calculada só com os tokens 0..t. Uma passada de T tokens dá, então, T previsões
de "próximo token" -- todas usadas no treino (Sprint 5), e só a última usada na
geração (src/generation/).

O modelo é curto porque a complexidade está no bloco. O que cabe aqui são três
decisões:

- Os embeddings são os da Sprint 2 (InputEmbedding), não duas nn.Embedding
  novas. A ordem de criação dos pesos é a do livro (tokens, posições, blocos,
  norma, saída), então a mesma seed sorteia os mesmos números do capítulo.
- final_norm: o fluxo residual cresce bloco a bloco e nunca é normalizado no
  meio da pilha (Pre-LayerNorm). A norma final devolve a escala antes da
  projeção no vocabulário; sem ela, a temperatura efetiva dos logits mudaria
  com a profundidade.
- Cabeça de saída sem bias e, por padrão, sem weight tying -- como no livro. Com
  `tie_weights=True`, ela passa a ser a própria tabela de tokens.
"""

import torch
import torch.nn as nn

from src.embeddings import InputEmbedding
from src.transformer import LayerNorm, TransformerBlock

from .config import GPTConfig

BYTES_PER_FLOAT32 = 4
BYTES_PER_MEGABYTE = 1024 * 1024

# Ordem em que parameter_breakdown() devolve os componentes: a ordem do forward.
COMPONENTS = (
    "token_embedding",
    "positional_embedding",
    "attention",
    "feed_forward",
    "block_norms",
    "final_norm",
    "out_head",
)


class GPTModel(nn.Module):
    """GPT decoder-only: embeddings, pilha de TransformerBlocks e cabeça de saída."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Monta o modelo a partir da configuração.

        Args:
            cfg: hiperparâmetros; ver src/model/config.py.
        """
        super().__init__()
        self.cfg = cfg

        self.embedding = InputEmbedding(
            vocab_size=cfg.vocab_size,
            emb_dim=cfg.emb_dim,
            context_length=cfg.context_length,
        )
        self.drop_emb = nn.Dropout(cfg.emb_drop_rate)

        # Sequential e não ModuleList: o bloco recebe e devolve um tensor só,
        # então a pilha inteira é uma chamada no forward.
        self.trf_blocks = nn.Sequential(
            *(
                TransformerBlock(
                    emb_dim=cfg.emb_dim,
                    context_length=cfg.context_length,
                    n_heads=cfg.n_heads,
                    ff_hidden_dim=cfg.feed_forward_dim,
                    drop_rate=cfg.shortcut_drop_rate,
                    attn_drop_rate=cfg.attn_drop_rate,
                    qkv_bias=cfg.qkv_bias,
                )
                for _ in range(cfg.n_layers)
            )
        )

        self.final_norm = LayerNorm(cfg.emb_dim)

        # Sem bias, como no GPT-2: a LayerNorm final já tem o seu shift.
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # Mesmo objeto Parameter nos dois lugares: um gradiente só, somado
            # das duas pontas. Os shapes coincidem -- (vocab_size, emb_dim) --
            # porque nn.Linear guarda o peso como (out_features, in_features).
            self.out_head.weight = self.embedding.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Calcula os logits do próximo token em cada posição.

        Args:
            token_ids: (B, T), dtype torch.long, com T <= context_length.

        Returns:
            Logits (B, T, vocab_size). A linha t é a previsão do token t+1 e
            depende só dos tokens 0..t. Ainda não são probabilidades: falta o
            softmax, que fica a cargo de quem usa (perda ou geração).

        Raises:
            ValueError: se T passar do context_length.
        """
        x = self.embedding(token_ids)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

    @property
    def context_length(self) -> int:
        return self.cfg.context_length

    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab_size

    def parameter_count(self) -> int:
        """Total de pesos treináveis.

        Com weight tying, a matriz compartilhada conta uma vez só:
        parameters() já remove a duplicata.
        """
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        """Pesos por componente, na ordem do forward.

        Returns:
            Dicionário com as chaves de COMPONENTS. Os blocos aparecem somados
            (atenção de todos os blocos, FeedForward de todos os blocos, as
            2·n_layers LayerNorms). A soma dos valores é parameter_count().
            Com weight tying, `out_head` vale 0: a matriz já foi contada em
            `token_embedding`.
        """
        breakdown = dict.fromkeys(COMPONENTS, 0)
        for name, parameter in self.named_parameters():
            breakdown[_component_of(name)] += parameter.numel()
        return breakdown

    def size_in_megabytes(self, bytes_per_parameter: int = BYTES_PER_FLOAT32) -> float:
        """Memória ocupada só pelos pesos, sem ativações nem otimizador.

        Args:
            bytes_per_parameter: 4 para float32, o padrão do PyTorch.
        """
        return self.parameter_count() * bytes_per_parameter / BYTES_PER_MEGABYTE


def _component_of(parameter_name: str) -> str:
    """Classifica um parâmetro pelo nome que o named_parameters() dá a ele.

    Raises:
        ValueError: para um nome que não pertence a nenhum componente -- sinal
            de que o modelo ganhou uma peça e a contagem ficou desatualizada.
    """
    if parameter_name.startswith("embedding.token_embedding."):
        return "token_embedding"
    if parameter_name.startswith("embedding.positional_embedding."):
        return "positional_embedding"
    if parameter_name.startswith("trf_blocks."):
        # trf_blocks.<i>.<subpeça>.<...>
        subpart = parameter_name.split(".")[2]
        if subpart == "att":
            return "attention"
        if subpart == "ff":
            return "feed_forward"
        if subpart in ("norm1", "norm2"):
            return "block_norms"
    if parameter_name.startswith("final_norm."):
        return "final_norm"
    if parameter_name.startswith("out_head."):
        return "out_head"
    raise ValueError(f"parâmetro {parameter_name!r} fora dos componentes conhecidos")
