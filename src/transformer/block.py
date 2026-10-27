"""TransformerBlock (seção 4.5): atenção e FeedForward em volta do fluxo residual.

    x ─┬─ LayerNorm ─ MultiHeadAttention ─ Dropout ─(+)─┬─ LayerNorm ─ FeedForward ─ Dropout ─(+)─ saída
       └────────────────────────────────────────────┘   └──────────────────────────────────────┘
                     atalho da atenção                              atalho da FeedForward

Escrito como duas linhas, é isto:

    x = x + Dropout(Atenção(LayerNorm(x)))
    x = x + Dropout(FeedForward(LayerNorm(x)))

O x que atravessa as duas linhas é o FLUXO RESIDUAL. Cada subcamada lê uma
cópia normalizada dele e devolve uma correção que é SOMADA, nunca uma
substituição. Daí as três propriedades que o bloco precisa ter:

- Shape constante: (B, T, emb_dim) entra e sai. É a condição da soma, e o que
  permite empilhar n_layers blocos sem ajustar dimensão.
- Gradiente com caminho direto: a derivada de x + f(x) em relação a x é
  1 + f'(x). O "1" atravessa todos os blocos sem multiplicar, e é o que impede
  o gradiente de sumir na pilha (seção 4.4).
- Pre-LayerNorm: a normalização fica DENTRO do ramo, antes da subcamada, e não
  depois da soma como no Transformer original (Post-LayerNorm). O fluxo
  residual em si nunca é normalizado no meio da pilha -- só na final_norm do
  GPTModel.

A causalidade vem inteira da MultiHeadAttention da Sprint 3: LayerNorm e
FeedForward agem token a token, então não têm como misturar posições.
"""

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention

from .feed_forward import FeedForward
from .layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """Um bloco do GPT: atenção causal multi-cabeça e FeedForward, Pre-LayerNorm."""

    def __init__(
        self,
        emb_dim: int,
        context_length: int,
        n_heads: int,
        ff_hidden_dim: int | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float | None = None,
        qkv_bias: bool = False,
    ) -> None:
        """Cria as quatro subpeças na ordem do livro.

        Args:
            emb_dim: dimensão do fluxo residual; entra e sai igual.
            context_length: maior sequência aceita pela máscara causal.
            n_heads: cabeças da atenção; emb_dim precisa ser divisível por ele.
            ff_hidden_dim: camada escondida da FeedForward; None = 4 * emb_dim.
            drop_rate: dropout aplicado à saída de cada subcamada, antes da
                soma residual (o `drop_shortcut` do livro).
            attn_drop_rate: dropout nos pesos de atenção. None herda
                `drop_rate`, como o livro faz com uma taxa só; o exercício 4.3
                separa as duas.
            qkv_bias: bias nas projeções Q, K e V. False como no GPT-2.

        A ordem de criação (att, ff, norm1, norm2) é a do livro de propósito:
        com a mesma seed, os pesos sorteados saem iguais aos do capítulo, e é
        isso que deixa tests/test_book_checkpoints.py conferir os logits da
        seção 4.6 número a número.
        """
        super().__init__()
        if attn_drop_rate is None:
            attn_drop_rate = drop_rate

        # d_in = d_out = emb_dim: a saída da atenção precisa ter o shape da
        # entrada para a soma residual. O teste de integração da Sprint 3
        # (tests/test_attention_pipeline.py) já fixava essa condição.
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            num_heads=n_heads,
            dropout=attn_drop_rate,
            qkv_bias=qkv_bias,
        )
        self.ff = FeedForward(emb_dim, ff_hidden_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

        # Um módulo para os dois ramos: dropout não tem estado além da taxa, e
        # cada chamada sorteia a própria máscara.
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica as duas subcamadas residuais.

        Args:
            x: (B, T, emb_dim) ou (T, emb_dim), com T <= context_length.

        Returns:
            Tensor do mesmo shape de `x`.

        Raises:
            ValueError: se T passar do context_length (vem da atenção).
        """
        # Subcamada 1: cada token lê os anteriores. Os pesos de atenção, que a
        # Sprint 3 devolve para os heatmaps, não são usados aqui.
        attended, _ = self.att(self.norm1(x))
        x = x + self.drop_shortcut(attended)

        # Subcamada 2: cada token é transformado sozinho.
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))

        return x

    @property
    def emb_dim(self) -> int:
        return self.att.d_in

    @property
    def n_heads(self) -> int:
        return self.att.num_heads

    @property
    def context_length(self) -> int:
        return self.att.context_length
