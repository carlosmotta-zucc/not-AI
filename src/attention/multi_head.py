"""Varias cabecas de atencao sobre a mesma sequencia.

Uma cabeca aprende UM criterio de relacao entre tokens. Concordancia,
referencia de pronome e proximidade semantica sao perguntas diferentes, e uma
projecao so tem que escolher; varias cabecas em paralelo fazem as tres.

Duas implementacoes, e a segunda so se entende contra a primeira:

- MultiHeadAttentionWrapper -- a leitura literal: uma lista de CausalAttention,
  rodadas em sequencia e concatenadas. Correta, e o preco aparece no laco.
- MultiHeadAttention -- o weight split: UMA projecao Q/K/V de d_in para d_out,
  fatiada em cabecas com view + transpose. Mesma conta, um matmul so.

Nas duas, d_out e o TOTAL: head_dim = d_out // num_heads. Uma troca pela outra
sem mexer em mais nada. O livro usa d_out por cabeca no Wrapper; aqui nao.

A parte perigosa e o reshape:

    (B, T, d_out) --view--> (B, T, H, head_dim) --transpose(1,2)--> (B, H, T, head_dim)

O view so reinterpreta as colunas (a cabeca h fica com o bloco contiguo
[h*head_dim : (h+1)*head_dim]); o transpose so troca strides, para que as duas
ultimas dimensoes voltem a ser (tokens, features). Pular o transpose e mandar
view(B, H, T, head_dim) direto e legal e da numero errado em silencio. Ha teste.

Mais cabecas com o mesmo d_out da cabecas MENORES, nao rede maior: o numero de
parametros nao depende de num_heads.
"""

import torch
import torch.nn as nn

from .causal import CausalAttention, causal_mask
from .scaled_dot_product import scaled_dot_product_attention


def head_dimension(d_out: int, num_heads: int) -> int:
    """Tamanho de cada cabeca, com erro explicito quando a divisao nao e exata.

    Args:
        d_out: dimensao TOTAL da saida, somando todas as cabecas.
        num_heads: quantas cabecas.

    Returns:
        d_out // num_heads.

    Raises:
        ValueError: se d_out nao for divisivel por num_heads.

    Aqui e nao nos dois __init__ para a mensagem existir uma vez so. Falha na
    construcao, onde os hiperparametros foram escolhidos, e nao no forward.
    """
    if d_out % num_heads != 0:
        raise ValueError(
            f"d_out={d_out} nao e divisivel por num_heads={num_heads}: "
            f"as cabecas teriam tamanhos diferentes"
        )
    return d_out // num_heads


class MultiHeadAttentionWrapper(nn.Module):
    """Multi-head pela definicao: uma lista de cabecas causais, concatenadas."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        """Cria `num_heads` CausalAttention de head_dim cada.

        Args:
            d_in: dimensao da entrada.
            d_out: dimensao TOTAL da saida, dividida entre as cabecas.
            context_length: maior sequencia aceita.
            num_heads: quantas cabecas, cada uma com suas Q, K e V.
            dropout: probabilidade aplicada aos pesos de cada cabeca.
            qkv_bias: bias nas projecoes. False como no GPT-2.
        """
        super().__init__()
        head_dim = head_dimension(d_out, num_heads)

        # ModuleList e nao lista comum: e o que registra as cabecas como
        # submodulos, para parameters(), .to() e train()/eval() alcancarem todas.
        self.heads = nn.ModuleList(
            CausalAttention(d_in, head_dim, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Roda as cabecas em sequencia e junta o que elas produziram.

        Args:
            inputs: (T, d_in) ou (B, T, d_in).

        Returns:
            (context, weights): (..., T, d_out) e (..., H, T, T).

        O custo esta neste laco: num_heads passadas pelo mesmo tensor e
        num_heads mascaras identicas alocadas, uma por cabeca. Em matematica
        isso e um matmul so -- e o que a classe seguinte faz.
        """
        saidas = [head(inputs) for head in self.heads]

        # cat em -1: cabecas viram blocos de colunas, na ordem da lista.
        # stack em -3: abre o eixo H onde o weight split tambem deixa os pesos,
        # e e o que torna as duas implementacoes comparaveis tensor a tensor.
        context = torch.cat([contexto for contexto, _ in saidas], dim=-1)
        weights = torch.stack([pesos for _, pesos in saidas], dim=-3)

        return context, weights

    @property
    def num_heads(self) -> int:
        return len(self.heads)

    @property
    def head_dim(self) -> int:
        return self.heads[0].d_out

    @property
    def d_in(self) -> int:
        return self.heads[0].d_in

    @property
    def d_out(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def context_length(self) -> int:
        return self.heads[0].context_length


class MultiHeadAttention(nn.Module):
    """Multi-head com weight split: uma projecao por matriz, fatiada em cabecas."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        """Cria as tres projecoes unicas, a projecao de saida, o dropout e a mascara.

        Args:
            d_in: dimensao da entrada.
            d_out: dimensao TOTAL da saida, dividida entre as cabecas.
            context_length: maior sequencia aceita.
            num_heads: em quantas cabecas d_out e fatiado.
            dropout: probabilidade aplicada aos pesos.
            qkv_bias: bias nas projecoes Q/K/V. False como no GPT-2 -- a
                out_proj tem bias de qualquer jeito, tambem como no GPT-2.
        """
        super().__init__()
        self.head_dim = head_dimension(d_out, num_heads)
        self.num_heads = num_heads

        # Uma Linear por matriz, nao uma por cabeca: as projecoes do Wrapper
        # empilhadas dao exatamente estas.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Sem ela as cabecas saem lado a lado e nunca conversam. E o unico
        # passo que o Wrapper nao tem.
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", causal_mask(context_length))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Projeta, fatia em cabecas, aplica a atencao mascarada e junta de volta.

        Args:
            inputs: (T, d_in) ou (B, T, d_in), com T <= context_length.

        Returns:
            (context, weights): (..., T, d_out) e (..., H, T, T).

        Raises:
            ValueError: se T passar do context_length.
        """
        # As outras tres classes do pacote aceitam entrada sem lote, e o view
        # precisa do eixo: ele e criado aqui e desfeito no fim.
        sem_lote = inputs.dim() == 2
        if sem_lote:
            inputs = inputs.unsqueeze(0)

        batch_size, num_tokens, _ = inputs.shape

        if num_tokens > self.context_length:
            raise ValueError(
                f"sequencia de {num_tokens} tokens passa do context_length "
                f"{self.context_length} da mascara"
            )

        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            """(B, T, d_out) -> (B, H, T, head_dim)."""
            return projected.view(
                batch_size, num_tokens, self.num_heads, self.head_dim
            ).transpose(1, 2)

        # A mascara (T, T) faz broadcast contra (B, H, T, T) sozinha.
        context, weights = scaled_dot_product_attention(
            split_heads(self.W_query(inputs)),
            split_heads(self.W_key(inputs)),
            split_heads(self.W_value(inputs)),
            mask=self.mask[:num_tokens, :num_tokens],
            dropout=self.dropout,
        )

        # Volta ao formato de sequencia. O contiguous e obrigatorio: o context
        # veio novo da atencao e o transpose quebra a leitura linear que o view
        # exige -- sem ele, RuntimeError.
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, num_tokens, self.d_out)
        context = self.out_proj(context)

        if sem_lote:
            return context.squeeze(0), weights.squeeze(0)

        return context, weights

    @property
    def d_in(self) -> int:
        return self.W_query.in_features

    @property
    def d_out(self) -> int:
        return self.W_query.out_features

    @property
    def context_length(self) -> int:
        return self.mask.shape[0]
