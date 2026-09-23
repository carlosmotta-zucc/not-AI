"""Atencao causal: a mesma conta, com o futuro escondido.

Deixar a query em i ver a chave em j > i e trapaca num modelo que preve o
proximo token: o alvo da posicao t esta na posicao t+1 da propria entrada. A
mascara tira o vazamento sem tirar o paralelismo.

Convencao do pacote: bool (T, T), True = BLOQUEADO, com masked_fill(-inf) ANTES
do softmax. A secao 3.5.1 do livro chega no mesmo lugar zerando os pesos depois
e renormalizando as linhas a mao; ha teste provando que os dois coincidem.

A mascara e register_buffer e nao parametro (nao ha o que aprender num
triangulo de booleanos), alocada uma vez no context_length maximo e recortada
em [:T, :T] a cada forward.
"""

import torch
import torch.nn as nn

from .scaled_dot_product import scaled_dot_product_attention


def causal_mask(num_tokens: int) -> torch.Tensor:
    """Monta a mascara triangular na convencao do projeto.

    Args:
        num_tokens: lado da matriz quadrada.

    Returns:
        Tensor bool (num_tokens, num_tokens), True acima da diagonal.

    diagonal=1 deixa a diagonal livre: a query em i le a chave em i.
    """
    return torch.triu(torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=1)


class CausalAttention(nn.Module):
    """Self-attention de uma cabeca com mascara causal e dropout nos pesos."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        """Cria as tres projecoes, o dropout e a mascara.

        Args:
            d_in: dimensao da entrada. E o emb_dim da Sprint 2.
            d_out: dimensao em que a atencao trabalha; na multi-head sera o
                head_dim de cada cabeca.
            context_length: maior sequencia aceita. Define o tamanho da
                mascara, e so dela -- as projecoes nao dependem de T.
            dropout: probabilidade aplicada aos PESOS, nao a entrada. Desligado
                por padrao; a configuracao base do pacote passa 0.1.
            qkv_bias: bias nas projecoes. False como no GPT-2.
        """
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # nn.Dropout e nao a taxa crua: o modulo sabe se esta em train() ou eval().
        self.dropout = nn.Dropout(dropout)

        # Buffer, nao parametro: booleano nao tem gradiente. Ver docstring.
        self.register_buffer("mask", causal_mask(context_length))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Projeta a entrada em Q, K e V e aplica a atencao mascarada.

        Args:
            inputs: (T, d_in) ou (B, T, d_in), com T <= context_length.

        Returns:
            (context, weights): (T, d_out) ou (B, T, d_out), e (T, T) ou
            (B, T, T). Em eval(), a linha i dos pesos tem zeros exatos depois
            da coluna i e soma 1.

        Raises:
            ValueError: se T passar do context_length.
        """
        num_tokens = inputs.shape[-2]

        # Sem isto o recorte daria mascara menor que os scores, e o erro sairia
        # como broadcast quebrado dentro do masked_fill.
        if num_tokens > self.context_length:
            raise ValueError(
                f"sequencia de {num_tokens} tokens passa do context_length "
                f"{self.context_length} da mascara"
            )

        return scaled_dot_product_attention(
            self.W_query(inputs),
            self.W_key(inputs),
            self.W_value(inputs),
            mask=self.mask[:num_tokens, :num_tokens],
            dropout=self.dropout,
        )

    @property
    def d_in(self) -> int:
        return self.W_query.in_features

    @property
    def d_out(self) -> int:
        return self.W_query.out_features

    @property
    def context_length(self) -> int:
        return self.mask.shape[0]
