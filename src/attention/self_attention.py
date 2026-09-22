"""Atencao com pesos treinaveis: a semelhanca passa a ser aprendida.

Em simplified.py o criterio esta congelado no produto escalar cru dos
embeddings. As tres matrizes daqui projetam os mesmos vetores em espacos que o
backpropagation ajusta, e a comparacao passa a acontecer la.

Por que tres e nao uma: W_query e W_key separadas quebram a simetria (com uma
so, o score de A para B seria o mesmo de B para A). W_value responde a outra
pergunta -- Q e K decidem quanto um token pesa, V decide o que ele contribui.

nn.Linear e primitiva, no mesmo nivel do nn.Embedding da Sprint 2; a regra do
projeto mira nn.MultiheadAttention e nn.Transformer.

Exercicio 3.1: nn.Linear(d_in, d_out) guarda o peso como (d_out, d_in), porque
calcula x @ weight.T. Quem multiplicar na mao precisa do .T -- e com
d_in == d_out o esquecimento da resultado errado em silencio. Ha teste.
"""

import torch
import torch.nn as nn

from .scaled_dot_product import scaled_dot_product_attention


class SelfAttention(nn.Module):
    """Self-attention de uma cabeca, com Q, K e V treinaveis e sem mascara."""

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        """Cria as tres projecoes (d_in -> d_out).

        Args:
            d_in: dimensao da entrada. E o emb_dim da Sprint 2.
            d_out: dimensao em que a atencao trabalha; na multi-head sera o
                head_dim de cada cabeca.
            qkv_bias: bias nas projecoes. False como no GPT-2 -- nao muda o que
                a atencao representa, so soma parametros.
        """
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Projeta a entrada em Q, K e V e entrega para a funcao nucleo.

        Args:
            inputs: (T, d_in) ou (B, T, d_in).

        Returns:
            (context, weights): (T, d_out) ou (B, T, d_out), e (T, T) ou
            (B, T, T).

        Sem `mask` e com a escala ligada. A ausencia da mascara e o unico que
        distingue esta classe de CausalAttention.
        """
        return scaled_dot_product_attention(
            self.W_query(inputs), self.W_key(inputs), self.W_value(inputs)
        )

    @property
    def d_in(self) -> int:
        return self.W_query.in_features

    @property
    def d_out(self) -> int:
        return self.W_query.out_features
