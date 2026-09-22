"""A conta da atencao, escrita uma vez so.

Todas as variantes do pacote terminam nesta mesma sequencia; o que muda e de
onde vem Q, K e V e quais passos opcionais estao ligados. Duplicar isso em
quatro arquivos seria quatro lugares para errar o eixo do softmax.

E funcao, e nao nn.Module, porque nao tem estado -- ate o dropout, que tem,
entra por parametro.
"""

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
    scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula a atencao de produto escalar escalado.

    Args:
        queries: (..., T_q, d_k).
        keys: (..., T_k, d_k).
        values: (..., T_k, d_v). d_v pode diferir de d_k.
        mask: bool (T_q, T_k) ou None. True marca posicao BLOQUEADA.
        dropout: nn.Dropout da camada que chamou, ou None.
        scale: divide por sqrt(d_k). Desligar so na atencao simplificada.

    Returns:
        (context, weights): (..., T_q, d_v) e (..., T_q, T_k), linhas somando 1.

    As dimensoes em "..." passam intactas -- e o que faz a mesma funcao servir
    a (B, T, d) e a (B, H, T, head_dim).
    """
    # transpose(-2, -1) e nao .T: .T so vale em 2D e inverteria todos os eixos.
    scores = queries @ keys.transpose(-2, -1)

    if scale:
        # Sem isso o produto escalar cresce com d_k, o softmax satura e o
        # gradiente some.
        scores = scores / keys.shape[-1] ** 0.5

    if mask is not None:
        # -inf antes do softmax vira zero depois, com as linhas ja somando 1.
        # Zerar os pesos depois exigiria renormalizar na mao.
        scores = scores.masked_fill(mask, float("-inf"))

    # dim=-1: cada query distribui 1.0 entre os tokens que pode ver.
    weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    return weights @ values, weights
