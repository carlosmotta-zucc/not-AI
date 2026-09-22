"""Atencao sem pesos treinaveis: o mecanismo isolado do aprendizado.

Ponderar (cada token como media pesada dos outros) e aprender o criterio da
ponderacao sao ideias separadas. Aqui so existe a primeira: Q, K e V sao as
proprias entradas e a semelhanca e o produto escalar cru. A segunda esta em
self_attention.py.

Duas consequencias, nos numeros do exemplo:

- Os SCORES sao simetricos (o produto escalar comuta), mas os pesos nao -- o
  softmax normaliza linha a linha. Com W_q e W_k separados nem os scores
  continuam simetricos, e e o que se quer.
- Cada token puxa para si, mas nem sempre ganha de si mesmo: "starts" pesa
  "journey" (0,2369) acima do proprio vetor (0,2326), porque os dois sao quase
  identicos. O mecanismo mede semelhanca e nada alem dela.

Unico ponto do pacote com scale=False, e o motivo de esse parametro existir.
"""

import torch

from .scaled_dot_product import scaled_dot_product_attention

# Exemplo de referencia do capitulo 3, pequeno o bastante para conferir a conta
# na mao. Fica aqui para notebook e testes partirem do mesmo tensor; e o unico
# dado de exemplo em src/, e existe so como fixture.
EXAMPLE_WORDS = ("Your", "journey", "starts", "with", "one", "step")

EXAMPLE_INPUTS = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)


def simplified_attention(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Atencao onde as entradas fazem os tres papeis: query, key e value.

    Args:
        inputs: (T, d) ou (B, T, d).

    Returns:
        (context, weights): context com o shape de `inputs`, weights (T, T) ou
        (B, T, T).

    Sem parametro treinavel: a saida e funcao apenas da entrada.
    """
    return scaled_dot_product_attention(inputs, inputs, inputs, scale=False)
