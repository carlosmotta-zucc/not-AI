"""GELU (seção 4.3): a não linearidade da FeedForward.

    GELU(x) = x · Φ(x),  Φ = função de distribuição acumulada da normal padrão

Lida como "deixa x passar com a probabilidade de uma normal ser menor que x":
perto de x para x grande, perto de 0 para x muito negativo, e suave no meio.
Diferenças para a ReLU que importam no treino:

- Não tem bico em 0, então o gradiente muda de forma contínua.
- Não zera os negativos: x = -1 sai -0,159 e ainda recebe gradiente. Na ReLU,
  um neurônio com entrada negativa para de aprender naquele exemplo.

O GPT-2 foi treinado com a aproximação por tanh, mais barata que a erf na época;
ela é o padrão aqui para reproduzir o livro. A forma exata fica disponível para
o experimento que mede a diferença entre as duas.
"""

import math

import torch
import torch.nn as nn

# Constantes da aproximação por tanh, obtidas por ajuste de curva no artigo
# original da GELU. Calculadas uma vez, e não a cada forward.
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
CUBIC_COEFFICIENT = 0.044715

APPROXIMATIONS = ("tanh", "none")


class GELU(nn.Module):
    """Gaussian Error Linear Unit, aproximada por tanh (GPT-2) ou exata (erf)."""

    def __init__(self, approximate: str = "tanh") -> None:
        """Escolhe a fórmula.

        Args:
            approximate: "tanh", a do GPT-2 e do livro, ou "none", a exata.
                Mesmos nomes do nn.GELU, que serve de baseline nos testes.

        Raises:
            ValueError: se `approximate` não for uma das duas.
        """
        super().__init__()
        if approximate not in APPROXIMATIONS:
            raise ValueError(
                f"approximate={approximate!r}; esperado um de {APPROXIMATIONS}"
            )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica a GELU elemento a elemento.

        Args:
            x: tensor de qualquer shape.

        Returns:
            Tensor do mesmo shape.
        """
        if self.approximate == "tanh":
            inner = SQRT_2_OVER_PI * (x + CUBIC_COEFFICIENT * x.pow(3))
            return 0.5 * x * (1.0 + torch.tanh(inner))

        # Φ(x) escrita com a erf: Φ(x) = (1 + erf(x / √2)) / 2.
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"
