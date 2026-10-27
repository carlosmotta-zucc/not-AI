"""Rede de demonstração da seção 4.4: o que a conexão de atalho faz pelo gradiente.

Não faz parte do GPT. É a rede de cinco camadas (Linear + GELU) que o livro usa
para isolar o efeito do atalho antes de colocá-lo no bloco: a mesma rede, com a
mesma seed, com e sem `x = x + camada(x)`. O atalho de verdade está em block.py.

O mecanismo, em uma linha: sem atalho, o gradiente que chega à camada 1 é o
produto das derivadas de todas as camadas acima dela, e cada fator menor que 1
encolhe o total. Com atalho, a derivada de x + f(x) é 1 + f'(x): o "1" cria um
caminho em que o gradiente atravessa a camada sem ser multiplicado.

Vive em src/ pelo mesmo motivo da atenção simplificada da Sprint 3: o teste dos
números do livro e o experimento que os publica usam a mesma rede.
"""

from typing import Sequence

import torch
import torch.nn as nn

from .gelu import GELU


class ShortcutDemoNetwork(nn.Module):
    """Pilha de Linear + GELU, com atalho opcional onde os shapes permitem."""

    def __init__(self, layer_sizes: Sequence[int], use_shortcut: bool) -> None:
        """Cria uma camada por par consecutivo de `layer_sizes`.

        Args:
            layer_sizes: dimensões de entrada e saída em sequência; o livro usa
                [3, 3, 3, 3, 3, 1], cinco camadas.
            use_shortcut: soma a entrada à saída de cada camada que mantém o
                shape. A última (3 -> 1) não mantém, e fica sem atalho.
        """
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(entrada, saida), GELU())
            for entrada, saida in zip(layer_sizes[:-1], layer_sizes[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passa `x` pelas camadas, somando o atalho quando ligado e possível.

        Args:
            x: (B, layer_sizes[0]).

        Returns:
            (B, layer_sizes[-1]).
        """
        for layer in self.layers:
            output = layer(x)
            if self.use_shortcut and output.shape == x.shape:
                x = x + output
            else:
                x = output
        return x


def mean_weight_gradients(network: nn.Module, inputs: torch.Tensor) -> list[float]:
    """Gradiente médio absoluto dos pesos de cada camada, da primeira à última.

    Args:
        network: a rede, com gradientes zerados.
        inputs: (B, d_in).

    Returns:
        Uma média por matriz de pesos (os bias ficam de fora, como no livro).

    A perda é o erro quadrático contra zero: não importa o que a rede deveria
    prever, só como o gradiente se distribui pelas camadas.
    """
    output = network(inputs)
    loss = nn.functional.mse_loss(output, torch.zeros_like(output))
    loss.backward()

    return [
        parameter.grad.abs().mean().item()
        for name, parameter in network.named_parameters()
        if name.endswith("weight")
    ]
