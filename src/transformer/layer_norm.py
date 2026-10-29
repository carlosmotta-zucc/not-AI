"""Normalização de camada (seção 4.2): média 0 e variância 1 por token.

O fluxo residual do GPT é uma soma que cresce bloco a bloco (ver block.py). Sem
normalização, a escala do que entra na atenção e na FeedForward dependeria da
profundidade, e o softmax e a GELU trabalhariam em regimes diferentes em cada
bloco. A LayerNorm devolve cada vetor à mesma escala antes de cada subcamada.

Normaliza na ÚLTIMA dimensão, a de features. Cada token é tratado sozinho: o
resultado não depende dos outros tokens da frase nem dos outros exemplos do
lote. É a diferença para a BatchNorm, e o que deixa a LayerNorm funcionar com
lote de 1 na geração de texto.

Implementação própria, como pede a regra do projeto; `nn.LayerNorm` entra só
como baseline no teste de equivalência.
"""

import torch
import torch.nn as nn

# O mesmo epsilon do livro e do nn.LayerNorm: evita divisão por zero num vetor
# constante, pequeno demais para mudar a conta de qualquer outro.
DEFAULT_EPS = 1e-5


class LayerNorm(nn.Module):
    """LayerNorm com escala e deslocamento treináveis."""

    def __init__(self, emb_dim: int, eps: float = DEFAULT_EPS) -> None:
        """Cria `scale` em 1 e `shift` em 0.

        Args:
            emb_dim: tamanho da última dimensão da entrada.
            eps: somado à variância antes da raiz.

        Começando em 1 e 0, a saída inicial é a normalização pura. O treino
        pode desfazê-la, canal a canal, se a escala original for mais útil --
        normalizar não tira capacidade do modelo.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normaliza cada vetor da última dimensão.

        Args:
            x: (..., emb_dim). No GPT, (B, T, emb_dim).

        Returns:
            Tensor do mesmo shape. Com scale = 1 e shift = 0, cada vetor sai
            com média 0 e variância 1.
        """
        # dim=-1 e não dim=2: a mesma linha serve a (T, D) e a (B, T, D).
        mean = x.mean(dim=-1, keepdim=True)
        # Variância enviesada (divide por n, não n-1), como o GPT-2 original em
        # TensorFlow. Com n = emb_dim a diferença é desprezível; a escolha é
        # por compatibilidade com os pesos da OpenAI (capítulo 6).
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized + self.shift

    @property
    def emb_dim(self) -> int:
        return self.scale.shape[0]
