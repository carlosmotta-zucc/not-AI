"""FeedForward (seção 4.3): a subcamada que trabalha em cada token sozinho.

    (B, T, D) --Linear--> (B, T, H) --GELU--> (B, T, H) --Linear--> (B, T, D)

com H = 4·D no GPT-2. A divisão de trabalho dentro do bloco é esta:

- A atenção MISTURA os tokens: o vetor de saída da posição t é uma média dos
  values de todas as posições até t. Mas é uma média ponderada -- uma operação
  linear nos values, com pouca transformação do conteúdo em si.
- A FeedForward TRANSFORMA cada token: a mesma rede de duas camadas é aplicada
  a cada posição, sem olhar para as outras. É onde está a não linearidade do
  bloco, e é a maior parte dos parâmetros dele (o dobro da atenção no GPT-2).

Expandir para 4·D e voltar dá à GELU um espaço maior para separar padrões; a
volta para D é o que mantém o shape e permite somar a saída ao fluxo residual.
"""

import torch
import torch.nn as nn

from .gelu import GELU

EXPANSION_FACTOR = 4


class FeedForward(nn.Module):
    """Duas camadas lineares com GELU no meio, aplicadas posição a posição."""

    def __init__(self, emb_dim: int, hidden_dim: int | None = None) -> None:
        """Cria as duas projeções.

        Args:
            emb_dim: dimensão de entrada e de saída (o D do fluxo residual).
            hidden_dim: dimensão da camada escondida. None reproduz o livro,
                4 * emb_dim; os experimentos variam este número.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = EXPANSION_FACTOR * emb_dim

        # Sequential com os três passos na ordem do diagrama do módulo: a
        # sequência é a documentação, e os índices 0 e 2 são as projeções.
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expande, ativa e contrai cada vetor.

        Args:
            x: (..., emb_dim). No GPT, (B, T, emb_dim).

        Returns:
            Tensor do mesmo shape de `x`. A saída da posição t depende só da
            entrada da posição t.
        """
        return self.layers(x)

    @property
    def emb_dim(self) -> int:
        return self.layers[0].in_features

    @property
    def hidden_dim(self) -> int:
        return self.layers[0].out_features
