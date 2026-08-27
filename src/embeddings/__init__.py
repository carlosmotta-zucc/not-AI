"""Do Token ID ao lote: segunda etapa do pipeline (Sprint 2, capitulo 2).

    Token IDs -> janela deslizante -> lotes (B, T) -> embeddings -> (B, T, D)

A Frente A parou no Token ID e mostrou por que ele nao basta: e rotulo nominal,
nao representacao. Este pacote cobre a metade do caminho que troca ID por
vetor, em dois passos:

- TokenEmbedding (token_embeddings.py) troca cada ID por um vetor treinavel.
- PositionalEmbedding e InputEmbedding (positional.py) acrescentam a posicao,
  que a tabela de tokens sozinha nao tem como representar.

A outra metade -- recortar a sequencia continua de Token IDs em pares
(entrada, alvo) e empilha-los em lotes -- e responsabilidade separada e vive
em src/dataset/ (`GPTDataset`, `create_dataloader`): janela deslizante e
"trocar ID por vetor" nao tem por que morar no mesmo pacote.

Os experimentos que medem o efeito de stride, context_length e emb_dim estao em
experimentos/sprint02/frente_b_experimentos.py.
"""

from .positional import InputEmbedding, PositionalEmbedding
from .token_embeddings import TokenEmbedding, one_hot_lookup

__all__ = [
    "InputEmbedding",
    "PositionalEmbedding",
    "TokenEmbedding",
    "one_hot_lookup",
]
