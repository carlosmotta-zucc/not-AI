"""Do Token ID a sequencia de amostras: janela deslizante e lotes.

    Token IDs -> janela deslizante -> pares (x, y) -> tensores -> lotes (B, T)

Duas camadas, uma responsabilidade cada:

- sequences.py resolve o recorte em Python puro: `sliding_window_pairs` e
  `expected_sample_count`, sem torch e sem tokenizador.
- loader.py empacota o mesmo recorte em tensores (`GPTDataset`) e monta o
  DataLoader que alimenta o treino (`create_dataloader`).

`src/embeddings/` consome `GPTDataset` e `create_dataloader` daqui para
completar o caminho ate os vetores que a Sprint 3 recebe.
"""

from .loader import GPTDataset, SupportsEncode, create_dataloader
from .sequences import expected_sample_count, sliding_window_pairs

__all__ = [
    "GPTDataset",
    "SupportsEncode",
    "create_dataloader",
    "expected_sample_count",
    "sliding_window_pairs",
]
