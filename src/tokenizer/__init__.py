"""Tokenizacao: primeira etapa do pipeline (Sprint 2, capitulo 2).

    texto -> split_text -> tokens -> Vocabulary -> Token IDs

Dois tokenizadores convivem aqui de proposito:

- SimpleTokenizer, por palavras, escrito a mao. E o que torna cada etapa do
  caminho inspecionavel.
- BPETokenizer, subword, sobre o tiktoken. E o que o GPT-2 usa e o que o
  projeto leva para as sprints seguintes.

O comparativo entre os dois esta em experimentos/sprint02/.
"""

from .bpe import BPETokenizer, load_gpt2_encoding
from .simple_tokenizer import SimpleTokenizer
from .splitter import count_tokens, split_text, unique_tokens
from .vocabulary import (
    DEFAULT_SPECIAL_TOKENS,
    END_OF_TEXT_TOKEN,
    UNKNOWN_TOKEN,
    Vocabulary,
)

__all__ = [
    "BPETokenizer",
    "DEFAULT_SPECIAL_TOKENS",
    "END_OF_TEXT_TOKEN",
    "SimpleTokenizer",
    "UNKNOWN_TOKEN",
    "Vocabulary",
    "count_tokens",
    "load_gpt2_encoding",
    "split_text",
    "unique_tokens",
]
