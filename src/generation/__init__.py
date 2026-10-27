"""Inferência e geração de texto (Sprint 4, seção 4.7 e seção 5.3 do livro).

    texto -> Token IDs -> GPTModel -> logits -> probabilidades -> próximo token
          ^                                                            |
          +----------------- anexado ao contexto ----------------------+

- decoding.py -- o que acontece com UM vetor de logits: top-k, temperatura,
  softmax e a escolha (gulosa ou por amostragem). Não conhece o modelo.
- autoregressive.py -- o laço: recorte do contexto, passada pelo modelo,
  escolha, concatenação. Mais as conversões texto <-> Token IDs.

Decisões que o resto do código segue:

- Uma função de geração só, `generate`. Com `temperature=0` (padrão) ela é o
  `generate_text_simple` do capítulo 4; com `temperature > 0` e `top_k`, é o
  `generate` da seção 5.3. Os experimentos comparam as estratégias mudando só
  esses argumentos.
- A geração é inferência: `generate` põe o modelo em eval(), roda sob
  no_grad() e devolve o modo original no fim.
- Aleatoriedade por `torch.Generator`, passado como argumento: dois sorteios
  com o mesmo gerador semeado dão o mesmo texto, sem depender da seed global.
"""

from .autoregressive import (
    SupportsTokenizer,
    generate,
    generate_text,
    text_to_token_ids,
    token_ids_to_text,
)
from .decoding import (
    GREEDY_TEMPERATURE,
    apply_top_k,
    next_token_probabilities,
    select_next_token,
)

__all__ = [
    "GREEDY_TEMPERATURE",
    "SupportsTokenizer",
    "apply_top_k",
    "generate",
    "generate_text",
    "next_token_probabilities",
    "select_next_token",
    "text_to_token_ids",
    "token_ids_to_text",
]
