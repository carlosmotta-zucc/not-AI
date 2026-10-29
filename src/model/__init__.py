"""Arquitetura GPT: os blocos empilhados num modelo (Sprint 4, capítulo 4).

    Token IDs (B, T) -> embeddings -> n_layers x TransformerBlock
                     -> LayerNorm final -> logits (B, T, vocab_size)

- config.py -- GPTConfig, a configuração validada, e as configurações prontas:
  GPT_CONFIG_124M (o GPT-2 small do livro), GPT2_CONFIGS (exercício 4.2) e
  BASE_CONFIG (o modelo pequeno do projeto).
- gpt.py -- GPTModel, com contagem de parâmetros por componente.

Correspondência com os nomes do livro:

    livro (listing 4.7)    projeto
    -------------------    ------------------------------------
    tok_emb                embedding.token_embedding (Sprint 2)
    pos_emb                embedding.positional_embedding (Sprint 2)
    drop_emb               drop_emb
    trf_blocks             trf_blocks
    final_norm             final_norm
    out_head               out_head

A geração de texto, que usa o modelo mas não faz parte dele, está em
src/generation/.
"""

from .config import BASE_CONFIG, GPT2_CONFIGS, GPT2_VOCAB_SIZE, GPT_CONFIG_124M, GPTConfig
from .gpt import COMPONENTS, GPTModel

__all__ = [
    "BASE_CONFIG",
    "COMPONENTS",
    "GPT2_CONFIGS",
    "GPT2_VOCAB_SIZE",
    "GPT_CONFIG_124M",
    "GPTConfig",
    "GPTModel",
]
