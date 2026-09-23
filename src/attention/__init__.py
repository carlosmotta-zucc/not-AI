"""Mecanismos de atencao: o nucleo do Transformer (Sprint 3, capitulo 3).

    (B, T, emb_dim) -> Q, K, V -> scores -> mascara -> softmax -> dropout
                    -> pesos (B, T, T) -> contexto (B, T, d_out)

Quatro degraus: simplified.py (sem pesos), self_attention.py (Q/K/V
treinaveis), causal.py (esconde o futuro), multi_head.py (em paralelo).

Decisoes que o resto do codigo segue:

- Funcao nucleo. `scaled_dot_product_attention(queries, keys, values,
  mask=None, dropout=None, scale=True)`: a conta e escrita uma vez e todas as
  classes chamam ela. Opera nas duas ultimas dimensoes, entao serve a
  (B, T, d) e a (B, H, T, head_dim) sem ramificar.
- Retorno. Ela e os `forward` devolvem `(context, weights)`; os pesos vao para
  os heatmaps sem recalculo. Em troca, o capitulo 4 escreve `x, _ = att(x)`.
- Mascara. Bool (T, T), True = bloqueado, `masked_fill(-inf)` antes do softmax
  -- depois exigiria renormalizar as linhas na mao.
- Dropout. Vem da classe que chama: funcao solta nao tem train()/eval().
- Nomes. `d_in`, `d_out`, `context_length`, `num_heads`, `dropout`,
  `qkv_bias`, com que o capitulo 4 instancia a Multi-Head.
- Configuracao base. A da Sprint 2 (emb_dim 256, context_length 128,
  batch_size 8, seed 42) mais 4 heads, dropout 0.1, qkv_bias False. Da
  head_dim 64, o do GPT-2.
- Frases de teste. "The bank of the river" e "The bank of the city" nos
  experimentos E4 e E5: prefixo igual, entao mudanca nos pesos de "bank" so
  pode ter vindo do final.

Experimentos em experimentos/sprint03/, analise em docs/sprint03/analise.md.
"""

from .causal import CausalAttention, causal_mask
from .multi_head import (
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
    head_dimension,
)
from .scaled_dot_product import scaled_dot_product_attention
from .self_attention import SelfAttention
from .simplified import EXAMPLE_INPUTS, EXAMPLE_WORDS, simplified_attention

__all__ = [
    "EXAMPLE_INPUTS",
    "EXAMPLE_WORDS",
    "CausalAttention",
    "MultiHeadAttention",
    "MultiHeadAttentionWrapper",
    "SelfAttention",
    "causal_mask",
    "head_dimension",
    "scaled_dot_product_attention",
    "simplified_attention",
]
