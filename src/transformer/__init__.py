"""Transformer Block: o bloco que o GPT empilha (Sprint 4, capítulo 4).

    (B, T, emb_dim) -> [LayerNorm -> MultiHeadAttention -> Dropout -> +x]
                    -> [LayerNorm -> FeedForward        -> Dropout -> +x]
                    -> (B, T, emb_dim)

Quatro peças, cada uma num arquivo, na ordem em que o livro as constrói:

- layer_norm.py -- LayerNorm: média 0 e variância 1 por token (seção 4.2).
- gelu.py -- GELU: a não linearidade da FeedForward (seção 4.3).
- feed_forward.py -- FeedForward: Linear -> GELU -> Linear, 4x de expansão
  (seção 4.3).
- block.py -- TransformerBlock: as peças acima mais a MultiHeadAttention da
  Sprint 3 e as conexões residuais (seções 4.4 e 4.5).

E uma peça didática, fora do GPT:

- shortcut_demo.py -- ShortcutDemoNetwork: a rede de cinco camadas da seção
  4.4, que isola o efeito do atalho sobre o gradiente.

Decisões que o resto do código segue:

- Implementação própria de LayerNorm e GELU. `nn.LayerNorm` e `nn.GELU` entram
  só como baseline nos testes, como `nn.MultiheadAttention` na Sprint 3.
- Pre-LayerNorm, como no GPT-2: a normalização fica dentro de cada ramo, antes
  da subcamada.
- O bloco recebe argumentos explícitos (`emb_dim`, `context_length`,
  `n_heads`, ...), como a MultiHeadAttention da Sprint 3, e não o `cfg` do
  livro. Quem lê a configuração é o GPTModel, em src/model/.
- Nomes dos atributos iguais aos do livro (`att`, `ff`, `norm1`, `norm2`,
  `drop_shortcut`), para o glossário do capítulo 4 valer para o código.

Experimentos em experimentos/sprint04/, documentação em docs/sprint04/.
"""

from .block import TransformerBlock
from .feed_forward import EXPANSION_FACTOR, FeedForward
from .gelu import GELU
from .layer_norm import LayerNorm
from .shortcut_demo import ShortcutDemoNetwork, mean_weight_gradients

__all__ = [
    "EXPANSION_FACTOR",
    "FeedForward",
    "GELU",
    "LayerNorm",
    "ShortcutDemoNetwork",
    "TransformerBlock",
    "mean_weight_gradients",
]
