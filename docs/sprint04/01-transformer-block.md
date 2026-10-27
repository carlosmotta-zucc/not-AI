# Entregável 2 — Implementação do Transformer Block

**Da atenção isolada ao bloco que o GPT empilha.** Documenta a estrutura do `TransformerBlock`, a função de cada componente e as dimensões dos tensores que passam por ele. Os números citados vêm de [`experimentos/sprint04/resultados/arquitetura.md`](../../experimentos/sprint04/resultados/arquitetura.md) (experimentos A1 e A9) e de [`experimentos/sprint04/resultados/livro.md`](../../experimentos/sprint04/resultados/livro.md) (L2 a L5).

Código: [`src/transformer/`](../../src/transformer/). Testes: [`tests/test_transformer_components.py`](../../tests/test_transformer_components.py) e [`tests/test_transformer_block.py`](../../tests/test_transformer_block.py).

---

## 1. Visão geral

A Sprint 3 entregou uma `MultiHeadAttention` que transforma `(B, T, emb_dim)` em `(B, T, emb_dim)`: cada token sai com um vetor que leva em conta os tokens anteriores. Sozinha, ela não basta para um modelo profundo, por três motivos que o bloco resolve:

| Lacuna da atenção sozinha | Peça que resolve | Arquivo |
| ------------------------- | ---------------- | ------- |
| É uma média ponderada dos *values*: mistura tokens, mas transforma pouco o conteúdo de cada um | FeedForward com GELU, aplicada a cada posição | `feed_forward.py`, `gelu.py` |
| Empilhada muitas vezes, a escala das ativações deriva com a profundidade | LayerNorm antes de cada subcamada | `layer_norm.py` |
| Empilhada muitas vezes, o gradiente precisa atravessar todas as camadas para chegar às primeiras | Conexões residuais (atalhos) | `block.py` |

O bloco junta as quatro peças — atenção, LayerNorm, FeedForward e atalhos — e mantém o shape de entrada na saída. É essa invariância que permite empilhar `n_layers` blocos no GPT (Entregável 3).

---

## 2. Estrutura

```
x ─┬─ LayerNorm 1 ─ MultiHeadAttention ─ Dropout ─(+)─┬─ LayerNorm 2 ─ FeedForward ─ Dropout ─(+)─ saída
   └──────────────────────────────────────────────┘   └───────────────────────────────────────┘
                 atalho da atenção                                atalho da FeedForward
```

Em duas linhas, que são literalmente o `forward` de [`block.py`](../../src/transformer/block.py):

```
x = x + Dropout(Atenção(LayerNorm₁(x)))
x = x + Dropout(FeedForward(LayerNorm₂(x)))
```

O `x` que atravessa as duas linhas é o **fluxo residual**. Cada subcamada lê uma cópia normalizada dele e devolve uma correção que é **somada** — nunca uma substituição. A normalização fica dentro do ramo, antes da subcamada: é a arquitetura **Pre-LayerNorm** do GPT-2. O Transformer original normalizava depois da soma (Post-LayerNorm); a seção 6 do documento de experimentos compara as duas.

---

## 3. Componentes

### 3.1 LayerNorm — `src/transformer/layer_norm.py`

**O que faz.** Para cada vetor da última dimensão, subtrai a média e divide pelo desvio padrão, e depois aplica uma escala e um deslocamento treináveis:

```
LayerNorm(x) = scale · (x − μ) / √(σ² + ε) + shift,   μ e σ² calculados sobre os emb_dim valores de cada token
```

**Função no modelo.** Garante que a atenção e a FeedForward recebam entradas na mesma escala em qualquer bloco, por mais que o fluxo residual tenha crescido. Sem isso, o softmax da atenção e a GELU trabalhariam em regimes diferentes em cada profundidade.

**Detalhes de implementação.**

- Normaliza em `dim=-1`, a dimensão das features. Cada token é tratado sozinho: o teste `test_layer_norm_normaliza_cada_token_sozinho` troca um token e confirma que só a linha dele muda. É a diferença para a BatchNorm, que mistura exemplos do lote e não funcionaria com lote de 1 na geração.
- `scale` começa em 1 e `shift` em 0: a saída inicial é a normalização pura, e o treino pode desfazê-la canal a canal se for útil. São `2 · emb_dim` parâmetros — 512 no modelo base, 1 536 no GPT-2 small.
- Variância enviesada (`unbiased=False`), dividindo por n e não por n − 1, por compatibilidade com o GPT-2 original. Com ela, a implementação própria coincide com `nn.LayerNorm` a 2,4e-07 (L2); com n − 1, cada saída sairia multiplicada por √(767/768) = 0,99935.
- O teste `test_layer_norm_ignora_escala_e_deslocamento_da_entrada` mostra a propriedade que importa para o bloco: `LayerNorm(7x − 2) = LayerNorm(x)`. A escala do fluxo residual não chega à subcamada.

**Contribuição para a estabilidade (A9).** Num modelo de 12 blocos sem nenhuma LayerNorm, o desvio padrão do fluxo residual sobe de 1,41 para 2,15 ao longo da pilha, o desvio dos logits dobra (0,577 → 1,241) e a perda inicial sobe de 11,00 para 11,61. Com as LayerNorms, os logits saem com desvio 0,577 em qualquer profundidade (A3: de 1 a 12 blocos, sempre 0,577).

### 3.2 Multi-Head Attention — `src/attention/multi_head.py` (Sprint 3)

**O que faz.** Projeta cada token em Q, K e V, divide em `n_heads` cabeças, calcula a atenção causal em cada uma e junta as cabeças com a `out_proj`. Detalhada na [análise da Sprint 3](../sprint03/analise.md).

**Função no bloco.** É a única peça que **mistura posições**: o vetor de saída da posição t depende dos tokens 0..t. LayerNorm e FeedForward agem token a token, então a causalidade do bloco inteiro é a da máscara da atenção — o teste `test_causalidade_sobrevive_ao_bloco` confirma que alterar tokens futuros não muda nenhuma saída anterior.

**Como entra.** Sem alteração em relação à Sprint 3, com `d_in = d_out = emb_dim` — a condição do atalho, já fixada pelo teste de integração daquela sprint. O retorno `(contexto, pesos)` é desempacotado como `attended, _ = self.att(...)`: os pesos servem aos heatmaps, não ao bloco.

### 3.3 FeedForward e GELU — `src/transformer/feed_forward.py` e `gelu.py`

**O que faz.** Duas camadas lineares com a GELU no meio, aplicadas a cada posição separadamente:

```
(B, T, D) ──Linear──▶ (B, T, 4D) ──GELU──▶ (B, T, 4D) ──Linear──▶ (B, T, D)
```

**Função no modelo.** É onde está a não linearidade do bloco e a maior parte dos seus parâmetros: 66,6% do bloco base (525 568 de 788 992) e 2,00x a atenção no GPT-2 small (4 722 432 contra 2 360 064, exercício 4.1). Se a atenção decide **de onde** trazer informação, a FeedForward decide **o que fazer** com ela em cada token. A expansão para 4D dá à GELU um espaço maior para separar padrões; a contração de volta para D é o que permite somar a saída ao fluxo residual.

**A GELU.** `GELU(x) = x · Φ(x)`, implementada pela aproximação por tanh que o GPT-2 usou no treino. Diferenças para a ReLU que importam no treino:

| x | −3 | −1 | −0,75 | 0 | 1 | 3 |
| - | -- | -- | ----- | - | - | - |
| GELU(x) | −0,0036 | −0,1588 | −0,1700 | 0 | 0,8412 | 2,9964 |
| ReLU(x) | 0 | 0 | 0 | 0 | 1 | 3 |

Os negativos não são zerados (GELU(−1) = −0,159), então um neurônio com entrada negativa continua recebendo gradiente. A curva é suave, sem o bico da ReLU em 0, e tem um único mínimo em x = −0,7525 (L3), o ponto que o livro cita como exceção ao gradiente não nulo. A diferença entre a aproximação por tanh e a GELU exata (erf) é de no máximo 4,7e-04 em [−3, 3].

### 3.4 Conexões residuais — `src/transformer/block.py`

**O que fazem.** Somam a entrada de cada subcamada à saída dela: `x + f(x)`.

**Função no modelo.** A derivada de `x + f(x)` em relação a `x` é `1 + f'(x)`. O "1" cria um caminho em que o gradiente atravessa o bloco sem ser multiplicado pelas derivadas da atenção e da FeedForward. Numa pilha sem atalho, o gradiente que chega ao primeiro bloco é o produto das derivadas de todos os blocos acima — e um produto de muitos fatores menores que 1 desaparece.

**Evidência.**

- Na rede de demonstração da seção 4.4 do livro (L4), reproduzida número a número: sem atalho, o gradiente da primeira camada é 25x menor que o da última; com atalho, 6x. A primeira camada recebe 1 099x mais gradiente com o atalho.
- No próprio GPT (A9): sem atalho **e** sem LayerNorm, o gradiente médio do bloco 1 é 1,8e-18, contra 2,3e-05 no bloco 12 — treze ordens de grandeza de diferença. Com qualquer um dos dois mecanismos, o gradiente chega aos 12 blocos praticamente igual (razão bloco 1 / bloco 12 entre 0,96 e 1,11). A análise completa está na seção 6 do [documento de experimentos](04-experimentos-e-analise.md).
- O teste `test_com_subcamadas_zeradas_o_bloco_e_a_identidade` zera a última projeção dos dois ramos e confirma que o bloco devolve a entrada bit a bit: o atalho é o caminho padrão, e cada subcamada é uma correção sobre ele.

### 3.5 Dropout — `drop_shortcut` e o dropout da atenção

**O que faz.** Durante o treino, zera ao acaso uma fração `p` das unidades e multiplica as restantes por 1/(1−p). Em `eval()` é a identidade.

**Onde está no bloco.** Em dois lugares: nos pesos de atenção, dentro da `MultiHeadAttention` (Sprint 3), e na saída de cada subcamada, **antes** da soma residual (`drop_shortcut`). Não age sobre o fluxo residual em si — o atalho passa intacto. As duas taxas podem ser separadas (`drop_rate` e `attn_drop_rate`), o que implementa o exercício 4.3; o A8 mede o efeito de cada uma.

---

## 4. Dimensões dos tensores

Medidas por ganchos de forward no modelo base (A1), com o primeiro lote do The Verdict: B = 8 sequências, T = 128 tokens, D = `emb_dim` = 256, H = 4 cabeças de `head_dim` = 64, FeedForward com 4D = 1 024.

| Etapa dentro do bloco | Entrada | Saída |
| --------------------- | ------- | ----- |
| LayerNorm 1 | (8, 128, 256) | (8, 128, 256) |
| Projeção Q (K e V iguais) | (8, 128, 256) | (8, 128, 256) |
| Q, K e V divididos em cabeças (`view` + `transpose`) | — | (8, 4, 128, 64) cada |
| Atenção multi-cabeça: contexto e pesos | (8, 128, 256) | (8, 128, 256) e (8, 4, 128, 128) |
| Soma com o atalho 1 | (8, 128, 256) + (8, 128, 256) | (8, 128, 256) |
| LayerNorm 2 | (8, 128, 256) | (8, 128, 256) |
| FeedForward, Linear 1 | (8, 128, 256) | (8, 128, 1024) |
| FeedForward, GELU | (8, 128, 1024) | (8, 128, 1024) |
| FeedForward, Linear 2 | (8, 128, 1024) | (8, 128, 256) |
| Soma com o atalho 2 = saída do bloco | (8, 128, 256) + (8, 128, 256) | (8, 128, 256) |

Duas dimensões aparecem só **dentro** do bloco e não sobrevivem à saída: `head_dim` (consumida pelo produto escalar da atenção) e a camada expandida da FeedForward. O tensor mais largo do bloco é o de pesos de atenção, `(B, H, T, T)`, que cresce com T²; o mais largo em features é o da FeedForward, `(B, T, 4D)`.

---

## 5. Parâmetros do bloco

| Subpeça | Fórmula | Modelo base (D = 256) | GPT-2 small (D = 768) |
| ------- | ------- | --------------------- | --------------------- |
| Atenção (Q, K, V sem bias; `out_proj` com bias) | 4D² + D | 262 400 (33,3%) | 2 360 064 |
| FeedForward (duas Linear com bias, H = 4D) | 2·D·H + H + D | 525 568 (66,6%) | 4 722 432 |
| 2 LayerNorms | 4D | 1 024 (0,1%) | 3 072 |
| **Total** | | **788 992** | **7 085 568** |

Cada bloco a mais custa esses parâmetros, independentemente de `context_length` e do vocabulário.

---

## 6. Decisões de implementação

| Decisão | Livro | Projeto | Motivo |
| ------- | ----- | ------- | ------ |
| Construtor | `TransformerBlock(cfg)` | Argumentos explícitos (`emb_dim`, `context_length`, `n_heads`, ...) | Mesmo padrão da `MultiHeadAttention` da Sprint 3; o bloco pode ser testado sem montar uma configuração de GPT inteira. Quem lê a configuração é o `GPTModel`. |
| Dropout | Uma taxa para tudo | `drop_rate` e `attn_drop_rate` (opcional, herda a primeira) | Exercício 4.3. Sem o segundo argumento, o comportamento é o do livro. |
| `forward` | Cinco linhas por subcamada com a variável `shortcut` | `x = x + Dropout(subcamada(norma(x)))` | A forma de uma linha deixa explícito que o fluxo residual nunca é substituído, só incrementado. A ordem das operações — e, portanto, dos sorteios do dropout — é a mesma. |
| LayerNorm e GELU | Implementação própria | Implementação própria, comparada com `nn.LayerNorm` e `nn.GELU` nos testes | Regra do projeto: componentes que a sprint pede para implementar não são substituídos por prontos do PyTorch. |
| Ordem de criação | `att`, `ff`, `norm1`, `norm2` | A mesma | Com a mesma seed, os pesos sorteados são os do livro. É o que permite os checkpoints da seção 7. |

---

## 7. Verificação

**Testes** (21 no total, todos passando):

- `tests/test_transformer_components.py` (12): LayerNorm com média 0 e variância 1, igual ao `nn.LayerNorm`, invariante a escala e deslocamento da entrada e independente entre tokens; GELU igual ao `nn.GELU` nas duas formas e com gradiente nos negativos; FeedForward com shape preservado, contagem de parâmetros pela fórmula e sem mistura entre posições.
- `tests/test_transformer_block.py` (9): shape preservado, parâmetros do exercício 4.1, identidade com subcamadas zeradas, causalidade, erro acima do `context_length`, dropout só em `train()`, taxas separadas e gradiente em todos os parâmetros.

**Checkpoints do livro** (L2 a L5 de `livro.md`, todos conferem): a saída da camada de exemplo e a normalização passo a passo da seção 4.2; o mínimo da GELU e os shapes da figura 4.9; os dez gradientes da seção 4.4, com e sem atalho, até a sexta casa significativa; o shape do bloco da seção 4.5 e as contagens do exercício 4.1.

```bash
python tests/test_transformer_components.py
python tests/test_transformer_block.py
python experimentos/sprint04/livro_exemplos.py        # L2 a L5
python experimentos/sprint04/arquitetura_experimentos.py  # A1 e A9
```
