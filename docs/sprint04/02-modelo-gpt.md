# Entregável 3 — Construção do modelo GPT

**Dos blocos ao modelo que devolve logits para o vocabulário.** Documenta a organização do `GPTModel`, o caminho de uma sequência de Token IDs até os logits e a distribuição dos parâmetros entre os componentes. Os números citados vêm de [`experimentos/sprint04/resultados/arquitetura.md`](../../experimentos/sprint04/resultados/arquitetura.md) (A1 a A3) e de [`experimentos/sprint04/resultados/livro.md`](../../experimentos/sprint04/resultados/livro.md) (L1 e L6).

Código: [`src/model/`](../../src/model/). Testes: [`tests/test_gpt_model.py`](../../tests/test_gpt_model.py) e [`tests/test_gpt_pipeline.py`](../../tests/test_gpt_pipeline.py).

---

## 1. Visão geral

O `GPTModel` é um Transformer *decoder-only*: não há encoder, e todo o modelo é uma pilha de blocos com atenção causal. Ele recebe Token IDs e devolve, para cada posição, um logit por token do vocabulário:

```
(B, T) Token IDs
  └─ InputEmbedding (Sprint 2) ─── tabela de tokens + tabela de posições ──── (B, T, D)
  └─ Dropout ──────────────────────────────────────────────────────────────── (B, T, D)
  └─ n_layers × TransformerBlock (Entregável 2) ─── fluxo residual ─────────── (B, T, D)
  └─ LayerNorm final ──────────────────────────────────────────────────────── (B, T, D)
  └─ cabeça de saída: Linear(D, V, sem bias) ──────────────────────────────── (B, T, V)
```

O modelo usa os componentes das sprints anteriores sem alteração: os embeddings da Sprint 2 (`InputEmbedding`), a atenção da Sprint 3 (dentro de cada bloco) e o tokenizador BPE da Sprint 2 para produzir os Token IDs.

---

## 2. Organização no código

| Componente | Atributo do `GPTModel` | Classe | Arquivo | Origem |
| ---------- | ---------------------- | ------ | ------- | ------ |
| Configuração | `cfg` | `GPTConfig` | `src/model/config.py` | Sprint 4 |
| Tabela de tokens | `embedding.token_embedding` | `TokenEmbedding` | `src/embeddings/token_embeddings.py` | Sprint 2 |
| Tabela de posições | `embedding.positional_embedding` | `PositionalEmbedding` | `src/embeddings/positional.py` | Sprint 2 |
| Dropout dos embeddings | `drop_emb` | `nn.Dropout` | — | Sprint 4 |
| Pilha de blocos | `trf_blocks` | `nn.Sequential` de `TransformerBlock` | `src/transformer/block.py` | Sprint 4 |
| Atenção (dentro de cada bloco) | `trf_blocks[i].att` | `MultiHeadAttention` | `src/attention/multi_head.py` | Sprint 3 |
| LayerNorm final | `final_norm` | `LayerNorm` | `src/transformer/layer_norm.py` | Sprint 4 |
| Cabeça de saída (Language Model Head) | `out_head` | `nn.Linear` | — | Sprint 4 |

Os nomes dos atributos são os do livro (listing 4.7), com uma exceção: `tok_emb` e `pos_emb` do livro são `embedding.token_embedding` e `embedding.positional_embedding`, porque os embeddings são os da Sprint 2, reaproveitados.

### A configuração

`GPTConfig` é uma dataclass congelada com os campos do `GPT_CONFIG_124M` do livro, validada na construção (`emb_dim` indivisível por `n_heads`, por exemplo, falha ali, e não no forward). Três configurações prontas:

| | `BASE_CONFIG` (projeto) | `GPT_CONFIG_124M` (livro) |
| - | ---------------------- | ------------------------- |
| `vocab_size` | 50 257 | 50 257 |
| `context_length` | 128 | 1 024 |
| `emb_dim` | 256 | 768 |
| `n_heads` (`head_dim`) | 4 (64) | 12 (64) |
| `n_layers` | 4 | 12 |
| FeedForward | 1 024 | 3 072 |
| `drop_rate` | 0,1 | 0,1 |
| `qkv_bias` | False | False |
| Parâmetros | 28 920 832 | 163 009 536 |
| Pesos em float32 | 110,32 MB | 621,83 MB |

A `BASE_CONFIG` continua a configuração das Sprints 2 e 3 (`emb_dim` 256, `context_length` 128, 4 cabeças de 64) e acrescenta 4 blocos: é o modelo que a Sprint 5 vai treinar em CPU. A `GPT_CONFIG_124M` existe para reproduzir o livro. `GPT2_CONFIGS` traz os quatro tamanhos do GPT-2 (exercício 4.2).

Além dos campos do livro, a configuração tem `ff_hidden_dim` (a largura da FeedForward, variada no A6), `drop_rate_emb`, `drop_rate_attn` e `drop_rate_shortcut` (exercício 4.3) e `tie_weights` (seção 5). Todos têm padrão que reproduz o livro.

---

## 3. O caminho de uma sequência

Shapes medidos no modelo base com o primeiro lote do The Verdict (A1): B = 8, T = 128, D = 256, V = 50 257.

| # | Etapa | Shape | O que acontece |
| - | ----- | ----- | -------------- |
| 1 | Token IDs | (8, 128) | Saída do `DataLoader` da Sprint 2, dtype `long` |
| 2 | Tabela de tokens | (8, 128, 256) | Cada ID escolhe uma linha da tabela `(V, D)` |
| 3 | Tabela de posições | (128, 256) | Uma linha por posição, sem dimensão de lote |
| 4 | Soma | (8, 128, 256) | O broadcasting replica as posições para os 8 exemplos |
| 5 | Dropout | (8, 128, 256) | Só em `train()` |
| 6 | 4 × TransformerBlock | (8, 128, 256) | Cada bloco soma ao fluxo residual a correção da atenção e da FeedForward |
| 7 | LayerNorm final | (8, 128, 256) | Cada vetor volta a média 0 e variância 1 |
| 8 | Cabeça de saída | (8, 128, 50 257) | Cada vetor de D dimensões vira V logits |

O shape é `(B, T, D)` do passo 2 ao 7: a profundidade do modelo não muda nenhuma dimensão. Só a primeira operação (indexar a tabela) e a última (projetar no vocabulário) mudam a última dimensão.

**Por que existe a LayerNorm final.** Como o GPT-2 é Pre-LayerNorm, o fluxo residual nunca é normalizado no meio da pilha: cada bloco soma algo a ele, e a escala cresce com a profundidade (A3: desvio de 1,425 com 1 bloco, 1,610 com 12). A norma final devolve a escala antes da projeção no vocabulário, e por isso os logits saem com o mesmo desvio — 0,577 — em todas as profundidades de 1 a 12 blocos. Sem ela, a "temperatura" efetiva dos logits dependeria do número de blocos.

---

## 4. O que os logits representam

A linha `t` dos logits é a previsão do token `t + 1`. Por causa da máscara causal, ela foi calculada só com os tokens `0..t` — o teste `test_logit_da_posicao_t_nao_depende_do_futuro` altera os tokens depois de `t` e confirma que os logits até `t` não mudam. Duas consequências:

- **No treino (Sprint 5)**, uma passada de T tokens dá T problemas de "prever o próximo", todos usados na perda. Os alvos do `DataLoader` da Sprint 2 — a entrada deslocada de uma posição — casam exatamente com as linhas dos logits: o teste de integração já calcula a cross-entropy entre os dois.
- **Na geração (Entregável 4)**, só a última linha interessa: as outras preveem tokens que já estão no texto.

Os logits ainda não são probabilidades: são números reais, sem limite, que só depois do softmax viram uma distribuição. Essa conversão fica fora do modelo, a cargo de quem usa (a perda ou a geração).

### O modelo sem treino (A2)

| | Modelo base | GPT-2 small | Distribuição uniforme |
| - | ----------- | ----------- | --------------------- |
| Desvio dos logits | 0,5774 | 0,5772 | — |
| Maior probabilidade (média por posição) | 1,93e-04 | 1,96e-04 | 1,99e-05 |
| Probabilidade dada ao token certo | 2,05e-05 | 2,02e-05 | 1,99e-05 |
| Entropia da previsão (nats) | 10,658 | 10,658 | ln V = 10,825 |
| Cross-entropy (nats) | 10,976 | 10,975 | 10,825 |

O modelo sem treino é quase um chute uniforme: dá ao token certo 2,05e-05 de probabilidade, praticamente o 1/V = 1,99e-05 de quem não sabe nada. É o ponto de partida do treino — uma perda inicial perto de 11 nats.

O desvio dos logits é o mesmo nos dois modelos, apesar de D, profundidade e seed diferentes, e tem uma explicação exata. A LayerNorm final entrega vetores com variância 1; a cabeça de saída é inicializada pelo PyTorch com pesos uniformes em ±1/√D, de variância 1/(3D). Cada logit soma D produtos, então sua variância é D · 1 · 1/(3D) = 1/3, e o desvio é 1/√3 = 0,577 — independente de D. Para logits gaussianos com variância σ², a entropia do softmax sobre V tokens é aproximadamente ln V − σ²/2 = 10,825 − 0,167 = 10,658, exatamente o valor medido. A LayerNorm final é o que torna essa escala previsível.

---

## 5. Parâmetros por componente

| Componente | Modelo base | Fração | GPT-2 small | Fração |
| ---------- | ----------- | ------ | ----------- | ------ |
| Tabela de tokens | 12 865 792 | 44,5% | 38 597 376 | 23,7% |
| Tabela de posições | 32 768 | 0,1% | 786 432 | 0,5% |
| Atenção (todos os blocos) | 1 049 600 | 3,6% | 28 320 768 | 17,4% |
| FeedForward (todos os blocos) | 2 102 272 | 7,3% | 56 669 184 | 34,8% |
| LayerNorms dos blocos | 4 096 | 0,0% | 36 864 | 0,0% |
| LayerNorm final | 512 | 0,0% | 1 536 | 0,0% |
| Cabeça de saída | 12 865 792 | 44,5% | 38 597 376 | 23,7% |
| **Total** | **28 920 832** | | **163 009 536** | |

**No modelo base, 89,1% dos parâmetros não estão nos blocos.** A tabela de tokens e a cabeça de saída têm `V × D` parâmetros cada, e com V = 50 257 isso pesa mais que os quatro blocos juntos (3,16 milhões). No GPT-2 small, mais largo e mais profundo, a fração cai para 47,8%. O A5 mostra a transição com `emb_dim`: a tabela cresce linear em D, os blocos crescem com D², e os blocos só passam a dominar em modelos largos e profundos.

**Por que 163 milhões e não 124.** O GPT-2 original usa *weight tying*: a cabeça de saída reaproveita a tabela de tokens, que tem o mesmo shape `(50 257, 768)` — `nn.Linear` guarda o peso como `(out_features, in_features)`. Sem contar a cabeça, o total é 124 412 160, o número oficial. O `GPTModel` implementa as duas opções:

- `tie_weights=False` (padrão, como no livro): duas matrizes independentes. O livro justifica a escolha pela qualidade do treino.
- `tie_weights=True`: `out_head.weight` passa a ser o **mesmo objeto** `Parameter` da tabela de tokens, e o gradiente das duas pontas é somado nele. O teste `test_weight_tying_compartilha_o_mesmo_tensor` confere a identidade e a contagem. No modelo base, o tying tiraria 12 865 792 parâmetros dos 28 920 832 — 44,5% do modelo.

**Memória.** Em float32, 4 bytes por parâmetro: 110,32 MB no modelo base e 621,83 MB no GPT-2 small, só de pesos. O exercício 4.2 (L6) estende a conta aos quatro tamanhos do GPT-2, até 1 637 792 000 parâmetros e 6 247,68 MB no XL. Os modelos grandes são contados no device `meta`, que cria os tensores sem alocar memória.

---

## 6. Decisões de implementação

| Decisão | Livro | Projeto | Motivo |
| ------- | ----- | ------- | ------ |
| Configuração | Dicionário `cfg` | Dataclass `GPTConfig` congelada e validada | Erro na construção, não no forward; `replace(BASE_CONFIG, n_layers=8)` para os experimentos variarem um campo de cada vez. |
| Embeddings | Duas `nn.Embedding` novas | `InputEmbedding` da Sprint 2 | Reuso do que já estava implementado e testado. A ordem dos sorteios (tokens, posições) é a mesma do livro. |
| Esqueleto (`DummyGPTModel`) | Classe separada com blocos e norma falsos | `GPTModel` com `n_layers=0` e `final_norm` trocada por identidade | Não duplica a classe; reproduz os logits do livro (L1). |
| Weight tying | Só calculado, não implementado | Opção `tie_weights` | Permite medir o efeito em vez de só descontar da contagem. |
| Contagem | `sum(p.numel() ...)` no notebook | `parameter_count()`, `parameter_breakdown()`, `size_in_megabytes()` | As tabelas da seção 5 e dos experimentos saem do próprio modelo. |

---

## 7. Verificação

**Testes** (18, todos passando):

- `tests/test_gpt_model.py` (14): shape dos logits, erro acima do `context_length`, componentes somando o total, contagem pela fórmula, números do livro para o GPT-2 small e o XL, weight tying, causalidade no modelo inteiro, logits de um prefixo iguais aos da sequência completa, dropout, taxas separadas, configuração inválida, esqueleto com zero blocos e `GPT2_VOCAB_SIZE` igual ao vocabulário do BPE.
- `tests/test_gpt_pipeline.py` (4): The Verdict → BPE → `DataLoader` → `GPTModel` na `BASE_CONFIG`, com shapes, perda inicial a menos de 1 nat de ln V, gradiente da cross-entropy em todos os parâmetros e geração a partir de um trecho do corpus.

**Checkpoints do livro** (L1 e L6 de `livro.md`, todos conferem): os Token IDs do lote de exemplo; os 48 valores de logits impressos para o `DummyGPTModel` e os 48 do `GPTModel` com seed 123, a menos de meia unidade na quarta casa; 163 009 536 parâmetros, 124 412 160 com weight tying, 621,83 MB; os shapes da tabela de tokens e da cabeça; o GPT-2 XL do exercício 4.2; as taxas do exercício 4.3. Reproduzir os logits é a verificação mais forte possível sem treino: mostra que embeddings, 12 blocos, norma final e cabeça fazem a mesma conta do livro, com os mesmos pesos.

```bash
python tests/test_gpt_model.py
python tests/test_gpt_pipeline.py
python tests/test_book_checkpoints.py
```
