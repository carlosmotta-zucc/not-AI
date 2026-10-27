# Entregável 4 — Inferência e geração de texto

**Do vetor de logits ao texto, um token por vez.** Documenta o procedimento de inferência, a relação entre logits, probabilidades e a escolha do próximo token, e o laço de geração autoregressiva. Os números citados vêm de [`experimentos/sprint04/resultados/geracao.md`](../../experimentos/sprint04/resultados/geracao.md) (G1, G2, G5 e G6) e de [`experimentos/sprint04/resultados/livro.md`](../../experimentos/sprint04/resultados/livro.md) (L7 e L8).

Código: [`src/generation/`](../../src/generation/). Testes: [`tests/test_generation.py`](../../tests/test_generation.py).

> O modelo ainda não foi treinado (isso é a Sprint 5). Os textos gerados aqui são incoerentes — o que se documenta é o **mecanismo**, que é o mesmo antes e depois do treino.

---

## 1. Visão geral

```
texto ─▶ Token IDs ─▶ GPTModel ─▶ logits (B, T, V) ─▶ última linha (B, V) ─▶ probabilidades ─▶ próximo token (B, 1)
            ▲                                                                                        │
            └──────────────────────────── anexado ao contexto ───────────────────────────────────────┘
```

O pacote tem duas partes, separadas pela pergunta que respondem:

| Arquivo | Responde | Funções |
| ------- | -------- | ------- |
| [`decoding.py`](../../src/generation/decoding.py) | Dado **um** vetor de logits, qual o próximo token? | `apply_top_k`, `next_token_probabilities`, `select_next_token` |
| [`autoregressive.py`](../../src/generation/autoregressive.py) | Como transformar um prompt num texto mais longo? | `generate`, `generate_text`, `text_to_token_ids`, `token_ids_to_text` |

`decoding.py` não conhece o modelo: recebe logits e devolve tokens, e por isso é testado com logits escritos à mão — inclusive os do vocabulário de 9 palavras da seção 5.3 do livro. `autoregressive.py` usa o modelo e repete a decisão de `decoding.py` em laço.

---

## 2. Logits, probabilidades e seleção

### 2.1 Logits

O `GPTModel` devolve `(B, T, V)`: para cada posição, um número real por token do vocabulário. A linha `t` é a previsão do token `t + 1` (Entregável 3, seção 4). Os logits não têm escala absoluta: só as **diferenças** entre eles importam. No G1, somar 100 a todos os 50 257 logits muda as probabilidades em no máximo 5,5e-10 — erro de arredondamento.

**Por que só a última linha.** Uma passada sobre `Every effort moves you` devolve 4 linhas de logits (G1). As três primeiras preveem `' effort'`, `' moves'` e `' you'`, que já estão no prompt; só a quarta prevê algo novo. No treino as quatro são úteis (cada uma é um exemplo); na geração, `generate` pega `logits[:, -1, :]`, de shape `(B, V)`.

### 2.2 Probabilidades

O softmax transforma os logits numa distribuição:

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

com `T` = temperatura (1 quando não há ajuste). Todas as probabilidades são positivas e somam 1. No modelo sem treino, a distribuição é quase uniforme: o token mais provável depois de `Every effort moves you` é `' ski'`, com logit 2,41 e probabilidade 1,89e-04 — 9,5 vezes o 1/V de uma escolha às cegas (G1).

### 2.3 Seleção do próximo token

| Estratégia | Como escolhe | Parâmetros em `generate` | Determinística? |
| ---------- | ------------ | ------------------------ | --------------- |
| Gulosa (*greedy*) | `argmax` dos logits | `temperature=0` (padrão) | Sim |
| Amostragem | Sorteio proporcional a `p_i` (`torch.multinomial`) | `temperature=1` | Não (reprodutível com gerador semeado) |
| Amostragem com temperatura | Sorteio sobre `softmax(z / T)` | `temperature=T > 0` | Não |
| Top-k | Só os k maiores logits ficam; o resto vai a −∞ antes do softmax | `top_k=k`, com `temperature > 0` | Não, exceto `top_k=1` |

Três propriedades que o código usa e os testes fixam:

- **O softmax é monotônico.** O token de maior logit é o de maior probabilidade, então a escolha gulosa nem calcula o softmax: `argmax` direto nos logits (o livro calcula, só para mostrar o caminho). Pela mesma razão, dividir pela temperatura não muda o `argmax` (`test_temperatura_nao_muda_o_argmax`).
- **A temperatura muda a forma, não a ordem.** `T < 1` amplia as diferenças entre logits e concentra a probabilidade nos primeiros; `T > 1` as reduz e achata a distribuição. No vocabulário de 9 palavras (L8), `forward` tem probabilidade 0,57 com T = 1, 0,99 com T = 0,1 e 0,24 com T = 5.
- **O top-k usa o mesmo truque da máscara causal.** Os logits fora do top-k viram −∞, e `exp(−∞) = 0` dá probabilidade zero exato a eles, com o restante já somando 1 — sem renormalização manual. A implementação mantém **exatamente** k tokens (`scatter` dos k valores sobre um tensor de −∞); a do livro compara com o menor dos k e manteria mais de k em caso de empate. Sem empate, as duas coincidem: no exemplo da seção 5.3, com k = 3, as probabilidades são `[0,0615, 0, 0, 0,5775, 0, 0, 0, 0,3610, 0]` nas duas.

A ordem é top-k primeiro, temperatura depois, como no livro. Ela não altera quais tokens sobrevivem, porque dividir por um número positivo não muda a ordem dos logits.

---

## 3. O procedimento de geração autoregressiva

`generate(model, token_ids, max_new_tokens, context_length=None, temperature=0.0, top_k=None, eos_id=None, generator=None)`:

1. **Preparar a inferência.** Guarda o modo do modelo, chama `model.eval()` e desliga o gradiente (`@torch.no_grad()`).
2. **Recortar o contexto.** `contexto = token_ids[:, -context_length:]` — só os últimos `context_length` tokens entram. Sem o recorte, um texto maior que a tabela de posições levantaria `ValueError`.
3. **Passada pelo modelo.** `model(contexto)` → `(B, T, V)`; fica só a última linha, `(B, V)`.
4. **Escolher o próximo token.** `select_next_token(logits, temperature, top_k, generator)` → `(B, 1)`.
5. **Fim de texto (opcional).** Com `eos_id`, a sequência que gera o token de fim para: ele é anexado uma vez e, se outras sequências do lote continuarem, ela é completada com ele para o lote continuar retangular. Quando todas param, o laço termina antes de `max_new_tokens`.
6. **Anexar.** `token_ids = cat([token_ids, próximo], dim=1)` → `(B, T + 1)`. O token produzido passa a fazer parte do contexto do passo seguinte — é isso que torna a geração **autoregressiva**.
7. **Repetir** os passos 2 a 6 `max_new_tokens` vezes.
8. **Restaurar** o modo original do modelo (num `finally`, mesmo se algo falhar no meio) e devolver `(B, T + n)`: o prompt intacto seguido dos tokens gerados.

### O laço observado passo a passo (G2)

Gulosa, a partir de `Every effort moves you`, uma chamada de `generate(..., max_new_tokens=1)` por linha:

| Passo | Tokens no contexto | Token escolhido | Probabilidade | Entropia (nats) |
| ----- | ------------------ | --------------- | ------------- | --------------- |
| 1 | 4 | `' ski'` | 1,89e-04 | 10,659 |
| 2 | 5 | `' lightweight'` | 1,76e-04 | 10,658 |
| 3 | 6 | `' deceit'` | 1,56e-04 | 10,659 |
| ... | ... | ... | ... | ... |
| 10 | 13 | `'>"'` | 1,84e-04 | 10,658 |

O contexto cresce um token por passo, e cada token escolhido aparece como entrada na linha seguinte. A entropia fica em 10,66 nats, perto do máximo ln V = 10,825, em todos os passos: sem treino, nenhum contexto torna o modelo mais confiante.

**A geração token a token e uma passada única dizem a mesma coisa.** Depois dos 10 passos, uma única passada sobre o texto final produz, nas mesmas posições, os mesmos logits que o laço calculou passo a passo (diferença máxima de 1,8e-06) e o mesmo `argmax` em todas elas. É a máscara causal vista de outro ângulo: o logit da posição t não depende de nada depois de t, então tanto faz se o futuro ainda não existe (no laço) ou se existe e está mascarado (na passada única). O teste `test_generate_guloso_coincide_com_a_passada_completa` fixa essa equivalência.

**O custo da repetição.** Para gerar 10 tokens, o laço passou 85 posições pelo modelo, porque cada passo refaz o contexto inteiro. Em 200 tokens com `context_length` 128 (G6), foram 17 850 posições — 89 por token novo. Os logits das posições antigas saem iguais aos do passo anterior; um cache de chaves e valores (KV cache) evitaria recalculá-los, e fica fora do escopo do livro.

### A janela de contexto (G6)

Enquanto o texto tem menos de `context_length` tokens, cada passo custa um pouco mais que o anterior: o contexto cresce. Com a `BASE_CONFIG`, o contexto enche no passo 125 (4 tokens de prompt + 124 gerados = 128). Dali em diante, a janela **desliza**: entra o token novo, sai o mais antigo, e o custo por passo para de crescer. Ao fim de 200 tokens, o texto tem 204 tokens e o modelo enxerga só os últimos 128 — o começo do prompt já foi esquecido. É o limite rígido da tabela de posições da Sprint 2.

---

## 4. Inferência: `eval()` e `no_grad()`

**`eval()`** desliga o dropout. O G5 mostra o que acontece sem ele: a mesma decodificação gulosa, rodada duas vezes num laço manual com o modelo em `train()`, produz textos diferentes em 28 das 30 posições, divergindo já no segundo token. Pela `generate`, que força `eval()`, as duas execuções são idênticas. Com o dropout ligado, a escolha "determinística" deixa de ser, e a amostragem ganharia um ruído que ninguém pediu.

Por isso `generate` não confia em quem chama: ela coloca o modelo em `eval()` e, no fim, devolve o modo original — o teste `test_generate_e_deterministico_e_restaura_o_modo` gera com o modelo em `train()` e confere as duas coisas.

**`no_grad()`** desliga a construção do grafo de gradiente. Nada na geração vai ter `backward`, e guardar as ativações de cada passo para ele só gastaria memória.

---

## 5. Reprodutibilidade

A aleatoriedade da amostragem vem de um `torch.Generator` passado como argumento, e não da seed global. Dois efeitos (G5):

- A mesma estratégia com o gerador na mesma seed produz o mesmo texto; com outra seed, um texto diferente. A gulosa não usa o gerador e é igual com qualquer seed.
- O gerador semeado reproduz os sorteios do livro, que usam a seed global: os 1 000 sorteios do vocabulário de 9 palavras com seed 123 dão as frequências impressas (73 `closer`, 582 `forward`, 2 `inches`, 343 `toward`), e `pizza` sai 32 vezes com T = 5 (exercício 5.1).

O exercício 5.3 pergunta como tornar `generate` determinística. As duas respostas funcionam no projeto: `temperature=0` (com qualquer `top_k`) e `top_k=1` (com qualquer temperatura positiva) — nos dois casos, os 1 000 sorteios saem todos `forward` (L8).

---

## 6. Conferência com o livro (L7)

A geração da seção 4.7, refeita com `generate` e o `GPTModel` de 124M com seed 123:

| Iteração | Contexto (Token IDs) | Próximo token |
| -------- | -------------------- | ------------- |
| 1 | `[15496, 11, 314, 716]` | 27018 `' Feature'` |
| 2 | `[..., 716, 27018]` | 24086 `'iman'` |
| 3 | `[..., 27018, 24086]` | 47843 `' Bye'` |
| 4 | `[..., 24086, 47843]` | 30961 `'swick'` |
| 5 | `[..., 47843, 30961]` | 42348 `'attribute'` |
| 6 | `[..., 30961, 42348]` | 7267 `' argue'` |

Resultado: `Hello, I am Featureiman Byeswickattribute argue`, os mesmos 10 Token IDs e o mesmo texto do livro. O texto é incoerente pelo motivo que o livro dá — os pesos são os sorteados pela seed — e reproduzi-lo exatamente mostra que o modelo e o laço fazem a mesma conta do livro.

---

## 7. Verificação

**Testes** (18 em `tests/test_generation.py`, todos passando): top-k com exatamente k sobreviventes, inclusive com empate; probabilidade zero fora do top-k; temperatura que concentra e achata sem mudar o `argmax`; parâmetros inválidos rejeitados; gulosa = `argmax`, e `top_k=1` determinístico; amostragem reprodutível com gerador; `generate` preservando o prompt, coincidindo com a passada completa, ultrapassando o `context_length` com o recorte, determinística e restaurando o modo; `eos` encerrando a geração e completando o lote; conversões texto ↔ Token IDs.

**Checkpoints do livro** (L7 e L8 de `livro.md`, todos conferem): a geração de "Hello, I am"; a escolha gulosa, as frequências de amostragem, as probabilidades por temperatura, o top-3 e os exercícios 5.1 e 5.3 do vocabulário de 9 palavras.

```bash
python tests/test_generation.py
python experimentos/sprint04/geracao_experimentos.py
```
