# Entregável 5 — Experimentação e análise

**O que cada hiperparâmetro custa e o que ele muda.** Cada experimento registra a configuração, os resultados e uma análise que relaciona a alteração ao comportamento do modelo e ao custo computacional. As tabelas completas são geradas pelos scripts e não são editadas à mão:

| Script | Resultados | Experimentos |
| ------ | ---------- | ------------ |
| [`livro_exemplos.py`](../../experimentos/sprint04/livro_exemplos.py) | [`livro.md`](../../experimentos/sprint04/resultados/livro.md) | L1–L8: reprodução dos exemplos do livro |
| [`arquitetura_experimentos.py`](../../experimentos/sprint04/arquitetura_experimentos.py) | [`arquitetura.md`](../../experimentos/sprint04/resultados/arquitetura.md) | A1–A9: arquitetura |
| [`geracao_experimentos.py`](../../experimentos/sprint04/geracao_experimentos.py) | [`geracao.md`](../../experimentos/sprint04/resultados/geracao.md) | G1–G7: inferência e geração |

Este documento interpreta os números de lá. Os experimentos A1, G1 e G2, que descrevem a estrutura e o procedimento, são discutidos nos documentos dos Entregáveis [2](01-transformer-block.md), [3](02-modelo-gpt.md) e [4](03-inferencia-e-geracao.md).

---

## 1. Metodologia

**Configuração de partida.** `BASE_CONFIG`: vocabulário do BPE do GPT-2 (50 257), `context_length` 128, `emb_dim` 256, 4 cabeças de 64, 4 blocos, FeedForward 1 024, dropout 0,1 — 28 920 832 parâmetros. Cada experimento muda **um** hiperparâmetro e mantém os outros, com `dataclasses.replace`.

**Dados.** O primeiro lote do The Verdict pelo `DataLoader` da Sprint 2 (8 sequências de 128 tokens, sem embaralhar), para perdas e estatísticas. O prompt `Every effort moves you` (seção 5.3 do livro) para a geração.

**O que se pode medir sem treino.** O modelo ainda não foi treinado, então não faz sentido medir qualidade de texto. O que se mede é o que já existe na inicialização e não depende do treino: **custo** (parâmetros, memória, FLOPs, tempo) e **comportamento numérico** (escala das ativações, escala dos logits, fluxo do gradiente, forma da distribuição do próximo token, determinismo). São justamente as propriedades que decidem se o modelo *pode* ser treinado — a Sprint 5 mede o resto.

**Custo.** Parâmetros contados no próprio modelo (`parameter_count()`); modelos grandes no device `meta`, sem alocar memória. FLOPs estimados pela fórmula das multiplicações de matriz, separadas pela origem:

```
projeções dos blocos:  L · 2T · (4D² + 2·D·H)     linear em T
atenção (QKᵀ e P·V):   L · 4 · T² · D              quadrático em T
cabeça de saída:       2 · T · D · V               linear em T, vezes o vocabulário
```

Tempo: mediana de 11 forwards após 3 de aquecimento, em CPU, sob `no_grad()`. A máquina é CPU-only e compartilhada: **valem as razões entre linhas, não os valores absolutos**, e diferenças de 1–2 ms são ruído. Os FLOPs são a medida de custo reprodutível; o tempo confirma a tendência.

**Reprodutibilidade.** Seed 42 antes de cada modelo; sorteios com `torch.Generator` semeado. Rodar os scripts de novo reproduz todas as colunas, exceto as de tempo.

---

## 2. Validação: reprodução do livro

**Configuração.** Os exemplos do capítulo 4 e da seção 5.3, refeitos com `src/` e as seeds do livro.

**Resultado.** **66 de 66 comparações conferem** ([`livro.md`](../../experimentos/sprint04/resultados/livro.md)): logits do `DummyGPTModel` e do `GPTModel` de 124M (96 valores), normalização da seção 4.2, gradientes da seção 4.4 (até a sexta casa significativa), contagens de parâmetros e memória, exercícios 4.1 a 4.3, a geração `Hello, I am Featureiman Byeswickattribute argue`, e as frequências de amostragem, probabilidades por temperatura e top-k do vocabulário de 9 palavras, com os exercícios 5.1 e 5.3.

**Análise.** A reprodução vai além do "shape certo": com a mesma seed, o projeto sorteia os mesmos pesos e faz as mesmas operações na mesma ordem — até os sorteios do dropout batem, já que os logits da seção 4.6 foram calculados em modo de treino. Isso valida a implementação antes dos experimentos: as variações medidas abaixo vêm do hiperparâmetro alterado, não de um defeito de implementação.

---

## 3. O modelo sem treino (A2)

**Configuração.** `BASE_CONFIG` (seed 42) e GPT-2 small (seed 123), em `eval()`, sobre o lote do The Verdict.

**Resultados.**

| | Base | GPT-2 small | Uniforme |
| - | ---- | ----------- | -------- |
| Desvio dos logits | 0,5774 | 0,5772 | — |
| Probabilidade dada ao token certo | 2,05e-05 | 2,02e-05 | 1/V = 1,99e-05 |
| Entropia da previsão (nats) | 10,658 | 10,658 | ln V = 10,825 |
| Cross-entropy (nats) | 10,976 | 10,975 | 10,825 |

**Análise.** O modelo sem treino é um chute quase uniforme sobre o vocabulário, nos dois tamanhos. O desvio dos logits, 0,577 = 1/√3, é consequência exata de dois detalhes da arquitetura: a LayerNorm final entrega vetores de variância 1, e a cabeça de saída é inicializada com variância 1/(3D) por peso, o que dá variância 1/3 por logit para qualquer D. Para logits com variância σ², a entropia do softmax é ≈ ln V − σ²/2 = 10,658, o valor medido. A perda inicial de ~11 nats é o ponto de partida da Sprint 5; uma perda inicial muito acima disso indicaria logits mal escalados, e muito abaixo, vazamento do alvo. O teste de integração fixa a perda a menos de 1 nat de ln V.

---

## 4. Dimensões da arquitetura

### 4.1 Número de blocos (A3)

**Configuração.** `n_layers` ∈ {1, 2, 4, 8, 12}.

**Resultados.**

| n_layers | Parâmetros | Nos blocos | GFLOP (T = 128) | Tempo (ms) | Desvio do fluxo residual | Desvio dos logits |
| -------- | ---------- | ---------- | --------------- | ---------- | ------------------------ | ----------------- |
| 1 | 26 553 856 | 3,0% | 3,51 | 11,9 | 1,425 | 0,577 |
| 4 | 28 920 832 | 10,9% | 4,17 | 14,8 | 1,475 | 0,577 |
| 12 | 35 232 768 | 26,9% | 5,91 | 24,1 | 1,610 | 0,577 |

**Análise — custo.** Cada bloco acrescenta exatamente 788 992 parâmetros e 0,22 GFLOP por sequência de 128 tokens: o custo da profundidade é linear e previsível. Mas parte de uma base alta: com um bloco só, o modelo já tem 26,5 milhões de parâmetros e 3,51 GFLOP, quase tudo da tabela de tokens e da cabeça de saída. Triplicar a profundidade (4 → 12 blocos) aumenta os parâmetros em 22%, os FLOPs em 42% e o tempo em 1,6x. O tempo cresce mais que os FLOPs porque cada bloco tem várias operações pequenas (LayerNorms, softmax, GELU) que não entram na conta de FLOPs e pesam em CPU.

**Análise — comportamento.** O fluxo residual cresce devagar com a profundidade (desvio de 1,425 para 1,610), porque cada bloco **soma** sua correção a ele. Os logits, ao contrário, ficam em 0,577 em todas as profundidades: é a LayerNorm final que desacopla a escala da saída do número de blocos. Sem ela (A9, variante sem LayerNorm), o desvio dos logits sobe para 1,241 com 12 blocos.

### 4.2 Número de cabeças (A4)

**Configuração.** `n_heads` ∈ {1, 2, 4, 8, 16}, com `emb_dim` 256 fixo (`head_dim` de 256 a 16).

**Resultados.** Parâmetros constantes em 28 920 832 nas cinco linhas. Células de scores (todos os blocos, L·H·T²) de 65 536 (0,25 MB) a 1 048 576 (4,00 MB). Tempo entre 13,4 e 16,6 ms, sem crescer com H. Perda inicial entre 10,975 e 10,976.

**Análise.** O resultado da Sprint 3 vale para o modelo inteiro: mais cabeças com o mesmo `emb_dim` dão cabeças **menores**, não modelo maior — os parâmetros não mudam. O que muda é a memória de ativação, que dobra a cada dobra de H, porque cada cabeça tem sua própria matriz T × T. Mas, com T = 128, essa matriz é pequena perto do resto: os scores de todos os blocos com 16 cabeças (4 MB por sequência) ainda são uma fração dos logits de uma sequência (24,5 MB, ver A7). Por isso o tempo não se move. O número de cabeças é uma escolha de **capacidade de representação** (quantos padrões de atenção independentes por bloco), quase gratuita em custo neste regime; o efeito dela sobre a qualidade só aparece depois do treino.

### 4.3 Dimensão do embedding (A5)

**Configuração.** `emb_dim` ∈ {64, 128, 256, 512, 768}, com 4 cabeças (`head_dim` = `emb_dim`/4) e FeedForward 4·`emb_dim`. Referência: GPT-2 small.

**Resultados.**

| emb_dim | Parâmetros | Tabelas + cabeça | Blocos (atenção + FFN) | GFLOP | Tempo (ms) |
| ------- | ---------- | ---------------- | ---------------------- | ----- | ---------- |
| 64 | 6 640 384 | 97,0% | 3,0% | 0,89 | 5,6 |
| 256 | 28 920 832 | 89,1% | 10,9% | 4,17 | 16,1 |
| 768 | 105 636 864 | 73,2% | 26,8% | 17,33 | 56,6 |
| GPT-2 small (768, 12 blocos) | 163 009 536 | 47,8% | 52,2% | 32,23 | — |

![Fração dos parâmetros por componente](../../experimentos/sprint04/resultados/figuras/arquitetura-parametros-por-componente.png)

**Análise — custo.** `emb_dim` é o hiperparâmetro mais caro: de 64 para 768 (12x), os parâmetros crescem 15,9x, os FLOPs 19,5x e o tempo 10,1x. São dois termos com crescimentos diferentes: as tabelas e a cabeça crescem **linear** em D (V·D cada), os blocos crescem com **D²** (12·D² por bloco). Com o vocabulário de 50 257 tokens, o termo linear domina em todas as linhas do projeto — os blocos só passam a ser a maioria dos parâmetros no GPT-2 small, que é largo **e** profundo. Para quatro blocos, a igualdade 2·V·D = 4·12·D² só aconteceria em D ≈ 2 100.

**Análise — comportamento.** A perda inicial fica entre 10,94 e 11,01 em todas as larguras: pela razão do A2, a escala dos logits na inicialização não depende de D. A largura muda o que o modelo **pode** aprender (o tamanho do espaço em que cada token é representado), não o ponto de partida.

**Consequência para o projeto.** No modelo base, 89% dos parâmetros estão em tabelas de consulta e na projeção para o vocabulário, e só 11% nos blocos, onde está o "raciocínio" do modelo. É a conclusão da Sprint 2 (o tamanho do vocabulário é uma decisão de arquitetura), agora medida no modelo completo. O weight tying (Entregável 3, seção 5) reduziria o modelo base de 28,9 para 16,1 milhões de parâmetros sem tocar nos blocos.

### 4.4 Dimensão da FeedForward (A6)

**Configuração.** `ff_hidden_dim` = fator · `emb_dim`, fator ∈ {1, 2, 4, 8}.

**Resultados.**

| Fator | FFN por bloco | Fração do bloco | Parâmetros do modelo | GFLOP dos blocos | GFLOP total | Tempo (ms) |
| ----- | ------------- | --------------- | -------------------- | ---------------- | ----------- | ---------- |
| 1 | 131 584 | 33,3% | 27 344 896 | 0,40 | 3,76 | 12,1 |
| 4 | 525 568 | 66,6% | 28 920 832 | 0,81 | 4,17 | 13,9 |
| 8 | 1 050 880 | 80,0% | 31 022 080 | 1,34 | 4,70 | 17,3 |

**Análise.** Com o fator 4 do GPT-2, a FeedForward é dois terços do bloco — o dobro da atenção, como no exercício 4.1. Dobrar o fator dobra os parâmetros e os FLOPs **da FeedForward**; nos blocos o efeito já é menor (as projeções da atenção não mudam: de fator 1 para 8, os FLOPs dos blocos crescem 3,4x, não 8x), e no modelo inteiro é diluído pela cabeça de saída: de fator 1 para 8, o modelo cresce 13%, os FLOPs totais 25% e o tempo 43%. A FeedForward é onde se compra capacidade não linear por token; no modelo base ela é barata em relação ao custo fixo da cabeça, e no GPT-2 small (onde é 34,8% do modelo) seria a maior conta.

---

## 5. Tamanho do contexto (A7)

**Configuração.** `context_length` ∈ {32, 64, 128, 256, 512, 1024}, forward de uma sequência com T = `context_length`.

**Resultados.**

| T | Parâmetros | Scores (L·H·T²) | Logits (T·V) | GFLOP blocos | GFLOP atenção | GFLOP cabeça | Atenção no total | Tempo (ms) |
| - | ---------- | --------------- | ------------ | ------------ | ------------- | ------------ | ---------------- | ---------- |
| 32 | 28 896 256 | 0,06 MB | 6,13 MB | 0,20 | 0,00 | 0,82 | 0,4% | 5,8 |
| 128 | 28 920 832 | 1,00 MB | 24,54 MB | 0,81 | 0,07 | 3,29 | 1,6% | 14,1 |
| 1024 | 29 150 208 | 64,00 MB | 196,32 MB | 6,44 | 4,29 | 26,35 | 11,6% | 104,1 |

**Análise — parâmetros.** O contexto quase não custa parâmetros: só a tabela de posições cresce (T·D), de 8 192 para 262 144 — 0,9% do modelo em T = 1 024. A máscara causal cresce com T², mas é buffer, não parâmetro.

**Análise — custo.** Multiplicar T por 32 multiplica os FLOPs por 36 e o tempo por 18 — em T pequeno, o custo fixo de cada operação (chamada, alocação) pesa mais que a conta, e a CPU fica subutilizada. Os termos lineares (blocos e cabeça) crescem 32x; o termo da atenção cresce 1 024x (de 0,004 para 4,29 GFLOP) e sobe de 0,4% para 11,6% do total. O termo quadrático está lá e cresce como a Sprint 3 mediu, mas **ainda não domina**: a cabeça de saída responde por 71% dos FLOPs em T = 1 024 (26,35 de 37,08 GFLOP). A atenção (4·L·T²·D) só igualaria a cabeça (2·T·D·V) em T = V/(2L) = 6 282 tokens. O mesmo vale para a memória: os logits de uma sequência (196 MB em T = 1 024) são 3x os scores de todos os blocos (64 MB); a igualdade T·V = L·H·T² só ocorreria em T = V/(L·H) ≈ 3 141.

**Análise — relação com a Sprint 3.** A Sprint 3 concluiu que `context_length` é o que estoura a memória no treino, pelo termo T². Isso continua verdadeiro **assintoticamente**, mas para um modelo pequeno com o vocabulário do GPT-2, no intervalo de T do projeto, o maior tensor é o de logits, linear em T e proporcional a V. Para a Sprint 5, isso significa que um lote de 8 × 128 já produz 196 MB só de logits (mais o mesmo tanto no gradiente deles), enquanto os scores de atenção do lote inteiro somam 8 MB.

---

## 6. Os componentes do bloco: ablação (A9)

**Configuração.** `BASE_CONFIG` com 12 blocos e dropout 0. Cinco variantes com **os mesmos pesos** (cópias do mesmo modelo, só o `forward` do bloco muda): o bloco do projeto (Pre-LN), sem atalho, sem LayerNorm (blocos e final), sem atalho e sem LayerNorm, e Post-LN (normalização depois da soma, como no Transformer original). Forward e backward da cross-entropy no lote do The Verdict.

**Resultados.**

| Variante | Desvio do fluxo após o bloco 12 | Desvio dos logits | Perda inicial | \|grad\| bloco 1 | \|grad\| bloco 12 | Bloco 1 / bloco 12 |
| -------- | ------------------------------- | ----------------- | ------------- | ---------------- | ----------------- | ------------------ |
| Pre-LN (projeto) | 1,610 | 0,577 | 11,000 | 1,98e-04 | 1,78e-04 | 1,11 |
| Sem atalho | 0,205 | 0,579 | 11,068 | 1,88e-03 | 1,96e-03 | 0,96 |
| Sem LayerNorm | 2,149 | 1,241 | 11,614 | 9,15e-04 | 9,45e-04 | 0,97 |
| Sem atalho e sem LayerNorm | 0,021 | 0,012 | 10,824 | 1,82e-18 | 2,34e-05 | 7,8e-14 |
| Post-LN | 1,000 | 0,577 | 10,999 | 3,18e-04 | 3,20e-04 | 0,99 |

![Escala do fluxo residual ao longo da pilha](../../experimentos/sprint04/resultados/figuras/arquitetura-ablacao-escala.png)

![Gradiente que chega a cada bloco](../../experimentos/sprint04/resultados/figuras/arquitetura-ablacao-gradiente.png)

**Análise — o atalho e a LayerNorm protegem o gradiente de formas diferentes, e um basta para o outro na inicialização.** O resultado mais forte está na quarta linha: sem os dois mecanismos, o modelo é a "rede profunda pura" da seção 4.4 do livro, e o gradiente que chega ao bloco 1 é 1,8e-18 — treze ordens de grandeza abaixo do bloco 12. O primeiro bloco não aprenderia nada. Ao mesmo tempo, o fluxo cai de 1,41 para 0,02 já no primeiro bloco, os logits colapsam (desvio 0,012) e a perda é exatamente ln V: a informação da entrada não atravessa a pilha.

Tirar só um dos dois **não** reproduz esse colapso, e o motivo é diferente em cada caso:

- **Sem atalho, com LayerNorm:** o fluxo também encolhe (0,205), mas cada bloco normaliza a própria entrada antes de usá-la, e a LayerNorm amplia o que recebe pequeno — o gradiente que volta por ela é reescalado na mesma proporção. Ele chega ao bloco 1 com o mesmo tamanho do bloco 12. O preço aparece no fluxo: depois do bloco 1, o embedding original não está mais lá (o desvio cai de 1,41 para 0,20 de uma vez), porque cada bloco **substitui** a representação em vez de corrigi-la.
- **Sem LayerNorm, com atalho:** o "1" da derivada de `x + f(x)` carrega o gradiente pela pilha (razão 0,97). O preço aparece na escala: o fluxo cresce bloco a bloco (1,41 → 2,15) e, sem a norma final, os logits saem com o dobro do desvio (1,241) e a perda inicial sobe para 11,61. Com mais blocos, a escala continuaria crescendo.

O bloco Pre-LN do GPT-2 tem os dois, e é a única variante que mantém **ao mesmo tempo** o embedding original no fluxo (o desvio só cresce de 1,41 para 1,61), a escala dos logits fixa (0,577) e o gradiente uniforme entre os blocos (razão 1,11).

**Pre-LN contra Post-LN.** Na inicialização, o Post-LN também é estável: normalizando depois de cada soma, o fluxo fica em exatamente 1,000 e o gradiente é uniforme. A diferença que o livro aponta ("dinâmica de treino pior") não é visível num único passo de backward — ela aparece ao longo do treino, quando a normalização em cima do atalho passa a interferir no caminho direto do gradiente. É um experimento para a Sprint 5. O que este experimento mostra é estrutural: no Post-LN, o fluxo é renormalizado a cada bloco, então o embedding de entrada não passa adiante intacto como no Pre-LN.

**Limitação.** Todas as medidas são na inicialização, com um passo de backward. Elas mostram se o gradiente **chega** a cada bloco, não como o treino evolui.

---

## 7. Dropout (A8)

**Configuração.** `drop_rate` ∈ {0, 0,1, 0,2, 0,5}, mesmos pesos em todas as linhas. Em `train()`: duas passadas sobre o mesmo lote (variação = desvio da diferença entre elas / desvio dos logits) e a perda média de 5 passadas. Depois, cada uma das três taxas do exercício 4.3 sozinha em 0,1.

**Resultados.**

| Configuração | Variação entre 2 passadas em train() | Perda em train() | Perda em eval() | eval() determinístico |
| ------------ | ------------------------------------ | ---------------- | --------------- | --------------------- |
| p = 0 | 0,000 | 10,9758 | 10,9758 | sim |
| p = 0,1 (livro) | 0,468 | 10,9787 | 10,9758 | sim |
| p = 0,5 | 1,027 | 10,9924 | 10,9758 | sim |
| só embeddings, 0,1 | 0,448 | 10,9798 | 10,9758 | sim |
| só pesos de atenção, 0,1 | 0,045 | 10,9759 | 10,9758 | sim |
| só antes dos atalhos, 0,1 | 0,142 | 10,9766 | 10,9758 | sim |

**Análise.** O dropout não custa nada em parâmetros nem em `eval()` — a perda em `eval()` é idêntica em todas as linhas e as duas passadas coincidem bit a bit. Em `train()`, ele torna cada passada uma amostra diferente da rede: com a taxa do livro, a diferença entre duas passadas já tem 47% do desvio dos próprios logits, e com p = 0,5 ela passa do tamanho dos logits. É a regularização: o modelo não pode depender de nenhuma unidade específica.

O exercício 4.3 separado mostra que os três lugares **não** pesam igual. O dropout dos embeddings sozinho responde por quase toda a variação (0,448 de 0,468), o dos pesos de atenção quase não aparece (0,045). A explicação está no fluxo residual: o dropout dos embeddings apaga 10% das features **da entrada**, e o atalho carrega esse buraco até a saída, sem que nenhum bloco o desfaça. O dropout de atenção apaga pesos de uma média sobre poucos tokens que já são parecidos entre si (o modelo sem treino tem atenção quase uniforme, Sprint 3), e a perturbação se dilui. É um argumento para ter as três taxas separadas: elas controlam regularizações de intensidades diferentes.

---

## 8. Parâmetros de geração

### 8.1 Temperatura (G3)

**Configuração.** Logits da última posição de `Every effort moves you`. Temperaturas de 0,25 a 2; 1 000 sorteios por temperatura com gerador semeado.

**Resultados.**

| Temperatura | Maior probabilidade | Entropia (nats) | Candidatos efetivos (e^H) | Tokens distintos em 1 000 | Sorteios = argmax |
| ----------- | ------------------- | --------------- | ------------------------- | ------------------------- | ----------------- |
| 0 (gulosa) | 1 | 0 | 1 | 1 | 100% |
| 0,25 | 0,0226 | 8,228 | 3 746 | 731 | 1,9% |
| 0,5 | 0,0013 | 10,165 | 25 982 | 970 | 0,2% |
| 1 | 0,0002 | 10,659 | 42 569 | 988 | 0,0% |
| 2 | 0,0001 | 10,783 | 48 207 | 988 | 0,0% |

![Temperatura e diversidade](../../experimentos/sprint04/resultados/figuras/geracao-temperatura.png)

**Análise.** A temperatura multiplica o desvio dos logits por 1/T, e a forma da distribuição acompanha: com T = 0,25 o desvio efetivo passa de 0,577 para 2,3, o token mais provável vai de 0,02% para 2,3% e o número efetivo de candidatos cai de 42 569 para 3 746. Acima de T = 1, a distribuição já estava perto da uniforme e quase não muda (de 42 569 para 48 207 candidatos, num vocabulário de 50 257). Em 1 000 sorteios, quase todos os tokens são distintos de T = 0,5 para cima — o limite ali é o número de sorteios, não a distribuição.

No modelo sem treino, mesmo T = 0,25 está longe de tornar a escolha previsível, porque os logits são quase planos. Num modelo treinado, os logits do token certo se separam dos outros por vários desvios, e a mesma temperatura teria um efeito muito mais forte — é o que o vocabulário de 9 palavras do livro mostra (L8): ali `forward` vai de 0,57 com T = 1 a 0,99 com T = 0,1. A temperatura não cria nem remove candidatos: só redistribui a probabilidade entre os que o modelo já ordenou.

### 8.2 Top-k (G4)

**Configuração.** Os mesmos logits, T = 1, k ∈ {1, 5, 20, 100, 1 000, sem top-k}, e uma grade de temperatura × top-k.

**Resultados.**

| top_k | Massa original mantida | Entropia (nats) | Tokens distintos em 1 000 |
| ----- | ---------------------- | --------------- | ------------------------- |
| 1 | 0,02% | 0 | 1 |
| 20 | 0,27% | 2,984 | 20 |
| 1 000 | 6,91% | 6,885 | 612 |
| sem top-k | 100% | 10,659 | 988 |

Na grade, com top-k 5 ou 50, os tokens distintos são exatamente 5 e 50 para qualquer temperatura entre 0,5 e 1,5.

**Análise.** O top-k impõe um teto **rígido** à diversidade: nunca saem mais de k tokens distintos, qualquer que seja a temperatura. A temperatura é um controle **suave** da forma da distribuição dentro desse teto. No modelo sem treino, os 20 melhores candidatos somam só 0,27% da probabilidade original — o top-k descarta 99,7% da massa e renormaliza o resto, o que mostra que ele não "corta a cauda improvável": corta tudo fora do ranking. Isso só é razoável quando o modelo concentra a probabilidade nos primeiros lugares, o que acontece depois do treino. Custo: o top-k é um `topk` e um `scatter` sobre V valores por passo, desprezível perto da passada do modelo.

### 8.3 Estratégias lado a lado e determinismo (G5)

**Configuração.** 30 tokens a partir do prompt com cinco estratégias; repetição de cada uma com a mesma seed e com outra; e a gulosa rodada duas vezes num laço manual com o modelo em `train()`.

**Resultados.**

| Estratégia | Tokens distintos | Log-prob média | Igual com a mesma seed | Igual com outra seed |
| ---------- | ---------------- | -------------- | ---------------------- | -------------------- |
| Gulosa | 30/30 | −8,625 | sim | sim |
| top-k 5, T = 1 | 30/30 | −8,706 | sim | não |
| top-k 50, T = 0,7 | 30/30 | −9,007 | sim | não |
| T = 1, sem top-k | 30/30 | −10,843 | sim | não |
| T = 1,5, sem top-k | 30/30 | −10,924 | sim | não |

Gulosa em `train()`: 28 das 30 posições diferentes entre duas execuções, a partir do segundo token. Pela `generate`: nenhuma.

**Análise.**

- **Log-prob média** ordena as estratégias pelo quanto elas se afastam do que o modelo considera provável: a gulosa escolhe sempre o máximo (−8,6, perto de ln 1,9e-04), o top-k fica próximo dela, e a amostragem sem top-k cai para o nível de um sorteio uniforme (ln(1/V) = −10,825). Num modelo treinado, essa é a troca entre texto previsível e texto diverso que o exercício 5.2 do livro discute.
- **Repetição.** O modelo sem treino não entra em laço nem na gulosa (30 tokens distintos em todas as estratégias). A repetição característica da decodificação gulosa é um fenômeno de modelos treinados, e este experimento não pode medi-la ainda.
- **Mesmo gerador, distribuições parecidas.** Os textos de T = 1 e T = 1,5 coincidem em boa parte. Não é coincidência: o `multinomial` usa os mesmos números aleatórios nas duas (mesma seed), e com distribuições quase uniformes o mesmo número cai no mesmo token. Comparar estratégias com a mesma seed isola o efeito da estratégia, mas exige cuidado ao ler semelhanças.
- **`eval()` é parte da inferência, não detalhe.** Com o dropout ligado, até a gulosa vira aleatória (28 de 30 posições mudam). É por isso que `generate` força `eval()` e restaura o modo no fim.

---

## 9. Custo da geração

### 9.1 Custo por token e janela de contexto (G6)

**Configuração.** 200 tokens gulosos com `BASE_CONFIG` (`context_length` 128); cada passo cronometrado 3 vezes.

**Resultados.** Mediana por passo: 5,2 ms nos passos 1–25 (4 a 28 tokens no contexto), 14,1 ms nos passos 76–100 (79 a 103), 17,7 ms nos passos 101–125 (104 a 128), e entre 14,4 e 15,7 ms depois que o contexto enche, no passo 125. Total: 17 850 posições processadas para 200 tokens novos.

![Custo por passo de geração](../../experimentos/sprint04/resultados/figuras/geracao-custo-por-passo.png)

**Análise.** O custo de cada token é o de uma passada sobre o contexto inteiro, então ele cresce com o tamanho do contexto até o `context_length` e para de crescer quando a janela começa a deslizar. O custo total de gerar n tokens é, portanto, quadrático em n enquanto o contexto não enche (cada passo reprocessa todos os anteriores: 89 posições por token novo, em média) e linear depois. Os logits das posições antigas são recalculados a cada passo e saem iguais (G2: 1,8e-06) — é a redundância que um KV cache eliminaria, guardando chaves e valores dos tokens já vistos e processando só o token novo.

### 9.2 Tamanho do modelo (G7)

**Resultados.** 20 tokens gulosos: `BASE_CONFIG` 4,3 ms por token (233 tokens/s); com 12 blocos, 9,4 ms (107 tokens/s); GPT-2 small, 25,4 ms (39 tokens/s).

**Análise.** Com contexto curto (4 a 24 tokens), o custo por token acompanha o número de parâmetros que participam de cada passada: o GPT-2 small tem 5,6x os parâmetros do modelo base e é 5,9x mais lento. A diferença entre 4 e 12 blocos (2,2x no tempo para 1,2x nos parâmetros) mostra de novo que o tempo em CPU depende do número de operações, não só dos FLOPs: oito blocos a mais são muitas operações pequenas, enquanto a cabeça de saída, que concentra os parâmetros, é uma multiplicação de matriz só.

---

## 10. Síntese

| Hiperparâmetro (aumentando) | Parâmetros | Memória de ativação | FLOPs por token | Comportamento na inicialização |
| --------------------------- | ---------- | ------------------- | --------------- | ------------------------------ |
| `n_layers` | +788 992 por bloco (linear) | Linear | Linear, sobre uma base alta (cabeça) | Fluxo residual cresce devagar; logits fixos pela LayerNorm final |
| `n_heads` (D fixo) | Não muda | Linear em H (scores) | Não muda | Nenhum efeito mensurável |
| `emb_dim` | Linear (tabelas, cabeça) + quadrático (blocos) | Linear | Linear + quadrático | Perda inicial e escala dos logits não mudam |
| `ff_hidden_dim` | Linear, só nos blocos | Linear | Linear, só nos blocos | Não muda |
| `context_length` | Só a tabela de posições | Linear (logits) + quadrático (scores) | Linear + termo quadrático que cresce (0,4% → 11,6%) | Não muda |
| `drop_rate` | Não muda | Máscaras em `train()` | Desprezível | Saída aleatória em `train()`; nenhum efeito em `eval()` |
| Temperatura | — | — | — | Controla a forma da distribuição; não muda a ordem |
| Top-k | — | — | Desprezível | Teto rígido de k candidatos |

Quatro conclusões atravessam os experimentos:

1. **No modelo base, o custo está nas pontas, não no meio.** Tabela de tokens e cabeça de saída têm 89% dos parâmetros e a cabeça faz 79% dos FLOPs de um forward (3,29 de 4,17 GFLOP). O tamanho do vocabulário, decidido na Sprint 2, é o que mais pesa no modelo da Sprint 4.
2. **O atalho e a LayerNorm são o que torna a profundidade possível.** Sem os dois, o gradiente do primeiro de 12 blocos é 13 ordens de grandeza menor que o do último; com o bloco Pre-LN do GPT-2, os 12 recebem o mesmo gradiente e os logits têm a mesma escala em qualquer profundidade.
3. **O termo quadrático da atenção existe, mas ainda não manda.** No intervalo de contexto do projeto, a memória e os FLOPs são dominados pelo termo T·V da cabeça; a atenção só igualaria em milhares de tokens.
4. **Geração é inferência com regras próprias.** Só a última linha dos logits é usada; `eval()` é obrigatório para a decodificação ser determinística; a temperatura molda a distribuição e o top-k limita os candidatos; e o laço reprocessa o contexto inteiro a cada token, o que torna o custo quadrático até a janela encher.
