# Sprint 3 — Atenção: análise

**Dos embeddings aos vetores de contexto.** Responde às questões da Sprint 3 a partir dos números medidos em [`experimentos/sprint03/resultados/atencao.md`](../../experimentos/sprint03/resultados/atencao.md), experimentos E1 a E8.

Frases de teste: `The bank of the river` e `The bank of the city`, 5 tokens cada pelo BPE do GPT-2, iguais nas quatro primeiras posições. Configuração base: a da Sprint 2 (`emb_dim` 256, `context_length` 128, seed 42) mais 4 cabeças, o que dá `head_dim` 64, o mesmo do GPT-2.

---

## O que foi implementado

| Etapa                         | Onde                                  | O que faz                                                                 |
| ----------------------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| Função núcleo                 | `src/attention/scaled_dot_product.py` | `scaled_dot_product_attention`: scores → escala → máscara → softmax → dropout → contexto |
| Atenção simplificada          | `src/attention/simplified.py`         | `simplified_attention`: Q, K e V são a própria entrada, sem pesos, com `scale=False` |
| Self-attention treinável      | `src/attention/self_attention.py`     | `SelfAttention`: projeções `W_query`, `W_key`, `W_value` como `nn.Linear` sem bias |
| Atenção causal                | `src/attention/causal.py`             | `causal_mask` e `CausalAttention`: máscara triangular em `register_buffer`, dropout nos pesos |
| Multi-head                    | `src/attention/multi_head.py`         | `MultiHeadAttentionWrapper` (lista de cabeças) e `MultiHeadAttention` (weight split com `view` + `transpose` e `out_proj`) |

Os testes em `tests/test_self_attention.py` (12), `tests/test_causal_attention.py` (8), `tests/test_multi_head_attention.py` (9) e `tests/test_attention_pipeline.py` (4) — 33 no total, todos passando — fixam cada afirmação feita abaixo.

### Decisões que divergem do livro

**A scaled dot-product é uma função, não um trecho repetido.** No livro, cada classe (`SelfAttention_v2`, `CausalAttention`, `MultiHeadAttention`) reescreve dentro do próprio `forward` a mesma sequência: produto `queries @ keys.T`, divisão por √d_k, máscara, softmax, dropout e multiplicação pelos values. Aqui essa sequência existe uma vez só, em `scaled_dot_product_attention`, e todas as classes a chamam. O que muda de uma classe para outra passa a ser só o que ela entrega à função: a simplificada passa `scale=False`, a `SelfAttention` não passa máscara, a `CausalAttention` passa a máscara e o dropout, a `MultiHeadAttention` passa tensores com uma dimensão a mais. A função opera nas duas últimas dimensões, então a mesma conta serve a `(B, T, d)` e a `(B, H, T, head_dim)` sem ramificar.

A consequência prática aparece nos experimentos: como a conta não está duplicada, as comparações são justas por construção. No E5, "antes da máscara" e "depois da máscara" são a mesma instância alimentando a mesma função duas vezes, uma com `mask=None`; no E8, Wrapper e weight split dão o mesmo resultado a 1,19e-07 porque não há duas implementações da atenção para divergir, só duas formas de montar os tensores.

**Retorno `(context, weights)`.** A função e todos os `forward` devolvem também os pesos, que alimentam os heatmaps sem recálculo. O custo é o capítulo 4 escrever `x, _ = att(x)`.

**Máscara bool, `True` = bloqueado, `-inf` antes do softmax.** O livro mostra primeiro o caminho de zerar os pesos depois do softmax e renormalizar; o pacote usa só o outro (ver questão 7).

**O dropout vem da classe que chama.** Uma função solta não tem `train()`/`eval()`; o `nn.Dropout` pertence à camada e é passado como argumento.

---

# Self-Attention e escala

## Questão 1 — Que problema a atenção resolve em relação ao RNN encoder-decoder?

O gargalo do hidden state. No RNN encoder-decoder, o encoder lê a frase inteira e a comprime num único vetor de tamanho fixo, e é só esse vetor que o decoder recebe. Tudo o que o modelo sabe da entrada precisa caber ali, e a informação do começo da frase vai sendo sobrescrita pelas atualizações seguintes.

A atenção troca "um resumo da sequência" por "acesso direto a todas as posições". Cada token calcula um peso para cada outro token e monta seu próprio vetor de contexto como média ponderada. Não existe mais um ponto por onde toda a informação precisa passar, e a distância entre duas palavras deixa de importar: a posição 0 e a posição 127 estão a um produto escalar de distância, não a 127 passos de recorrência. O E4 mostra isso direto: trocar só o último token (`' river'` por `' city'`) altera o vetor de contexto de `'The'`, na posição 0, em 1,27 de distância L2 — informação que atravessou a frase inteira numa camada só.

Esse acesso tem preço, e o E2b mede qual: uma matriz de pesos T×T por sequência. Com T = 1024 são 1 048 576 células, 4 MB em float32, para uma única cabeça e uma única frase. O RNN tinha custo linear em T e perdia informação; a atenção não perde, mas paga quadrático. Essa troca é o que limita o `context_length` de toda LLM (questão 5).

---

## Questão 2 — Qual a diferença entre embedding de entrada e context vector? (E4)

O embedding de entrada representa o token **isolado**: depende só do Token ID e da posição, lidos das duas tabelas da Sprint 2. O vetor de contexto representa o token **dentro desta frase**: é uma média dos values de todos os tokens, pesada pela relevância de cada um.

O E4 separa as duas coisas com um par de frases que diferem só no final. Nas posições 0 a 3, a distância entre os embeddings de entrada das duas frases é 0,000000 — exatamente zero, porque o mesmo Token ID na mesma posição lê as mesmas linhas das tabelas. A distância entre os vetores de contexto, nas mesmas posições, vai de 1,27 (`'The'`) a 1,91 (`' bank'`). Como a única diferença entre as frases está na posição 4, toda essa distância entrou pela camada de atenção vinda de `' river'`/`' city'`.

É a resposta à ambiguidade que a Sprint 2 deixou aberta: o embedding de `bank` é o mesmo em "margem do rio" e em "banco da cidade". O vetor de contexto não é. Os pesos ainda são aleatórios, então a diferença medida não é "sentido" — mas mostra que o mecanismo tem por onde levar o sentido do resto da frase para dentro de cada token.

Nesta camada não há máscara, e por isso a mudança chega também nas posições anteriores ao token trocado. A questão 6 mostra o que acontece quando ela entra.

---

## Questão 3 — Por que as matrizes de Q, K e V precisam ser treináveis? (E3)

Porque sem elas o critério de atenção está congelado no produto escalar cru dos embeddings, e esse critério é quase sempre "olhe para si mesmo".

O E3 passa os mesmos vetores de entrada pelas duas versões. Na simplificada, sem pesos, os 5 tokens de `The bank of the river` põem peso 1,000 em si mesmos, com entropia 0,000: nenhuma atenção a mais nada. Com `W_q`, `W_k` e `W_v`, só 1 dos 5 tokens olha mais para si, e as entropias sobem para 1,05–1,54 nats — a atenção se espalha pela frase. A diagonal da versão simplificada nem é afinidade: com 256 dimensões e sem escala, o produto de um vetor por ele mesmo fica na casa das centenas e os outros perto de zero, e o softmax satura (é a linha `d_k = 256, escala = não` do E1).

Três matrizes, e não uma, por dois motivos que a implementação deixa explícitos:

- **Q e K separadas quebram a simetria.** Com `W_q = W_k`, o score de A para B seria igual ao de B para A. Mas "o adjetivo procura o substantivo" não é a mesma relação que "o substantivo procura o adjetivo".
- **V responde a outra pergunta.** Q e K decidem **quanto** um token pesa; V decide **o que** ele contribui. Separar as duas coisas deixa o modelo usar uma parte do embedding para ser encontrado e outra para ser copiado.

E precisam ser treináveis porque o critério certo não é conhecido de antemão: é justamente o que o treino vai descobrir. Os pesos treináveis são fixos depois do treino; os attention weights que eles produzem mudam a cada frase.

---

## Questão 4 — Por que dividir por √d_k? (E1)

Porque o produto escalar de dois vetores aleatórios de média zero e variância 1 tem desvio padrão √d_k. Sem a divisão, os scores crescem com a dimensão; com ela, ficam em torno de 1 em qualquer dimensão. O E1 mede exatamente isso: com escala, o desvio dos scores fica entre 0,97 e 1,02 de d_k = 16 a d_k = 4096; sem escala, vai de 4,09 a 64,4.

Scores grandes empurram o softmax para uma função degrau. Sem escala, o peso máximo médio sobe de 0,707 (d_k = 16) para 0,981 (d_k = 4096), e a entropia cai de 0,91 para 0,06 nats, de um máximo possível de 3,47: a atenção vira "olhe um token só", decidida pelo acaso da inicialização e não pelo treino.

O dano está no gradiente. A **sensibilidade** Σp(1−p) — o fator pelo qual o softmax multiplica o gradiente que passa por ele — fica parada em ~0,93 com escala e cai de 0,41 para 0,033 sem ela: uma ordem de grandeza a menos, que é a saturação em número.

O E1 registra um detalhe que desmonta a leitura ingênua: a norma do gradiente nos scores **não** desaba junto (fica entre 1,5e-02 e 3,9e-02 sem escala). Quando a atenção se concentra num token, o contexto vira uma linha inteira de V, a perda cresce e o gradiente que chega ao softmax cresce com ela; os dois efeitos se compensam numa camada só. Olhar só a norma levaria a concluir que a escala não muda nada. A conclusão certa é que ela muda o regime: o fator de 0,033 aparece em cada camada, e numa rede de 12 blocos ele se multiplica.

---

## Questão 5 — Como o custo cresce com d e com T? (E2)

Os dois custam coisas diferentes, e em lugares diferentes.

**A dimensão custa peso.** No E2a, com T fixo, os parâmetros são 3·d²: 768 para d = 16, 1 769 472 para d = 768. Crescimento quadrático, mas em **parâmetros** — memória permanente, paga uma vez, independente do tamanho da frase.

**O comprimento custa ativação.** No E2b, com d = 256 fixo, os parâmetros ficam em 196 608 em todas as linhas: uma sequência mais longa não cria peso nenhum. O que ela cria é a matriz de scores T×T, que vai de 64 células (T = 8) a 1 048 576 (T = 1024), 4 MB em float32. Também quadrático, mas em memória **temporária**, alocada a cada forward, multiplicada pelo tamanho do lote e pelo número de cabeças, e guardada para o backward durante o treino.

O tempo acompanha: dobrar T de 512 para 1024 leva de 1,76 ms para 7,42 ms, mais de 4× — o termo quadrático já domina. A máquina é CPU-only e os tempos absolutos valem pouco, mas a razão entre linhas é o que interessa.

Para o projeto, isso diz onde está o limite. Aumentar `emb_dim` deixa o modelo maior em disco; aumentar `context_length` é o que estoura memória no treino. É o mesmo argumento da questão 1 da Sprint 2 (sequências curtas porque a atenção é quadrática em T), agora medido.

---

# Causal e Multi-Head

## Questão 6 — Por que um GPT precisa de atenção causal? (E5)

Porque o GPT é treinado para prever o próximo token, e o alvo da posição t está na posição t+1 da própria entrada. Sem máscara, a posição t pode olhar para t+1 e copiar a resposta: a perda cai e o modelo não aprende a prever nada.

O E5 mede o vazamento com o par de frases do E4. Pela `SelfAttention`, trocar `' river'` por `' city'` move os vetores de contexto das posições 0 a 3 entre 1,27 e 1,91. Pela `CausalAttention`, a mesma troca move essas posições em 0,000000. Com a máscara, nada que veio antes do token trocado muda em um bit — a posição 3 não tem como saber que a próxima palavra é `river`.

É isso que permite treinar todas as posições de uma vez. Uma sequência de 128 tokens dá 128 problemas de "prever o próximo", resolvidos num único forward, e nenhum deles lê a própria resposta. A máscara tira o vazamento sem tirar o paralelismo.

A tabela do E5 também avisa sobre uma leitura errada: com máscara, a entropia da posição 0 cai para 0,000 e o peso máximo vai a 1,000. Não é a atenção ficando mais decidida — o primeiro token vê um candidato só, e não tem escolha. A cada posição o número de tokens visíveis cresce (1, 2, 3, 4, 5), e a entropia sobe com ele (0,00 → 0,67 → 1,01 → 1,37 → 1,38). Na última posição, que já via tudo, a máscara não muda nada: 0,460 e 1,381 antes e depois.

A máscara fica em `register_buffer`, e não como parâmetro: um triângulo de booleanos não tem nada a aprender, mas precisa acompanhar o modelo no `.to(device)` e no `state_dict`.

---

## Questão 7 — Por que usar -inf antes do softmax em vez de zerar depois? (E5)

Os dois caminhos dão o mesmo resultado. O E5 compara os dois sobre os mesmos scores: diferença máxima de 5,96e-08, abaixo do epsilon do float32 (1,19e-07). Não é zero exato porque são duas sequências diferentes de operações em ponto flutuante, mas é a mesma conta.

A diferença está no número de passos. Zerar depois exige três: softmax sobre todas as posições, multiplicar pela máscara triangular, renormalizar cada linha para voltar a somar 1. Com -inf antes, é um só: como e^(−∞) = 0, o softmax já produz zero nas posições futuras e já distribui 1,0 só entre as permitidas. Não há renormalização manual para esquecer ou errar.

Há um argumento de fundo também: no caminho de zerar depois, o softmax chega a calcular os pesos das posições futuras, e o denominador de cada linha os inclui antes da correção. Com -inf, o futuro nunca entra na conta — o que é a definição de causalidade, e não uma aproximação dela.

---

## Questão 8 — Qual o papel do dropout na matriz de atenção? (E5)

Regularização: durante o treino, zera ao acaso parte dos pesos de atenção, para o modelo não depender de uma conexão específica entre dois tokens. No pacote, ele age depois do softmax e depois da máscara, e antes da multiplicação pelos values.

O E5 mostra as duas coisas que ele faz. A primeira é apagar: com p = 0,1, 6,7% das posições permitidas vão a zero; com p = 0,5, 40%. (A fração total zerada é maior — 44% e 64% — porque a máscara já zerava 10 das 25 posições; só a fração sobre as permitidas se compara com p.) A segunda é aumentar quem sobrou: o fator medido é 1,1111 para p = 0,1 e 2,0000 para p = 0,5, exatamente 1/(1−p). É isso que mantém o **valor esperado** da soma de cada linha em 1.

Esperado, não realizado. A soma média das linhas numa amostra em `train()` deu 1,025 para p = 0,1 e 1,314 para p = 0,5. Durante o treino, a matriz de atenção deixa de ser uma distribuição de probabilidade linha a linha. Em `eval()` o dropout desliga e as linhas voltam a somar 1 exatamente — e por isso todos os pesos citados nesta análise foram medidos em `eval()`.

A configuração base usa p = 0,1; o teste de integração usa 0,0, porque ali se conferem valores.

---

## Questão 9 — O que várias heads acrescentam em relação a uma? (E8)

A possibilidade de olhar para mais de uma coisa ao mesmo tempo. Uma cabeça produz uma distribuição de pesos por token: se ela concentra a atenção no substantivo, não tem como concentrar também no verbo. Com 4 cabeças, cada token tem 4 distribuições independentes, e o contexto final junta as 4.

O E8 mostra o ponto de partida disso. Com a mesma `MultiHeadAttention` e a mesma frase, `' river'` olha mais para `' bank'` na cabeça 0 (0,321), para `'The'` na cabeça 1 (0,399), de novo para `' bank'` na cabeça 2 (0,283) e para si mesmo na cabeça 3 (0,320). Os quatro heatmaps são triangulares — a máscara vale em toda cabeça — e diferentes entre si.

Isso ainda não é especialização: os pesos acabaram de ser sorteados, e nenhuma cabeça "aprendeu" sintaxe ou semântica. O que o E8 mostra é que as cabeças **começam** diferentes. É essa quebra de simetria que permite o treino levar cada uma para um papel próprio; se começassem iguais, receberiam o mesmo gradiente e seriam, para sempre, uma cabeça só repetida 4 vezes.

O que as cabeças **não** acrescentam é capacidade. Com saída de 256, as projeções somam 196 608 parâmetros nas três variantes do E8 — uma cabeça de 256, quatro de 64 no Wrapper, quatro de 64 no weight split. Quatro cabeças de 64 são a mesma quantidade de peso que uma de 256, cortada em fatias. A diferença de parâmetros da `MultiHeadAttention` (262 400) é só a `out_proj`: 256×256 + 256 de bias, a camada que deixa as cabeças trocarem informação depois da concatenação. Sem ela, as fatias saem lado a lado e nunca se misturam.

Wrapper e weight split, por fim, são a mesma conta: com os pesos copiados, diferença máxima de 1,19e-07 nos contextos e 2,98e-08 nos pesos. O weight split é o que o projeto leva adiante porque faz um matmul por matriz em vez de um laço de quatro — 0,52 ms contra 0,64 ms no E8, mesmo carregando a `out_proj` a mais.

---

## Questão 10 — Por que o número de heads não altera os parâmetros, mas altera a memória? (E6 e E7)

Porque parâmetros dependem de `d_in` e `d_out`, e o número de cabeças não aparece em nenhum dos dois. A memória de ativação depende do shape dos scores, `(B, H, T, T)`, e H está nele.

O E6 fixa `d_out` = 256 e varia as cabeças de 1 a 16. Os parâmetros ficam em 262 400 em todas as linhas: as projeções continuam sendo 256 → 256, e o que muda é só em quantos blocos as colunas da saída são lidas (`head_dim` vai de 256 a 16). Já o tensor de scores dobra a cada dobra de H: de 16 384 células (0,06 MB) com uma cabeça para 262 144 (1,00 MB) com 16. Cada cabeça tem sua própria matriz T×T, e T×T não encolhe quando a cabeça fica mais estreita.

O E7 é o espelho: fixa H = 4 e faz a cabeça crescer, de `head_dim` 16 a 128. Agora os parâmetros crescem — de 53 312 para 655 872, mais que linear porque a `out_proj` é `d_out`×`d_out` — e o tensor de scores fica parado em 65 536 células nas quatro linhas. `head_dim` é a dimensão que o produto escalar consome; ela não sobrevive no shape `(B, H, T, T)`.

Juntando os dois: **mais cabeças** com o mesmo `d_out` dão cabeças menores, não rede maior, e custam memória temporária; **cabeças maiores** dão rede maior e não custam ativação. Por isso escalar um GPT aumenta `d_out` e H juntos, mantendo `head_dim` em torno de 64 — que é exatamente a configuração base do projeto (256 em 4 cabeças).

---

## Questão 11 — O que a Sprint 4 recebe desta Sprint? (teste de integração)

Uma `MultiHeadAttention` pronta para entrar no Transformer Block, e um teste que prova que ela encaixa no pipeline da Sprint 2. O `tests/test_attention_pipeline.py` percorre The Verdict → BPE → `create_dataloader` → `InputEmbedding` → `MultiHeadAttention`, na configuração base (lote de 8, T = 128, `emb_dim` 256, 4 cabeças), e fixa quatro garantias:

- **Os shapes fecham de ponta a ponta.** `(8, 128)` de Token IDs vira `(8, 128, 256)` de embeddings, que vira `(8, 128, 256)` de contexto, com pesos `(8, 4, 128, 128)`.
- **A saída tem o mesmo shape da entrada.** Com `d_in = d_out = emb_dim`, a conta `x + attn(x)` é válida. É a conexão residual do bloco da Sprint 4; sem essa igualdade, o bloco não monta.
- **O backward atravessa tudo.** O gradiente chega não nulo nas duas tabelas de embedding e em todas as matrizes da atenção, e a máscara (buffer) não recebe gradiente. Um `detach` esquecido no meio congelaria parte da rede em silêncio, e o treino nunca acusaria.
- **Na tabela de tokens, o gradiente alcança só as linhas dos IDs do lote.** Consequência de a tabela ser indexada e não multiplicada — o que a Sprint 5 vai herdar ao treinar.

O contrato para o capítulo 4:

- Construtor com os nomes do livro: `MultiHeadAttention(d_in, d_out, context_length, num_heads, dropout, qkv_bias)`, com `qkv_bias=False` como no GPT-2.
- Retorno `(context, weights)`: o bloco escreve `x, _ = self.att(x)`.
- `context_length` é o máximo; sequências mais curtas usam o recorte `[:T, :T]` da máscara, e sequências maiores levantam `ValueError` (o teste está na `CausalAttention`, que tem a mesma checagem).
- `train()`/`eval()` ligam e desligam o dropout interno, e a geração de texto precisa rodar em `eval()`.

O que falta para a Sprint 4 é o que fica em volta: LayerNorm, feed-forward, as conexões residuais e o empilhamento dos blocos. A atenção entra como está.

---

## Como reproduzir

```bash
python tests/test_self_attention.py        # 12 testes
python tests/test_causal_attention.py      # 8 testes
python tests/test_multi_head_attention.py  # 9 testes
python tests/test_attention_pipeline.py    # 4 testes (baixa o corpus e a tabela do BPE na 1ª vez)
python experimentos/sprint03/atencao_experimentos.py  # regenera resultados/atencao.md e as figuras
```

Seed fixa em 42 antes de cada tensor sorteado: duas execuções produzem as mesmas tabelas, exceto pelas colunas de tempo, que dependem da máquina.
