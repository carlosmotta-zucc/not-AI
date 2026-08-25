# Sprint 2 — Frente A: análise

**Do texto ao Token ID.** Responde às questões 1 a 4 da seção 5 do enunciado da Sprint 2, a partir dos números medidos em [`experimentos/sprint02/resultados/frente-a.md`](../../experimentos/sprint02/resultados/frente-a.md).

Corpus: _The Verdict_, conto de Edith Wharton, 20 479 caracteres — o mesmo do capítulo 2 do livro, o que permite conferir os resultados contra os valores publicados.

---

## O que foi implementado

| Etapa                     | Onde                              | O que faz                                                          |
| ------------------------- | --------------------------------- | ------------------------------------------------------------------ |
| 3.1 Tokenização           | `src/tokenizer/splitter.py`       | Quebra o texto em tokens por regex                                  |
| 3.2 Vocabulário/Token IDs | `src/tokenizer/vocabulary.py`     | Mapeia token ↔ ID, incluindo `<\|unk\|>` e `<\|endoftext\|>`        |
| 3.2 Encode/decode         | `src/tokenizer/simple_tokenizer.py` | Texto → IDs → texto                                               |
| Comparativo               | `src/tokenizer/bpe.py`            | Adapta o BPE do GPT-2 (tiktoken) à mesma interface                  |

Os testes em `tests/test_tokenizer.py` (28, todos passando) fixam cada afirmação feita abaixo — se um deles cair, alguma frase desta análise deixou de ser verdadeira.

### Duas decisões que divergem do livro

**Reconhecer em vez de separar.** O livro usa `re.split` com o delimitador capturado e depois filtra as strings vazias que a operação produz. Aqui a busca é por `re.findall`: o padrão descreve o que *é* token, não o que separa tokens. O resultado é equivalente e não há lista intermediária para limpar.

**Números e acentuação.** O padrão trata `1.500,00` como um token só e usa `\w`, que em Python 3 é unicode — `coração` não vira três tokens. Isso explica a divergência de contagem em relação ao livro: 1 149 tokens distintos no corpus, contra os 1 130 relatados por Raschka. A diferença não é erro de nenhum dos dois; é a consequência direta de duas regras de quebra diferentes, e é justamente o ponto da questão 3 mais abaixo.

---

## Questão 1 — Por que um LLM não pode trabalhar diretamente com o texto bruto?

Porque a rede é uma sequência de operações de álgebra linear, e texto é dado categórico. Não existe multiplicação de matriz por `"gato"`: toda camada do modelo — atenção, feed-forward, normalização — opera sobre tensores de ponto flutuante. O texto precisa virar número antes da primeira operação, e é isso que o pipeline desta Sprint faz.

Mas "virar número" não basta, e é aqui que a resposta fica interessante. Bastasse, o Token ID resolveria o problema e a camada de embedding não existiria. O que a conversão precisa entregar são três coisas, e cada uma exige uma etapa:

1. **Unidades discretas e finitas** — a tokenização. Sem uma regra de quebra, não há o que numerar.
2. **Um índice estável por unidade** — o vocabulário. É o que garante que `"gato"` receba o mesmo número em toda ocorrência, em toda execução.
3. **Uma representação onde a distância signifique algo** — o embedding (Frente B). O ID sozinho não tem essa propriedade, pelo motivo da questão 4.

Há ainda uma restrição prática que o experimento E1 mostra: a sequência precisa ser **curta**. O custo da atenção cresce com o quadrado do número de tokens, então a granularidade escolhida na tokenização define diretamente quanto texto cabe na janela de contexto. Tokenizar por caractere daria um vocabulário minúsculo, mas as sequências ficariam longas demais; por palavra, sequências curtas, mas vocabulário que explode e quebra em palavra nova. O BPE fica no meio: no corpus medido, ~4 caracteres por token.

---

## Questão 2 — Qual é a função do vocabulário?

Fixar o conjunto de unidades que o modelo reconhece e associar a cada uma um índice estável. São dois dicionários espelhados: `token → id` para o encode, `id → token` para o decode.

Duas consequências, ambas medidas:

**O vocabulário define o tamanho da tabela de embeddings.** Cada entrada corresponde a uma linha de pesos treináveis. No corpus: 1 151 entradas (1 149 do texto + 2 especiais) contra as 50 257 do GPT-2. Com a dimensão de embedding de 256 que a Frente B usa, isso é a diferença entre uma tabela de ~295 mil parâmetros e uma de ~12,9 milhões — só para representar tokens, antes de qualquer camada de atenção.

**O vocabulário é fechado, e isso tem custo.** O experimento E2 mostra os dois comportamentos possíveis diante de uma palavra ausente:

| Situação                      | Resultado                                             |
| ----------------------------- | ----------------------------------------------------- |
| Sem `<\|unk\|>`               | `KeyError` — o encode simplesmente quebra             |
| Com `<\|unk\|>`               | Encode funciona, mas a palavra vira um token genérico |
| BPE                           | Fatia em subwords; nada se perde                      |

O `<|unk|>` não resolve o problema, apenas troca a falha ruidosa por uma silenciosa. `quantum blockchain` codifica como `[1150, 1150]`: dois conceitos sem relação nenhuma recebem o **mesmo** ID, e o decode devolve `<|unk|> <|unk|>`. A informação não foi comprimida, foi apagada.

O BPE dissolve o problema em vez de contorná-lo, porque seu vocabulário fechado é de *bytes*, e todo texto é feito de bytes. `Akwirw ier` — palavra que nunca existiu — vira 6 tokens e o decode devolve a string original exatamente. É por isso que o GPT-2 não tem `<|unk|>` e por isso que o projeto leva o BPE para as sprints seguintes.

Os tokens especiais, aliás, exigem tratamento fora da regex. `<|endoftext|>` passado pelo splitter viraria `['<', '|', 'endoftext', '|', '>']` e o marcador se perderia. O `SimpleTokenizer` fatia o texto nas ocorrências dos especiais *antes* de tokenizar — que é exatamente o que o parâmetro `allowed_special` do tiktoken faz.

---

## Questão 3 — Qual é a diferença entre um token e um Token ID?

O token é uma **string**, produzida pela regra de quebra. O Token ID é o **inteiro** que identifica essa string dentro de um vocabulário específico. São objetos de natureza diferente, produzidos por etapas diferentes, e nenhum dos dois existe sem sua etapa.

A distinção fica concreta na mesma frase pelos dois tokenizadores (E4):

```
entrada        : It's the last he painted, you know.
próprio tokens : ['It', "'", 's', 'the', 'last', 'he', 'painted', ',', 'you', 'know', '.']
próprio IDs    : [57, 2, 868, 1007, 616, 547, 761, 5, 1145, 610, 8]
BPE tokens     : ['It', "'s", ' the', ' last', ' he', ' painted', ',', ' you', ' know', '.']
BPE IDs        : [1026, 338, 262, 938, 339, 13055, 11, 345, 760, 13]
```

Três leituras:

**O mesmo texto dá tokens diferentes.** 11 contra 10, e não pelas mesmas fronteiras: `It's` vira três tokens no tokenizador próprio (`It`, `'`, `s`) e dois no BPE (`It`, `'s`). Não existe "o token" de um texto — existe o token *daquele tokenizador*.

**O mesmo token dá IDs diferentes.** `It` é 57 aqui e 1026 no GPT-2. O ID não é propriedade do token, é propriedade do par token + vocabulário.

**O BPE guarda o espaço dentro do token.** ` the` (com espaço à esquerda) é um token diferente de `the`. O tokenizador próprio descarta o espaço na quebra — e é por isso que ele não consegue reconstruir o texto original, enquanto o BPE consegue.

Isso corrige a estimativa do capítulo 1, que tratava o número de tokens como ≈ palavras + pontuação. O E3 mostra o quanto essa aproximação falha fora do inglês: as mesmas 5 frases custam 46 tokens em inglês e 99 em português no BPE — **2,15× mais** para dizer a mesma coisa. Em caracteres por token: 5,85 em inglês contra 2,74 em português.

A causa é visível na fragmentação:

```
attention      -> 2 tokens: ['att', 'ention']
atenção        -> 3 tokens: ['aten', 'ç', 'ão']
informações    -> 7 tokens: ['in', 'form', 'a', 'ç', '�', '�', 'es']
```

O BPE do GPT-2 aprendeu suas fusões sobre um corpus majoritariamente em inglês, então sequências comuns do português nunca ganharam token próprio. Em `informações`, a fronteira entre tokens chega a cair no meio de um caractere UTF-8 de dois bytes — daí os caracteres de substituição na saída, que não são erro de codificação e sim prova de que a unidade do BPE é o byte, não a letra.

A consequência prática é direta e vale para o resto do projeto: com janela de contexto fixa, um texto em português ocupa mais que o dobro do orçamento de tokens. Metade do contexto útil se perde só na escolha do tokenizador.

---

## Questão 4 — Por que os Token IDs não são utilizados diretamente como representação semântica?

Porque o ID é um rótulo **nominal**, e usá-lo como número seria interpretar como quantidade algo que só é nome.

O ID sai da posição do token na lista ordenada alfabeticamente. No vocabulário construído, `younger` é 1146 e `your` é 1147 — vizinhos porque `y-o-u` vem antes de `y-o-u-r` no alfabeto, não por qualquer parentesco de sentido. Trocar a ordenação para frequência mudaria todos os IDs sem mudar nada do que o corpus significa.

Três operações que a rede faria naturalmente sobre números, e que não fazem sentido nenhum sobre IDs:

- **Ordem.** `1147 > 1146` não diz que `your` é "mais" que `younger` em dimensão alguma.
- **Distância.** Se `rei` fosse 902 e `rainha` 1503, a diferença 601 não mede parentesco semântico. Duas palavras sinônimas podem ficar em pontas opostas do vocabulário.
- **Média.** A média de dois IDs cai num terceiro token arbitrário, que não é nenhum tipo de "meio-termo" entre os dois.

Um único escalar também não teria como carregar o que um token significa. Palavras têm muitas dimensões de sentido ao mesmo tempo — registro, classe gramatical, campo semântico, polaridade — e comprimir tudo isso numa reta força relações que não existem.

É por isso que o ID serve para **exatamente uma coisa**: indexar a linha da tabela de embeddings. Nenhuma conta é feita sobre o ID em si. O que carrega semântica é o vetor que o ID vai buscar, e esse vetor começa aleatório e é **aprendido durante o treino**, junto com o resto da rede. O significado não é definido, é ajustado — e é o que a Frente B implementa.

Um detalhe que a implementação deixa explícito: em `o gato mordeu o rato`, os dois `o` recebem o mesmo ID e, portanto, o mesmo embedding, apesar de estarem em posições diferentes. A representação de token não sabe nada sobre posição — daí os positional embeddings, também na Frente B.

---

## Comparativo consolidado

| Propriedade                     | Próprio (por palavras) | BPE do GPT-2 (subword) |
| ------------------------------- | ---------------------- | ---------------------- |
| Tamanho do vocabulário          | 1 151                  | 50 257                 |
| Tokens no corpus                | 4 730                  | 5 145                  |
| Caracteres por token            | 4,33                   | 3,98                   |
| Palavra fora do vocabulário     | `<\|unk\|>` ou KeyError | Fatiada em subwords    |
| Reconstrói o texto original     | Não                    | Sim                    |
| Espaço em branco                | Descartado             | Parte do token         |
| Precisa de `<\|unk\|>`          | Sim                    | Não                    |
| Vocabulário depende do corpus   | Sim                    | Não (já treinado)      |
| Tempo de encode do corpus       | ~1,5 ms                | ~1,5 ms                |

O tokenizador próprio produz **menos** tokens no corpus (4 730 contra 5 145) e tem vocabulário 44× menor — o que parece vantagem até se olhar para as duas linhas que importam. Ele não reconstrói o texto original e depende inteiramente do corpus de construção: aplicado a qualquer texto de fora, degrada em `<|unk|>`. As 4 730 unidades são baratas porque o vocabulário é pequeno, e o vocabulário é pequeno porque não cobre quase nada.

O BPE paga 8,8% mais tokens e um vocabulário maior, e em troca não tem palavra desconhecida, reconstrói o texto sem perda e vale para qualquer texto. Essa é a troca que justifica levá-lo para a Sprint 3 em diante.

Sobre o tempo: os dois empatam em ~1,5 ms neste corpus, mas a comparação não é justa e não deve ser lida como resultado. O tokenizador próprio é Python puro sobre 20 KB; o tiktoken é Rust compilado e escalaria muito melhor. O que o experimento mostra é apenas que nenhum dos dois é gargalo nesta escala.

---

## Conexão com a Frente B e a Sprint 3

O que esta frente entrega para a etapa seguinte:

- **A sequência de Token IDs** — entrada da camada de embedding, que os troca por vetores.
- **O tamanho do vocabulário** — número de linhas dessa tabela (`nn.Embedding(vocab_size, emb_dim)`).
- **O BPE como tokenizador oficial** — é ele que entra no `GPTDataset` da Frente B.

O E1 já antecipa o parâmetro que a Frente B vai variar. A relação entre tamanho de contexto e número de amostras cai fora da janela deslizante, mas o insumo é a contagem total de tokens medida aqui: 5 145 tokens no corpus pelo BPE. Com `max_length = 4` e `stride = 4`, isso dá ~1 286 amostras sem sobreposição; com `stride = 1`, ~5 141 amostras altamente redundantes entre si.

A curva de vocabulário do E1 também vale registrar. A razão entre tokens distintos e tokens totais cai de 0,66 (nos primeiros 500 caracteres) para 0,24 (no corpus inteiro): o vocabulário cresce muito mais devagar que o texto, porque as palavras frequentes se repetem e as novas ficam cada vez mais raras. É o que torna viável um vocabulário fixo de 50 257 entradas cobrir corpora de bilhões de tokens.

---

## Como reproduzir

```bash
python tests/test_tokenizer.py                        # 28 testes
python experimentos/sprint02/frente_a_experimentos.py # regenera os resultados
```

Não há aleatoriedade em nenhum dos quatro experimentos — duas execuções produzem tabelas idênticas, exceto pelo tempo medido no E4. O corpus é baixado automaticamente para `data/` na primeira execução, com verificação de tamanho.
