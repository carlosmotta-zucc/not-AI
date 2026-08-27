# Glossário - Capítulo 2: Working with Text Data
---

## 1. Corpus (corpus / dados de texto)

O texto bruto que serve de matéria-prima para tudo o que o capítulo constrói. Antes de qualquer tokenização, é só uma sequência de caracteres — nenhuma estrutura numérica, nenhum vocabulário, nenhum token.

O corpus é o ponto de partida obrigatório porque toda etapa seguinte depende de tê-lo por inteiro, ao menos uma vez, antes de rodar: a tokenização precisa de uma regra de quebra aplicada sobre ele, e o vocabulário precisa enumerar tudo o que aparece nele para atribuir um índice a cada unidade. Um corpus pequeno tem uma vantagem didática que um corpus grande não tem: dá para conferir contagens na mão e olhar o vocabulário inteiro de uma vez.

O projeto usa _The Verdict_, conto de Edith Wharton de 1908, 20 479 caracteres, baixado automaticamente por `src/corpus.py` na primeira execução. É pequeno o bastante para que cada número citado nas análises da Sprint — 5 145 Token IDs pelo BPE, 1 151 entradas no vocabulário próprio — seja verificável sem infraestrutura nenhuma.

---

## 2. Tokenization (tokenização)

A operação que quebra o corpus em pedaços menores segundo uma regra fixa, antes de qualquer um desses pedaços virar número. É a primeira transformação do pipeline e a que define a granularidade de tudo o que vem depois.

A regra de quebra não é única nem óbvia: pode ser por palavra, por caractere ou por subpalavra, e a escolha tem consequência direta na questão 1 da Frente A — a sequência precisa ser curta porque o custo da atenção cresce com o quadrado do número de tokens. Tokenizar por caractere daria um vocabulário minúsculo e sequências longas demais; por palavra, sequências curtas e vocabulário que quebra em toda palavra nova. O projeto implementa as duas pontas para comparar: um splitter por palavras, escrito à mão (`src/tokenizer/splitter.py`), e o BPE do GPT-2 via `tiktoken` (`src/tokenizer/bpe.py`), que fatia em subpalavras.

No corpus de 20 479 caracteres, o splitter próprio produz 4 730 tokens; o BPE produz 5 145. A diferença não é erro — é a consequência direta de duas regras de quebra diferentes, e é exatamente o que a questão 3 da Frente A investiga.

---

## 3. Token

A unidade que a tokenização produz: uma string, nada além disso. Não existe "o token" de um texto — existe o token de um tokenizador específico, porque tokenizadores diferentes fatiam o mesmo texto em fronteiras diferentes.

A confusão mais comum é tratar token como sinônimo de palavra. Não é: pode ser uma palavra inteira, um pedaço dela, um sinal de pontuação ou um marcador especial como `<|endoftext|>`. A prova de que a identidade do token depende do tokenizador está em comparar os dois implementados no projeto sobre a mesma frase: o splitter próprio e o BPE não concordam nem no número de tokens, nem nas fronteiras entre eles.

Em `"It's the last he painted, you know."`, o splitter próprio produz 11 tokens, incluindo `It`, `'`, `s` como três peças separadas; o BPE produz 10, com `It's` fatiado em só duas peças (`It`, `'s`) e o espaço à esquerda de cada palavra preservado dentro do próprio token (`' the'`, `' last'`). Mesmo texto, dois conjuntos de tokens diferentes.

---

## 4. Vocabulary (vocabulário)

O conjunto fechado de todos os tokens que um tokenizador reconhece, cada um associado a um índice inteiro único. Na implementação, são dois dicionários espelhados — token para ID e ID para token — porque o encode precisa de um sentido e o decode do outro.

O vocabulário é o que transforma tokenização em números: sem ele, um token continua sendo string. Ele também define diretamente o tamanho da tabela de embeddings mais adiante, porque cada entrada do vocabulário vira uma linha treinável dessa tabela — um vocabulário maior custa mais parâmetros antes de qualquer camada de atenção existir. E ele é fechado por construção: uma palavra que não estava no corpus usado para montá-lo simplesmente não tem entrada, o que obriga a alguma estratégia para lidar com o caso (ver Special Tokens).

O vocabulário próprio, construído sobre _The Verdict_ inteiro, tem 1 151 entradas (1 149 do texto mais dois tokens especiais). O vocabulário do BPE do GPT-2 tem 50 257 — já treinado, e não depende do corpus específico do projeto. Com `emb_dim=256`, essa diferença de tamanho é a diferença entre uma tabela de 294 656 parâmetros e uma de 12 865 792, só para representar tokens.

---

## 5. Token ID

O inteiro que identifica um token dentro de um vocabulário específico. É um rótulo nominal: sai da posição do token numa lista ordenada, e essa posição não carrega nenhum significado além de "é aqui que este token mora nesta tabela".

É por ser nominal que o Token ID não pode ser usado diretamente como representação semântica — essa é a questão 4 da Frente A. Ordem, distância e média entre dois IDs não significam nada sobre a relação entre os tokens que eles identificam: dois sinônimos podem cair em pontas opostas do vocabulário, e a média de dois IDs cai num terceiro token arbitrário. O Token ID serve para exatamente uma coisa — indexar a linha correspondente na tabela de embeddings — e é essa indexação, não o valor do ID em si, que o resto do modelo usa.

No vocabulário próprio, `younger` é o ID 1146 e `your` é 1147 — vizinhos porque `y-o-u` vem antes de `y-o-u-r` no alfabeto, não por parentesco de sentido nenhum. Trocar o critério de ordenação (por frequência, por exemplo) mudaria todos os IDs sem mudar nada do que o corpus significa.

---

## 6. Special Tokens (tokens especiais)

Entradas do vocabulário que não correspondem a palavra nenhuma do texto, e sim a uma informação sobre a própria sequência: que uma palavra é desconhecida, ou que um documento terminou e outro começou. Precisam de tratamento fora da regra normal de tokenização, porque a regex do splitter os quebraria em pedaços sem sentido.

Sem tokens especiais, o pipeline tem dois problemas sem solução: o encode quebra assim que aparece uma palavra fora do vocabulário fechado, e não há como sinalizar ao modelo onde um texto termina e o próximo começa quando vários documentos são concatenados num único fluxo de treino. O `SimpleTokenizer` resolve isso fatiando o texto nas ocorrências dos marcadores especiais antes de aplicar a regex de tokenização — exatamente o que o parâmetro `allowed_special` do `tiktoken` faz no BPE.

O projeto implementa os dois tokens especiais do GPT-2: `<|unk|>`, para palavra desconhecida, e `<|endoftext|>`, para fronteira entre documentos. Os dois entram no vocabulário como qualquer outra entrada e por isso também ocupam linhas treináveis na tabela de embeddings.

---

## 7. `<|unk|>` (token desconhecido)

O substituto padrão de qualquer token que não esteja no vocabulário fechado. Evita que o encode quebre com uma exceção quando aparece uma palavra nova, ao custo de apagar a diferença entre todas as palavras desconhecidas.

O ponto central da questão 2 da Frente A é que `<|unk|>` não resolve o problema de vocabulário fechado — só troca uma falha ruidosa (`KeyError`) por uma silenciosa. Duas palavras sem relação semântica nenhuma recebem o mesmo ID quando ambas são desconhecidas, e o decode não tem como distingui-las de volta. É por isso que o BPE, cujo vocabulário fechado é de bytes em vez de palavras, não precisa desse mecanismo: todo texto é feito de bytes, então nada fica de fora.

A frase `"quantum blockchain"`, testada contra o vocabulário próprio tolerante a desconhecidos, codifica como `[1150, 1150]` — o mesmo ID duas vezes — e o decode devolve `<|unk|> <|unk|>`. A informação não foi comprimida, foi apagada.

---

## 8. `<|endoftext|>` (marcador de fim de texto)

O token que sinaliza a fronteira entre documentos independentes concatenados no mesmo fluxo de treinamento. Sem ele, treinar sobre vários textos colados ensinaria o modelo a prever transições que não existem de verdade — como se o fim de um conto continuasse diretamente no início de outro.

É o único token especial que o GPT-2 usa, e o BPE trata seu ID (50 256, o último do vocabulário) como um marcador de controle: só é aceito no encode quando explicitamente permitido, para que um texto não confiável não consiga injetar o marcador como se fosse conteúdo comum. Essa mesma proteção existe no `BPETokenizer` do projeto via `allowed_special`.

No BPE do GPT-2, `bpe.encode("<|endoftext|>")` devolve `[50256]` — um único ID, o maior de todo o vocabulário, tratado como caso especial em vez de fatiado como texto comum.

---

## 9. Byte Pair Encoding — BPE (codificação por pares de bytes)

O algoritmo de tokenização por subpalavras que o GPT-2 usa de verdade, em contraste com o tokenizador por palavras escrito à mão no projeto. Em vez de um vocabulário fechado de palavras inteiras, o BPE parte de bytes individuais e funde repetidamente os pares mais frequentes até atingir o tamanho de vocabulário desejado.

Essa construção resolve o problema de palavra fora do vocabulário pela raiz: como todo texto é, no fim, uma sequência de bytes, e bytes sempre existem, uma palavra nunca vista é fatiada em subpalavras conhecidas ou, no pior caso, em bytes soltos — nunca precisa virar `<|unk|>`. O projeto não reimplementa o algoritmo de fusão; `src/tokenizer/bpe.py` embrulha o `tiktoken` (a implementação oficial, em Rust) atrás da mesma interface `encode`/`decode`/`tokenize` do tokenizador próprio, para que o comparativo entre os dois seja justo.

`Akwirw ier`, uma sequência de caracteres que nunca existiu como palavra, vira 6 tokens pelo BPE, e o decode desses 6 tokens reconstrói a string original exatamente — sem perda, sem `<|unk|>`, ao contrário do que aconteceria no tokenizador por palavras.

---

## 10. Subword (subpalavra)

A unidade que o BPE produz quando uma palavra não corresponde a um único token do vocabulário: um fragmento dela, curto o bastante para já ter aparecido com frequência suficiente durante o treino do BPE para merecer sua própria entrada.

A ideia central é que subpalavras conhecidas conseguem se recombinar em qualquer palavra, mesmo uma que o vocabulário nunca viu por inteiro. É esse mecanismo que substitui o `<|unk|>`: em vez de colapsar tudo o que é desconhecido num único símbolo genérico, o BPE decompõe o desconhecido em pedaços que ele já reconhece. O tamanho da fragmentação também carrega informação — palavras raras ou fora do domínio de treino do BPE tendem a virar mais subpalavras que palavras comuns em inglês.

`attention` vira 2 tokens no BPE do GPT-2 (`att`, `ention`), mas o equivalente em português, `atenção`, vira 3 (`aten`, `ç`, `ão`) — o BPE foi treinado majoritariamente em inglês, então fusões comuns do português nunca ganharam token próprio, e é por isso que o mesmo conteúdo custa 2,15× mais tokens em português que em inglês nas frases medidas pela Frente A.

---

## 11. Context Length (tamanho do contexto)

O número de Token IDs que cabem numa única janela de entrada do modelo — o `T` dos shapes `(batch, context_length)`. É um teto rígido: diferente do vocabulário, não existe posição além do `context_length` para o modelo consultar.

`context_length` é o parâmetro que conecta a tokenização (Frente A) ao restante do pipeline de dados (Frente B): ele determina tanto o tamanho de cada amostra de treino quanto o número de linhas da tabela de embeddings posicionais, que precisa ter uma entrada para cada posição possível dentro da janela. Contextos maiores permitem ao modelo enxergar dependências mais distantes no texto, mas encarecem a atenção quadraticamente e, com stride igual ao contexto, reduzem o número de amostras que um corpus fixo produz por época.

Na configuração de referência da Frente B (`context_length=128`), o corpus de 5 145 Token IDs produz 40 amostras com `stride=128` (sem sobreposição); com `context_length=256`, o mesmo corpus produz só 20.

---

## 12. Sliding Window (janela deslizante)

O mecanismo que recorta a sequência contínua de Token IDs do corpus em pedaços de tamanho `context_length`, avançando um número fixo de posições (`stride`) a cada amostra. É o que transforma uma lista única de IDs em várias amostras de treino de tamanho fixo.

A implementação (`src/dataset/sequences.py`) resolve isso em Python puro antes de qualquer tensor entrar em cena: dado `context_length` e `stride`, uma função calcula quantas janelas cabem na sequência sem gerar nenhuma amostra, e outra gera de fato os pares. Uma decisão de projeto importa aqui e diverge do livro: o pedaço final que não tem Token IDs suficientes para formar uma janela completa (entrada mais alvo) é descartado, nunca preenchido com padding — um alvo de padding ensinaria o modelo a prever algo que não é linguagem real.

Com `context_length=128` sobre o corpus de 5 145 tokens, o `stride=1` produz 5 017 amostras (quase uma por token do corpus, de tão sobreposta que a janela fica); `stride=128` produz só 40.

---

## 13. Stride

O número de Token IDs que a janela deslizante avança entre uma amostra e a próxima. É o parâmetro que controla, sozinho, a sobreposição entre amostras vizinhas — o `context_length` não entra nessa conta.

Quando `stride == context_length`, nenhum token pertence a mais de uma amostra: sobreposição zero, e o menor número possível de amostras por passagem no corpus. Quando `stride < context_length`, amostras vizinhas compartilham `context_length - stride` tokens, o que gera mais amostras a partir do mesmo corpus, mas às custas de reapresentar o mesmo trecho de texto repetidamente, em janelas quase idênticas. `stride > context_length` também é válido e pularia trechos do corpus inteiramente, mas nenhum experimento do projeto explora esse regime.

Em `context_length=256`, `stride=1` faz o corpus de 5 145 tokens gerar 1 251 584 "tokens vistos por época" — uma redundância de ~243× sobre o corpus real — contra apenas 5 120 tokens vistos por época quando `stride=256` (sem sobreposição).

---

## 14. Input-Target Pair (par entrada-alvo)

Cada amostra que a janela deslizante produz: duas sequências de `context_length` Token IDs, em que a segunda (o alvo) é a primeira (a entrada) deslocada exatamente uma posição à frente.

Esse deslocamento de uma posição é o que define a tarefa de um LLM: dados os tokens até a posição `t`, prever o token `t+1`. Não existe rótulo externo nenhum anotado por um humano — o próprio texto contínuo funciona simultaneamente como pergunta e gabarito, e é essa propriedade que permite treinar sobre texto cru, sem qualquer anotação. `alvo[i] == entrada[i+1]` para toda posição, exceto a última do alvo, que é o próximo Token ID do corpus ainda não incluído na entrada.

Com `entrada = [40, 367, 2885, 1464]`, o alvo correspondente é `[367, 2885, 1464, 1807]`: os três primeiros IDs do alvo repetem os três últimos da entrada, e o último ID do alvo (1807) é o token seguinte do corpus que a entrada ainda não continha.

---

## 15. DataLoader

O componente que agrupa pares entrada-alvo individuais em lotes de tensores com shape fixo, `(batch_size, context_length)`, e decide o que fazer quando o número de amostras da época não é múltiplo do `batch_size`. É a última etapa antes de qualquer tensor entrar na camada de embedding.

`create_dataloader` (`src/dataset/loader.py`) expõe dois parâmetros com padrão deliberadamente diferente do PyTorch: `shuffle=True`, porque janelas vizinhas são trechos consecutivos do mesmo texto e, sem embaralhar, um lote inteiro cairia dentro do mesmo parágrafo — o gradiente refletiria aquele assunto específico, não o corpus inteiro; e `drop_last=True`, porque o último lote de uma época tende a vir menor que os demais, e a perda média sobre menos amostras tem variância maior, o que injetaria um passo de escala diferente no fim de cada época.

Com `context_length=128, stride=128` (40 amostras na época), `batch_size=16` produz 2 lotes completos e descarta 8 amostras; `batch_size=8` produz 5 lotes completos sem descartar nenhuma, porque 40 é múltiplo de 8.

---

## 16. Batch (lote)

O agrupamento de várias amostras entrada-alvo num único tensor, processado junto pelo modelo numa mesma passagem. É a dimensão extra, `batch_size`, que aparece antes de `context_length` em todo shape do pipeline a partir do `DataLoader`.

Processar em lote existe por eficiência de hardware — multiplicação de matriz em lote aproveita paralelismo que processar amostra por amostra desperdiçaria — mas também estabiliza o treino: o gradiente calculado sobre um lote é uma média sobre várias amostras, menos ruidosa que o gradiente de uma amostra isolada. O tamanho do lote é, por isso, tanto uma decisão de custo computacional quanto uma decisão sobre a variância do treino.

Na configuração de referência da Frente B (`context_length=128, batch_size=8`), cada lote sai com shape `(8, 128)` de Token IDs, e depois de passar pela camada de embeddings, `(8, 128, 256)` — a dimensão de lote atravessa sem mudança até a saída do modelo.

---

## 17. Embedding

Um vetor de `emb_dim` posições, treinável, que substitui um objeto discreto (um Token ID, uma posição) por um ponto num espaço vetorial contínuo. É a resposta direta à questão 4 da Frente A: o que o Token ID sozinho não conseguia ser.

Diferente do Token ID, cujo valor é arbitrário e fixo desde a construção do vocabulário, um embedding começa em ruído aleatório e é ajustado por backpropagation junto com o resto da rede — o significado que ele carrega não é definido por ninguém, é aprendido a partir do objetivo de treino. É essa propriedade que torna a distância entre dois embeddings potencialmente informativa (sinônimos podem terminar próximos no espaço vetorial), ao contrário da distância entre dois Token IDs, que nunca significa nada.

O projeto usa dois embeddings que se somam: o de token, que captura identidade, e o posicional, que captura ordem — a combinação das duas é o que a Sprint 3 recebe como entrada.

---

## 18. Embedding Layer (camada de embedding)

A implementação concreta de uma tabela de embeddings: uma matriz de pesos `(vocab_size, emb_dim)`, em que cada linha é o vetor de um token do vocabulário. Dado um Token ID, a camada devolve a linha correspondente — uma operação de indexação, não uma multiplicação de matriz completa.

`TokenEmbedding` (`src/embeddings/token_embeddings.py`) é um `nn.Module` com parâmetros porque a tabela precisa ser treinável, ao contrário de uma tabela de consulta fixa. Ela troca um lote de Token IDs de shape `(batch, context)` por um tensor de vetores de shape `(batch, context, emb_dim)`, sem fazer conta nenhuma sobre o valor do ID em si — ele só escolhe qual linha ler. Dois IDs iguais, em posições diferentes da sequência, devolvem exatamente o mesmo vetor nesta etapa; é o motivo pelo qual o embedding posicional precisa existir.

Para o vocabulário do BPE (50 257 entradas) com `emb_dim=256`, essa tabela sozinha tem 12 865 792 parâmetros — o maior bloco de parâmetros do modelo antes de qualquer camada de atenção.

---

## 19. Embedding Dimension (dimensão do embedding)

O número de posições de cada vetor de embedding — `emb_dim` nos shapes do projeto. Controla tanto a capacidade representacional de cada token quanto o custo, em parâmetros e memória, de toda tabela de embedding.

O custo de uma tabela de tokens é exatamente `vocab_size × emb_dim`, então dobrar `emb_dim` dobra os parâmetros dessa tabela, e o mesmo vale para a tabela posicional (`context_length × emb_dim`). Mas os dois fatores não pesam igual: aumentar `vocab_size` de 1 151 para 50 257 (43,7×) tem impacto maior no custo que aumentar `emb_dim` de 16 para 768 (48×) teria isoladamente, porque o vocabulário do BPE já é a maior das duas dimensões em jogo. `emb_dim` também é a largura que se propaga sem alteração por todo o resto da arquitetura — atenção, feed-forward, normalização — então a escolha feita aqui reaparece em cada camada seguinte.

Com `emb_dim=768` (a dimensão do GPT-2 small) sobre o vocabulário do BPE, a tabela de tokens sozinha chega a 38 597 376 parâmetros, contra 804 112 em `emb_dim=16` — exatamente 48× mais, o mesmo fator de crescimento de `emb_dim`.

---

## 20. Positional Embedding (embedding posicional)

Uma segunda tabela de embeddings, indexada pela posição dentro da janela em vez de pelo Token ID, que resolve o ponto cego da tabela de tokens: ela não sabe distinguir a primeira ocorrência de um token da segunda, dentro da mesma sequência.

A tabela posicional tem shape `(context_length, emb_dim)` — uma linha por posição possível, construída sobre `torch.arange(context_length)` para indexar as posições em ordem. Essas são posições absolutas e treináveis, como no GPT-2: a posição 0 tem seu próprio vetor, a posição 1 tem outro, e ambos são ajustados por treino, ao contrário das posições fixas por seno e cosseno do Transformer original. O vetor devolvido não tem dimensão de lote — a posição 2 é a posição 2 para toda sequência do lote, então o broadcasting na soma com o tensor de tokens `(batch, context, emb_dim)` replica o vetor posicional `(context, emb_dim)` automaticamente ao longo da dimensão de lote.

Medido diretamente: o Token ID que se repete nas posições 4 e 66 dos primeiros 128 tokens de _The Verdict_ tem vetores de token idênticos (distância 0,000000) e vetores finais, após a soma posicional, com distância 23,141924 entre si.

---

## 21. Input Embedding (embedding de entrada)

A soma, elemento a elemento, entre o embedding de token e o embedding posicional — o tensor final que sai do pipeline de dados da Sprint 2 e entra na primeira camada de atenção da Sprint 3.

A soma, e não a concatenação, é o que mantém a dimensionalidade `emb_dim` constante em toda a rede — concatenar dobraria a largura do vetor e obrigaria a redimensionar cada camada seguinte só para acomodar um sinal cujo papel é quebrar a simetria entre posições, não adicionar um subespaço de capacidade inteiro. O preço dessa escolha é que conteúdo e posição passam a dividir o mesmo espaço vetorial, e cabe ao treino aprender a separar os dois. `InputEmbedding` (`src/embeddings/positional.py`) implementa exatamente essa soma sobre as duas tabelas.

Na configuração de referência da Frente B, um lote de Token IDs `(8, 128)` vira, depois do `InputEmbedding`, um tensor `(8, 128, 256)` — e o bloco inteiro (tabela de tokens mais tabela posicional) soma 12 898 560 parâmetros treináveis, o ponto de partida sobre o qual a Sprint 3 constrói.
