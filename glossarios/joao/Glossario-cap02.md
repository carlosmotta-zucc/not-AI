# Glossário — Capítulo 2: Working with Text Data

João Antônio Sarturi Popp — Inteligência Artificial e Sistemas Inteligentes — UNOESC

---

### 1. Tokenization

**Equivalente:** tokenização.  
**Definição:** quebrar o texto bruto em unidades menores segundo uma regra fixa — por espaço, por pontuação, por subpalavra ou por byte. É a primeira operação do pipeline, antes de qualquer número aparecer.  
**Função:** define a granularidade em que o modelo enxerga a linguagem. Transforma o texto em unidades que depois viram Token IDs e, por fim, os vetores que o modelo usa.  
**Relação:** produz os tokens que alimentam o vocabulary. A versão didática do capítulo usa regex; a de produção é o BPE.  
**Exemplo (do livro):** re.split(r'([,.:;?_!"()\']|--|\s)', texto) separa "Hello, world." em ['Hello', ',', 'world', '.']. O padrão fica entre parênteses de propósito, pra o próprio delimitador voltar como token — pontuação carrega informação sintática e jogar fora seria perda.

### 2. Token — correção da entrada do capítulo 1

**Equivalente:** token — o termo é usado em português sem tradução.  
**Definição:** lá eu tinha registrado que o número de tokens dá mais ou menos o número de palavras somado à pontuação. Isso vale só pro tokenizador ingênuo. A definição correta é: token é o que quer que o tokenizador escolhido produza — palavra inteira, pedaço de palavra, sinal de pontuação ou marcador especial. Não existe token independente de tokenizador.  
**Função:** a consequência prática dessa correção é que a contagem de tokens não é propriedade do texto, e sim do par texto + tokenizador. O mesmo parágrafo dá contagens diferentes em tokenizadores diferentes.  
**Relação:** o token vira Token ID pelo vocabulary e só depois vira embedding. São três objetos distintos, e é a distinção que organiza o capítulo inteiro.  
**Exemplo:** com o BPE do GPT-2, " the" (com o espaço à esquerda) é um único token, enquanto uma palavra inventada como "Akwirw" é quebrada em vários.

### 3. Vocabulary

**Equivalente:** vocabulário.  
**Definição:** o conjunto de todos os tokens que o modelo reconhece, com um inteiro único associado a cada um. Na prática são dois dicionários espelhados, um de token pra inteiro e outro de inteiro pra token.  
**Função:** determina quais tokens o modelo reconhece e, na camada de embedding, corresponde ao número de vetores que podem ser associados aos Token IDs. Vocabulário maior significa tabela de embeddings maior.  
**Relação:** na implementação didática do capítulo, é construído a partir dos tokens encontrados no corpus, ordenados e associados a IDs. É consultado nas duas direções: o encode usa o dicionário de token pra inteiro, o decode usa o espelhado. O caminho de volta não é perfeito no tokenizador simples — o espaçamento original se perde na quebra e precisa ser remontado por uma regex que cola a pontuação na palavra anterior. Cresce quando os special tokens entram.  
**Exemplo (do livro):** The Verdict, o conto de Edith Wharton usado no capítulo, tem 20 479 caracteres, dá 4 690 tokens pelo tokenizador simples e 1 130 tokens únicos — ou seja, vocabulário de 1 130, que vira 1 132 depois dos dois special tokens. O BPE do GPT-2, pra comparar, tem 50 257.

### 4. Token ID

**Equivalente:** identificador numérico do token.  
**Definição:** o inteiro que identifica um token dentro do vocabulário. É um rótulo arbitrário: sai da posição do token na lista ordenada, então a magnitude e a ordem dele não querem dizer nada.  
**Função:** serve exclusivamente como índice de linha na tabela de embeddings. Nenhuma conta é feita em cima do ID em si.  
**Relação:** é a resposta direta de uma das perguntas de análise da Sprint — por que os Token IDs não são usados como representação semântica. Porque são nominais. O que carrega semântica é o embedding, que é o vetor que o ID vai buscar.  
**Exemplo:** se "rei" for 902 e "rainha" for 1503, a diferença 601 não significa nada e a média dos dois não dá nenhuma palavra do meio. Já os embeddings dessas duas palavras tendem a ficar próximos depois do treino.

### 5. Special tokens

**Equivalente:** tokens especiais.  
**Definição:** entradas do vocabulário que não correspondem a palavras do texto, e sim a informações sobre a sequência — palavra desconhecida e limite entre fontes de texto diferentes.  
**Função:** dão ao modelo como representar coisas que o texto puro não expressa. Sem eles, o tokenizador simples quebra assim que aparece uma palavra fora do vocabulário, e o modelo não tem como saber onde um documento termina e outro começa.  
**Relação:** entram no vocabulary e portanto na tabela de embeddings, ocupando linhas que o modelo também treina. Precisam de tratamento de exceção dentro do encode, senão são tokenizados como texto comum.  
**Exemplo:** os dois implementados no capítulo são <|unk|> e <|endoftext|>.

### 6. <|unk|>

**Equivalente:** token desconhecido.  
**Definição:** o substituto de qualquer token que não esteja no vocabulário.  
**Função:** evita que o encode quebre com KeyError quando aparece uma palavra nova. A limitação é que todas as palavras desconhecidas passam a ser representadas pelo mesmo token, então o que distinguia uma da outra se perde.  
**Relação:** existe por causa da limitação do vocabulário fechado do tokenizador por palavras. O BPE torna ele desnecessário, e por isso o GPT-2 não usa.  
**Exemplo (do livro):** a SimpleTokenizerV2 troca o acesso direto ao dicionário por uma verificação — se o token não estiver em str_to_int, ele vira <|unk|>.

### 7. <|endoftext|>

**Equivalente:** marcador de fim de texto.  
**Definição:** o marcador de fronteira entre documentos independentes concatenados no mesmo fluxo de treinamento.  
**Função:** indica que o conteúdo seguinte pertence a uma fonte de texto independente, ainda que esteja concatenado no mesmo fluxo. Sem isso, treinar sobre vários textos colados faz o modelo aprender transições que não existem.  
**Relação:** é o único special token que o GPT-2 usa. Quando padding é necessário em algum cenário, ele é reaproveitado nesse papel também.  
**Exemplo:** no vocabulário do GPT-2 ele é o ID 50256 — o último, já que o vocabulário tem 50 257 entradas indexadas de 0 a 50256.

### 8. Byte pair encoding (BPE)

**Equivalente:** codificação por pares de bytes.  
**Definição:** algoritmo de tokenização por subpalavras. Parte de um vocabulário mínimo de caracteres ou bytes individuais e vai fundindo, de forma iterativa, o par adjacente mais frequente no corpus, guardando cada fusão como um token novo, até chegar no tamanho de vocabulário desejado.  
**Função:** acaba com o problema de palavra fora do vocabulário, porque no pior caso a palavra é decomposta até virar bytes soltos, que sempre existem — e ainda assim o vocabulário fica em 50 257 em vez de milhões. O efeito que não é óbvio na primeira leitura é que sequências de caracteres frequentes acabam entrando no vocabulário como subwords e sendo reaproveitadas em palavras diferentes.  
**Relação:** é o que o GPT-2 usa de verdade e o que passamos a usar no projeto a partir daqui. Torna o <|unk|> dispensável.  
**Exemplo:** uma palavra inventada como "someunknownPlace" é fatiada em fragmentos conhecidos, e decodificar esses fragmentos devolve a palavra original exatamente igual — nada se perde no caminho.

### 9. Subword

**Equivalente:** subpalavra.  
**Definição:** token que representa uma parte de palavra, ou qualquer sequência de caracteres que o BPE tenha identificado como unidade frequente o bastante pra merecer entrada própria.  
**Função:** permite representar palavras conhecidas e desconhecidas com unidades menores, sem que o vocabulário precise conter todas as palavras possíveis. Fica entre os dois extremos: vocabulário por caractere é minúsculo mas gera sequências longuíssimas; por palavra gera sequências curtas mas explode de tamanho e quebra em palavra nova.  
**Relação:** é o que corrige a estimativa do capítulo 1, que tratava o número de tokens como aproximadamente o de palavras mais a pontuação — com subwords, palavras podem render mais de um token.  
**Exemplo:** texto em português é tokenizado de forma bem menos eficiente pelo BPE do GPT-2, que foi treinado majoritariamente em inglês. A mesma frase gasta mais tokens, ou seja, sobra menos contexto útil.

### 10. tiktoken

**Equivalente:** N/A — nome próprio (biblioteca da OpenAI).  
**Definição:** a biblioteca da OpenAI que implementa os tokenizadores BPE usados pelos modelos GPT.  
**Função:** entrega o tokenizador do GPT-2 já treinado, com as fusões prontas, e é bem mais rápida que a implementação didática em Python. É o tokenizador exato do GPT-2, o mesmo com que o modelo original foi treinado — usar outro faria os Token IDs deixarem de corresponder ao que o modelo aprendeu.  
**Relação:** é o tokenizador que entra dentro do GPTDatasetV1.  
**Exemplo (do livro):** tokenizer = tiktoken.get_encoding("gpt2") e depois tokenizer.encode(texto, allowed_special={"<|endoftext|>"}). Sem o allowed_special, a biblioteca lança exceção ao encontrar o marcador no texto.

### 11. Next-word prediction

**Equivalente:** predição da próxima palavra. Como a partir do BPE a unidade é o token e não a palavra, o mais preciso é predição do próximo token — o livro usa "next-word prediction".  
**Definição:** a tarefa de treino do GPT — dada uma sequência de tokens, prever qual é o token seguinte.  
**Função:** é o que torna o pretraining autossupervisionado. O label não vem de anotador humano, vem do próprio texto: a resposta certa é a palavra que já estava ali. É isso que permite treinar sobre bilhões de tokens sem rotular nada.  
**Relação:** o capítulo 1 já tinha apresentado a ideia; aqui ela vira estrutura de dados concreta, nos input-target pairs. É o que amarra a tarefa ao comportamento autorregressivo do decoder — o modelo só pode olhar pra trás, senão prever o próximo token seria trivial.  
**Exemplo:** de "LLMs learn to predict" saem os pares ("LLMs" → "learn"), ("LLMs learn" → "to") e ("LLMs learn to" → "predict").

### 12. Sliding window

**Equivalente:** janela deslizante.  
**Definição:** a técnica de amostragem que percorre a sequência de Token IDs recortando blocos de tamanho fixo e avançando um número fixo de posições a cada recorte.  
**Função:** transforma um fluxo contínuo de tokens num dataset com número conhecido de amostras, cada uma já no formato (entrada, alvo).  
**Relação:** é parametrizada por max_length e stride, e é o miolo do __init__ do GPTDatasetV1.  
**Exemplo (do livro):** o laço é for i in range(0, len(token_ids) - max_length, stride), com x = token_ids[i:i+max_length] e y = token_ids[i+1:i+max_length+1]. O alvo é literalmente a entrada deslocada uma posição.

### 13. Input-target pairs

**Equivalente:** pares entrada-alvo.  
**Definição:** dois tensores de mesmo comprimento em que o segundo é o primeiro deslocado em uma posição pra direita.  
**Função:** permitem calcular a perda em todas as posições da janela de uma vez só, e não apenas no último token. Uma janela de 4 tokens gera 4 sinais de treino, o que multiplica o aproveitamento de cada amostra.  
**Relação:** saem da sliding window e são o formato final em que o dataset guarda cada amostra.  
**Exemplo (do livro):** x = [40, 367, 2885, 1464] e y = [367, 2885, 1464, 1807]. Na posição 0 o modelo vê [40] e deve prever 367; na posição 1 vê [40, 367] e deve prever 2885; e assim por diante.

### 14. Context size / max_length

**Equivalente:** tamanho de contexto.  
**Definição:** quantos tokens o modelo processa de uma vez em uma sequência.  
**Função:** delimita o horizonte de dependências que o modelo consegue enxergar — nada fora da janela existe pra ele. É também o que define quantas linhas a tabela de positional embeddings precisa ter.  
**Relação:** junto com o stride e com o tamanho do corpus, determina quantas amostras o dataset vai ter.  
**Exemplo:** o GPT-2 usa 1 024. Nos exemplos do capítulo o valor é 4, só pra caber na página e dar pra conferir os números na mão.

### 15. Stride

**Equivalente:** passo da janela.  
**Definição:** quantas posições a janela avança entre uma amostra e a seguinte.  
**Função:** determina quanto a janela avança a cada nova amostra. Com stride igual a max_length não há sobreposição e cada token aparece uma única vez como entrada. Com stride menor, ganha-se mais amostras, mas elas ficam redundantes entre si, o que o livro associa a maior risco de overfitting.  
**Relação:** é o parâmetro que responde à pergunta de análise sobre a relação entre tamanho de contexto e quantidade de amostras de treino.  
**Exemplo:** com max_length = 4 e stride = 4, as janelas não se sobrepõem; com stride = 2, cada janela compartilha metade dos tokens com a anterior.

### 16. Dataset (GPTDatasetV1)

**Equivalente:** conjunto de dados.  
**Definição:** a classe que herda de torch.utils.data.Dataset e implementa três métodos — __init__, que tokeniza o texto e monta todas as janelas; __len__, que devolve o número de amostras; e __getitem__, que devolve o par (x, y) de índice pedido.  
**Função:** padroniza o acesso a uma amostra individual. É esse contrato de três métodos que permite ao DataLoader montar batches sem saber nada sobre o corpus.  
**Relação:** empacota dentro dele o tiktoken, a sliding window, o max_length e o stride.  
**Exemplo (do livro):** todas as janelas são pré-computadas e guardadas em duas listas de tensores no construtor, o que gasta memória mas deixa o __getitem__ trivial.

### 17. DataLoader

**Equivalente:** carregador de dados.  
**Definição:** o iterador do PyTorch que pega amostras de um Dataset e as agrupa em batches, com opções de embaralhar, paralelizar o carregamento e descartar o último batch incompleto.  
**Função:** agrupa as amostras do Dataset em batches e entrega esses dados ao modelo durante o treino, já no formato (batch_size, max_length).  
**Relação:** é a última etapa antes da camada de embedding, e a saída dele define as duas primeiras dimensões de todos os tensores do resto do modelo. Os parâmetros que importam: shuffle evita que o modelo aprenda a ordem do corpus, drop_last descarta o último batch incompleto pra manter a forma constante, e num_workers define quantos processos paralelos pré-processam os dados.  
**Exemplo (do livro):** com batch_size=8 e max_length=4, next(iter(dataloader)) devolve um tensor [8, 4] — oito sequências independentes de quatro tokens, processadas em paralelo.

### 18. Embedding layer

**Equivalente:** camada de embedding.  
**Definição:** o módulo torch.nn.Embedding(vocab_size, output_dim), que guarda uma matriz de pesos com uma linha por token do vocabulário e, dado um tensor de índices, devolve as linhas correspondentes.  
**Função:** é onde os Token IDs viram vetores. O ponto importante é que os pesos começam aleatórios e vão sendo ajustados durante o treino, junto com o resto da rede — a camada faz parte do modelo, não é pré-processamento. O significado de cada token é aprendido, não definido.  
**Relação:** recebe a saída do DataLoader e entrega o token embedding. É matematicamente equivalente a multiplicar um vetor one-hot por uma camada linear sem viés, só que implementada como consulta indexada.  
**Exemplo (do livro):** com torch.manual_seed(123) e torch.nn.Embedding(6, 3), passar torch.tensor([2, 3, 5, 1]) devolve um tensor [4, 3] — quatro linhas da matriz, uma por ID.

### 19. One-hot encoding

**Equivalente:** codificação one-hot.  
**Definição:** representar um token como um vetor do tamanho do vocabulário, com 1 na posição do ID e 0 em todo o resto.  
**Função:** nenhuma na implementação — o livro só menciona de passagem, mas foi o que me fez entender por que a nn.Embedding existe. Multiplicar o vetor one-hot pela matriz de pesos dá exatamente a linha correspondente ao ID, então o resultado é o mesmo; o que muda é que quase toda a multiplicação é por zero e vai fora. Buscar a linha direto pelo índice chega no mesmo lugar sem esse desperdício.  
**Relação:** ajuda a entender por que a consulta na matriz de pesos é descrita como equivalente a essa multiplicação.  
**Exemplo:** com vocabulário de 6, o ID 2 vira [0, 0, 1, 0, 0, 0].

### 20. Token embeddings

**Equivalente:** embeddings de token.  
**Definição:** os vetores que saem da camada de embedding para os Token IDs de um batch.  
**Função:** carregam a informação de qual token é, e só isso. São idênticos pra ocorrências do mesmo token em posições diferentes da sequência, o que é precisamente o problema que os positional embeddings vêm resolver.  
**Relação:** é a construção concreta do embedding vector que o capítulo 1 só tinha citado.  
**Exemplo (do livro):** a forma é [batch_size, context_length, embedding_dim]. Com vocab_size = 50257, output_dim = 256, batch de 8 e max_length = 4, isso dá [8, 4, 256].

### 21. Embedding dimension (output_dim)

**Equivalente:** dimensão do embedding.  
**Definição:** o número de componentes de cada vetor, ou seja, a largura da representação de um token.  
**Função:** controla a capacidade de representação do modelo. Mais dimensões permitem codificar distinções mais finas, ao custo de mais parâmetros e mais memória.  
**Relação:** é a dimensão que atravessa o modelo inteiro: uma vez escolhida, todo tensor que circula pela rede tem essa largura. Junto com o vocab_size, define o tamanho da tabela de embeddings.  
**Exemplo:** 768 no GPT-2; nos exemplos do capítulo é 256, pra ficar mais leve.

### 22. Positional embeddings

**Equivalente:** embeddings posicionais.  
**Definição:** vetores associados à posição que o token ocupa na sequência, e não ao token em si. São somados aos token embeddings, componente a componente.  
**Função:** injetam a noção de ordem, que a self-attention não tem. A atenção compara todos os tokens com todos ao mesmo tempo, sem percorrer a sequência, então sem esse acréscimo "o gato mordeu o rato" e "o rato mordeu o gato" produziriam exatamente o mesmo conjunto de entradas.  
**Relação:** a tabela tem uma linha por posição possível, ou seja, max_length linhas — e não vocab_size. É a diferença que mais confunde entre as duas camadas de embedding.  
**Exemplo (do livro):** torch.nn.Embedding(context_length, output_dim) recebe torch.arange(context_length) e devolve [context_length, embedding_dim], ou seja, [4, 256] — sem dimensão de batch, porque a posição é a mesma em todas as sequências.

### 23. Absolute positional embeddings

**Equivalente:** embeddings posicionais absolutos.  
**Definição:** um vetor próprio pra cada índice de posição — um pra posição 0, outro pra posição 1, e assim por diante.  
**Função:** é o que o GPT usa. Esses vetores são aprendidos durante o treino, começando aleatórios como qualquer outro parâmetro.  
**Relação:** a limitação direta é que o modelo só sabe lidar com posições que existiam na tabela durante o treino, o que amarra o tamanho máximo de contexto à arquitetura.  
**Exemplo:** com context_length = 1024, existem 1 024 vetores de posição treinados; a posição 1 025 simplesmente não tem representação.

### 24. Input embeddings

**Equivalente:** embeddings de entrada.  
**Definição:** a soma dos token embeddings com os positional embeddings. É o tensor que efetivamente entra no primeiro bloco transformer.  
**Função:** junta as duas informações que o modelo precisa antes de qualquer atenção — qual é o token e onde ele está.  
**Relação:** é o produto final do capítulo — o livro encerra dizendo que é esse tensor que vai alimentar o mecanismo de atenção. A soma funciona por broadcasting: [context_length, embedding_dim] somado a [batch_size, context_length, embedding_dim], com o tensor de posições sendo replicado por todas as amostras do batch, porque a posição não depende de qual sequência é.  
**Exemplo (do livro):** input_embeddings = token_embeddings + pos_embeddings, ou [8, 4, 256] + [4, 256] = [8, 4, 256].

### 25. Word2Vec

**Equivalente:** N/A — nome próprio (método de geração de embeddings de palavras).  
**Definição:** um dos primeiros e mais conhecidos métodos de gerar embeddings de palavras, treinado na tarefa auxiliar de prever palavras vizinhas dentro de uma janela. A ideia por trás é que palavras que aparecem nos mesmos contextos tendem a ter sentidos parecidos. Produz vetores estáticos: um fixo por palavra.  
**Função:** entra no capítulo como contraste. LLMs não carregam embedding pré-treinado de fora: treinam a própria tabela junto com o resto da rede, o que dá representações ajustadas à tarefa e ao domínio em vez de genéricas.  
**Relação:** a limitação prática é que um vetor fixo por palavra não desambigua nada. Num LLM, a mesma palavra em contextos diferentes acaba com representação diferente.  
**Exemplo:** no Word2Vec, "banco" tem uma representação fixa, tanto faz se aparece no sentido de assento ou de instituição financeira.
