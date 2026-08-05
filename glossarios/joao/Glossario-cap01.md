# Glossário — Capítulo 1: Understanding Large Language Models

João Antônio Sarturi Popp — Inteligência Artificial e Sistemas Inteligentes — UNOESC

---

### 1. Machine Learning

**Equivalente:** Aprendizado de Máquina.  
**Definição:** algoritmos que inferem as regras a partir de exemplos, em vez de recebê-las escritas por um programador. Inverte o modelo clássico: no lugar de codificar se ... então ... para cada caso, monta-se um training dataset e o algoritmo ajusta um modelo interno até reduzir o erro das próprias previsões.  
**Função:** introduz o mecanismo que sustenta todo o resto — a otimização. O modelo corrige os parâmetros na direção que diminui o erro no training dataset, e o objetivo real é generalizar para dados que ele nunca viu, não decorar os de treino.  
**Relação:** contém o deep learning. Se opõe aos sistemas de regras.  
**Exemplo (do livro):** filtro de spam. Em vez de escrever as regras na mão, você joga um monte de e-mail marcado como spam ou legítimo e o algoritmo descobre sozinho o que caracteriza cada um.

### 2. Deep Learning

**Equivalente:** Aprendizado Profundo.  
**Definição:** parte do ML que usa neural networks com três camadas ou mais. O "deep" é literalmente a profundidade da pilha, e cada camada aprende uma representação mais abstrata a partir da saída da camada anterior.  
**Função:** elimina a necessidade de escolher atributos na mão — o próprio modelo descobre o que é relevante nos dados brutos. É isso que torna viável trabalhar com dado não estruturado como texto, que é o insumo do LLM.  
**Relação:** a diferença prática pro ML tradicional é que aqui não precisa de feature extraction manual — o modelo acha as características sozinho. Os dois ainda precisam de labels quando o treino é supervisionado.  
**Exemplo:** o GPT-3 tem 96 camadas transformer. Não dá pra pensar em escolher atributos na mão nessa escala.

### 3. Feature extraction (manual)

**Equivalente:** extração manual de atributos.  
**Definição:** etapa em que um especialista humano examina os dados brutos e decide, antes do treino, quais características vão alimentar o modelo. O desempenho fica limitado pela qualidade dessa escolha.  
**Função:** nenhuma no LLM — é exatamente o passo que o deep learning eliminou. Registrar aqui serve pra marcar o contraste: o gargalo saiu do conhecimento do especialista e passou pro volume de dados e de computação.  
**Relação:** é o que separa ML tradicional de deep learning.  
**Exemplo (do livro):** pra classificar spam na mão, alguém teria que contar quantas vezes aparecem palavras como "prize", "win", "free", quantos pontos de exclamação tem, se tem palavra toda em maiúscula, se tem link suspeito.

### 4. Natural Language Processing (NLP)

**Equivalente:** Processamento de Linguagem Natural.  
**Definição:** área que faz o computador interpretar e gerar linguagem humana. Reúne tarefas como tradução, classificação de texto, análise de sentimento, sumarização e resposta a perguntas.  
**Função:** é o campo onde o LLM se encaixa. Os métodos anteriores iam bem em categorizar e reconhecer padrão simples, mas travavam justamente onde o LLM é forte: compreensão de contexto e geração de texto coerente.  
**Relação:** antes dos LLMs cada tarefa tinha seu modelo próprio — um pra classificar texto, outro pra traduzir. Funcionavam bem, mas só naquilo. O LLM faz várias tarefas com um modelo só.  
**Exemplo:** tradução, sentiment analysis, question answering, sumarização.

### 5. Neural network / deep neural network

**Equivalente:** rede neural (profunda).  
**Definição:** camadas de unidades ligadas entre si, cada conexão com um peso ajustável. Cada camada faz uma transformação linear seguida de uma função de ativação não linear. Vira "profunda" de três camadas pra cima.  
**Função:** é a estrutura em cima da qual o transformer é montado. A não linearidade é o que justifica empilhar camadas — sem ela, qualquer pilha de camadas colapsaria matematicamente numa única transformação linear.  
**Exemplo:** uma camada faz h = σ(Wx + b), sendo W a matriz de pesos, b o vetor de vieses e σ a ativação.

### 6. Parameters / weights

**Equivalente:** parâmetros / pesos.  
**Definição:** os valores ajustáveis que o treinamento otimiza. Englobam os weights (as matrizes de conexão) e os biases (os vieses). Na prática "weights" é usado como sinônimo de parâmetros, mas é o subconjunto maior deles. Partem de valores aleatórios e vão sendo corrigidos a cada passo do treino.  
**Função:** é onde fica guardado tudo que o modelo aprendeu — treinado, o modelo é literalmente esse conjunto de números salvos em disco. Também é a medida padrão de tamanho do modelo e o primeiro sentido do "large" em LLM. Vale notar que não guardam os textos de treino, e sim relações estatísticas extraídas deles.  
**Relação:** o livro adianta que vamos aproveitar weights já treinados e disponíveis publicamente, carregando eles na nossa arquitetura. Isso permite pular o pretraining, que é a parte cara.  
**Exemplo:** GPT-3 tem 175 × 10⁹ parameters.

### 7. Transformer

**Definição:** a arquitetura que veio do artigo *Attention Is All You Need*, de 2017. Foi feita originalmente pra tradução automática (inglês → alemão e francês). Tem dois submódulos, encoder e decoder, com várias camadas ligadas por self-attention.  
**Função:** é a base de praticamente todo LLM moderno. A característica que a destaca é processar a sequência inteira de uma vez, sem depender de percorrer palavra por palavra em ordem — o que resolve as dependências longas e ainda permite paralelizar o treino, que foi o que viabilizou a escala atual. O autor atribui o sucesso dos LLMs a ela somada ao volume de dados.  
**Relação:** duas ressalvas que o autor faz e que confundem: nem todo transformer é LLM (usam transformer em computer vision também) e nem todo LLM é transformer (existem LLMs recorrentes e convolucionais, feitos pra ser mais eficientes computacionalmente).  
**Exemplo:** o transformer original repetia os blocos encoder/decoder 6 vezes. O GPT-3 usa 96 camadas.

### 8. Encoder

**Equivalente:** codificador.  
**Definição:** o módulo que lê o texto de entrada inteiro de uma vez e o transforma numa série de vetores que carregam a informação de contexto. É bidirecional: cada token enxerga tanto o que vem antes quanto o que vem depois dele.  
**Função:** monta a representação que o decoder vai usar. Essa bidirecionalidade o torna bom em compreender e inútil pra gerar, porque gerar exige justamente não conhecer o que vem à frente.  
**Relação:** é em cima dele que o BERT foi construído. No GPT ele nem existe.  
**Exemplo (figura 1.4):** numa tradução, o encoder pega "This is an example" e devolve embedding vectors pro decoder.

### 9. Decoder

**Equivalente:** decodificador.  
**Definição:** o módulo que recebe os vetores e vai gerando o texto de saída, um token por vez. Processa de forma unidirecional, da esquerda pra direita: só enxerga o que já veio antes, nunca o que está à frente.  
**Função:** é o único módulo que o GPT usa. A cada passo emite um token, que volta como entrada do passo seguinte — é essa realimentação que dá coerência ao texto, já que toda palavra nova é condicionada à sequência inteira que a precede.  
**Relação:** de onde vem o comportamento autoregressive e a tarefa de prever a próxima palavra.  
**Exemplo (figura 1.4):** com "This is an example" mais a tradução parcial "Das ist ein", ele gera "Beispiel".

### 10. Decoder-only architecture

**Definição:** o GPT descarta o encoder e fica só com a pilha de decoders. Sem o módulo de entrada separado, o prompt do usuário e o texto que o modelo já gerou passam pela mesma pilha.  
**Função:** simplifica a arquitetura e a especializa em geração iterativa. O custo dessa escolha é a perda da visão bidirecional, o que deixa o GPT abaixo do BERT em tarefas puras de classificação.  
**Relação:** dos três formatos, esse é o do GPT. O BERT é encoder-only e o transformer original é encoder-decoder.

### 11. Attention mechanism

**Equivalente:** mecanismo de atenção.  
**Definição:** o que dá ao modelo acesso seletivo à sequência de entrada inteira enquanto ele vai montando a saída palavra por palavra. Em vez de comprimir a entrada num único estado, o modelo consulta todas as posições e distribui pesos de relevância entre elas.  
**Função:** é a ideia central do transformer e o que substitui o processamento sequencial das arquiteturas antigas. O autor diz que está "no coração de todo LLM".  
**Relação:** o capítulo 1 não explica direito, adia pro capítulo 3 por causa da complexidade.  
**Exemplo:** em "O aluno que estava na sala entrou", pra escolher o verbo o modelo precisa olhar pra "aluno" e não pra "sala", mesmo com palavras no meio.

### 12. Self-attention

**Definição:** o tipo de attention que conecta as camadas do encoder e do decoder. O "self" está aí porque a comparação é da sequência com ela mesma: cada token pesa a importância de todos os outros tokens da mesma entrada em relação a ele.  
**Função:** é o que permite capturar dependências de longo alcance e relações de contexto dentro da entrada, o que melhora a coerência da saída. Qualquer posição acessa qualquer outra diretamente, sem importar a distância, e como não existe dependência sequencial o cálculo pode ser paralelizado. Os pesos de atenção são calculados na hora, para cada entrada — não ficam guardados no modelo.  
**Relação:** é o componente que vamos implementar na Sprint 3.  
**Exemplo:** em "O prêmio não coube na mala porque ela era pequena", é a self-attention que liga "ela" a "mala" e não a "prêmio".

### 13. Embedding vector

**Definição:** representação numérica do texto que guarda vários fatores diferentes em dimensões diferentes (é a definição que a legenda da figura 1.4 dá).  
**Função:** é a ponte entre texto e conta — a rede só opera sobre números, então tudo precisa virar vetor antes. No transformer original, é o que sai do encoder e entra no decoder.  
**Relação:** o capítulo 1 só usa o termo. Como se constrói isso é o capítulo 2.

### 14. Transformer layer / block

**Equivalente:** camada ou bloco transformer.  
**Definição:** a unidade que se repete empilhada e dá profundidade ao modelo. Cada bloco tem a mesma estrutura interna, mas pesos próprios e independentes dos demais.  
**Função:** é o que se escala pra aumentar a capacidade do modelo — empilhar mais blocos permite representações progressivamente mais abstratas. A contagem de camadas é, junto com o número de parâmetros, a descrição padrão do tamanho de uma arquitetura.  
**Relação:** o que tem dentro do bloco o capítulo 1 não fala, só o capítulo 4.  
**Exemplo:** 6 repetições no transformer original, 96 camadas no GPT-3.

### 15. Token

**Definição:** a unidade de texto que o modelo lê, obtida quebrando o texto antes de qualquer conta. O modelo nunca vê letras nem palavras — vê sequências de tokens. Pra estimativa, o número de tokens de um dataset dá mais ou menos o número de palavras somado aos sinais de pontuação, mas o capítulo 2 mostra que a quebra em subpalavras produz mais tokens do que palavras.  
**Função:** é a unidade em que tudo é medido — tamanho de dataset, custo de API e limite de contexto. É também sobre tokens, e não sobre palavras, que a self-attention calcula os pesos de relevância.  
**Relação:** sai da tokenization. Como vira número é o capítulo 2.  
**Exemplo:** o CommonCrawl filtrado do GPT-3 tem 410 bilhões de tokens e ocupa uns 570 GB.

### 16. Pretraining

**Equivalente:** pré-treinamento.  
**Definição:** a primeira fase, em cima de um dataset grande, variado e sem label, pra o modelo pegar uma noção ampla de linguagem. Ninguém rotula nada: a tarefa é prever a próxima palavra e a resposta certa é simplesmente a palavra que já estava ali no texto original, o que faz o label sair do próprio dado.  
**Função:** produz o foundation model, que já sai sabendo fazer text completion e um pouco de few-shot. É essa dispensa de rotulagem humana que torna possível treinar sobre trilhões de palavras — anotar esse volume à mão seria inviável.  
**Relação:** é a primeira metade do esquema de dois estágios. É a parte cara.  
**Exemplo:** o do GPT-3 custou ~US$ 4,6 milhões. No projeto vamos fazer com dataset pequeno, rodando em hardware normal.

### 17. Fine-tuning

**Equivalente:** ajuste fino.  
**Definição:** pegar o modelo já pré-treinado e treinar de novo, agora num dataset menor e rotulado, específico de uma tarefa ou domínio. Aqui os labels vêm de fora, no esquema supervisionado tradicional — ao contrário do pretraining.  
**Função:** especializa o modelo gastando bem menos recurso. Como os parâmetros já chegam ajustados pelo pretraining, o treino apenas os refina em vez de partir do zero, o que reduz drasticamente a exigência de dado e de computação.  
**Relação:** segunda metade do esquema. Se divide em instruction (pares de instrução e resposta, que transformam o modelo em assistente) e classification (textos com suas classes, que o transformam em classificador). O livro cita que modelo com fine-tuning em dado próprio pode ganhar de LLM genérico — exemplos: BloombergGPT (finanças) e LLMs pra question answering médico.

### 18. Downstream task

**Equivalente:** tarefa derivada.  
**Definição:** a tarefa específica pra qual o modelo pré-treinado é adaptado depois, via fine-tuning. O "downstream" indica a posição no fluxo: vem depois e a jusante do pretraining, aproveitando o que ele produziu.  
**Função:** é o que justifica economicamente o esquema de dois estágios — um único pretraining caro alimenta muitas tarefas derivadas baratas.  
**Exemplo:** classificar texto ou seguir instrução, que segundo o autor são as mais comuns na prática e em pesquisa.

### 19. Autoregressive model

**Equivalente:** modelo autorregressivo.  
**Definição:** o modelo usa as saídas que ele mesmo já gerou como entrada das próximas predições. Gera um token, anexa esse token à sequência e roda tudo de novo.  
**Função:** no GPT cada palavra nova é escolhida com base na sequência inteira que veio antes, e é isso que deixa o texto coerente. Tem três consequências práticas diretas: explica o efeito de digitação em streaming das interfaces de chat, explica por que texto longo custa progressivamente mais (a entrada cresce a cada passo) e explica por que erro cometido cedo se propaga, já que vira contexto fixo pro resto da geração.  
**Relação:** cai direto da arquitetura decoder-only mais a tarefa de prever a próxima palavra.  
**Exemplo (figura 1.8):** a saída da iteração anterior volta como entrada da seguinte, e assim vai.

### 20. Text completion

**Definição:** conseguir terminar uma frase que o usuário começou, produzindo continuação plausível.  
**Função:** é o que o foundation model já sabe fazer antes de qualquer fine-tuning, e é consequência direta do objetivo de treino: um modelo treinado pra prever a próxima palavra completa texto por construção. Note que completar não é obedecer — seguir instrução só aparece depois do instruction fine-tuning.  
**Exemplo:** entrada "Breakfast is the most" → "important meal of the day."

### 21. Zero-shot learning

**Definição:** dar conta de uma tarefa totalmente nova sem ter recebido nenhum exemplo dela, apenas a descrição no prompt.  
**Função:** resolve tarefa nova sem retreinar, sem fine-tuning e sem mexer na arquitetura. Serve como medida de o quanto o modelo generaliza para tarefas que nunca viu formuladas daquele jeito.  
**Exemplo (figura 1.6):** entrada "Translate English to German: breakfast =>" e ele traduz, sem exemplo nenhum antes.

### 22. Few-shot learning

**Definição:** aprender a tarefa a partir de pouquíssimos exemplos que o usuário coloca na própria entrada, antes do pedido real.  
**Função:** ajuda quando o zero-shot não dá conta, sem precisar mexer no modelo. É especialmente útil quando o formato da saída importa e é mais fácil demonstrar do que descrever. Como o "aprendizado" acontece só dentro da janela de contexto daquela chamada, ele desaparece quando a conversa acaba — daí o nome in-context learning.  
**Exemplo (figura 1.6):** "gaot => goat; sheo => shoe; pohne =>" → "phone".

> **Ponto que demorei pra separar:** zero-shot e few-shot acontecem só na entrada, o modelo não muda. Fine-tuning é outra coisa, ali os weights são realmente treinados de novo.

### 23. Emergent behavior

**Equivalente:** comportamento emergente.  
**Definição:** o modelo faz tarefa que ninguém treinou ele pra fazer. Não é ensinado, aparece por consequência de ter visto muito dado em muito contexto.  
**Função:** é o argumento pra usar um modelo geral em vez de vários especializados. Traz junto uma consequência incômoda: fica difícil prever de antemão o que um modelo maior vai saber fazer, e igualmente difícil avaliá-lo, já que a lista de habilidades não é conhecida antes de o treino terminar.  
**Relação:** vem da escala e da diversidade do dataset combinadas com um objetivo de treino simples. A explicação plausível é que, pra prever bem a próxima palavra num corpus com texto traduzido, código e diálogo, o modelo precisa internalizar as estruturas por trás desses textos.  
**Exemplo:** o caso clássico é tradução. O GPT traduz entre idiomas mesmo tendo sido treinado só pra prever a próxima palavra, e o livro diz que isso pegou os pesquisadores de surpresa.

### 24. Long-range dependencies

**Equivalente:** dependências de longo alcance.  
**Definição:** relação entre partes distantes da sequência de entrada, em que interpretar um trecho exige informação dita muito antes.  
**Função:** é o problema que a self-attention resolve, e o que permite manter coerência em texto longo. Era exatamente aí que as arquiteturas recorrentes falhavam: elas liam palavra por palavra carregando um estado interno comprimido, então a informação do começo da sequência se degradava até chegar ao fim.  
**Exemplo:** um pronome no fim do parágrafo se referindo a um substantivo lá do começo.
