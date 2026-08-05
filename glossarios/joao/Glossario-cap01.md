# Glossário — Capítulo 1: Understanding Large Language Models

João Antônio Sarturi Popp — Inteligência Artificial e Sistemas Inteligentes — UNOESC

---

### 1. Machine Learning

**Equivalente:** Aprendizado de Máquina.  
**Definição:** algoritmos que aprendem a partir dos dados em vez de receberem as regras prontas.  
**Função:** é de onde vem a ideia de otimização — o modelo vai ajustando os parâmetros até errar menos no training dataset.  
**Relação:** contém o deep learning. Se opõe aos sistemas de regras.  
**Exemplo (do livro):** filtro de spam. Em vez de escrever as regras na mão, você joga um monte de e-mail marcado como spam ou legítimo e o algoritmo descobre sozinho o que caracteriza cada um.

### 2. Deep Learning

**Equivalente:** Aprendizado Profundo.  
**Definição:** parte do ML que usa neural networks com três camadas ou mais.  
**Função:** é o que torna o LLM possível.  
**Relação:** a diferença prática pro ML tradicional é que aqui não precisa de feature extraction manual — o modelo acha as características sozinho. Os dois ainda precisam de labels quando o treino é supervisionado.  
**Exemplo:** o GPT-3 tem 96 camadas transformer. Não dá pra pensar em escolher atributos na mão nessa escala.

### 3. Feature extraction (manual)

**Equivalente:** extração manual de atributos.  
**Definição:** quando um especialista olha os dados brutos e escolhe quais características vão alimentar o modelo.  
**Função:** nenhuma no LLM — é exatamente o passo que o deep learning eliminou.  
**Relação:** é o que separa ML tradicional de deep learning.  
**Exemplo (do livro):** pra classificar spam na mão, alguém teria que contar quantas vezes aparecem palavras como "prize", "win", "free", quantos pontos de exclamação tem, se tem palavra toda em maiúscula, se tem link suspeito.

### 4. Natural Language Processing (NLP)

**Equivalente:** Processamento de Linguagem Natural.  
**Definição:** área que trata computador lidando com linguagem humana.  
**Função:** é onde o LLM é aplicado.  
**Relação:** antes dos LLMs cada tarefa tinha seu modelo próprio — um pra classificar texto, outro pra traduzir. Funcionavam bem, mas só naquilo. O LLM faz várias tarefas com um modelo só.  
**Exemplo:** tradução, sentiment analysis, question answering, sumarização.

### 5. Neural network / deep neural network

**Equivalente:** rede neural (profunda).  
**Definição:** camadas de unidades ligadas entre si, com pesos que dá pra ajustar. Vira "profunda" de três camadas pra cima.  
**Função:** é a estrutura em cima da qual o transformer é montado.  
**Exemplo:** uma camada simples faz y = Wx, sendo W a matriz de pesos que o treino ajusta.

### 6. Parameters / weights

**Equivalente:** parâmetros / pesos.  
**Definição:** os valores ajustáveis da rede, que o treinamento vai otimizando pra acertar a próxima palavra.  
**Função:** é onde fica guardado tudo que o modelo aprendeu. Também é a medida padrão de tamanho do modelo.  
**Relação:** o livro adianta que vamos aproveitar weights já treinados e disponíveis publicamente, carregando eles na nossa arquitetura. Isso permite pular o pretraining, que é a parte cara.  
**Exemplo:** GPT-3 tem 175 × 10⁹ parameters.

### 7. Transformer

**Definição:** a arquitetura que veio do artigo *Attention Is All You Need*, de 2017. Foi feita originalmente pra tradução automática (inglês → alemão e francês). Tem dois submódulos, encoder e decoder, com várias camadas ligadas por self-attention.  
**Função:** é a base de praticamente todo LLM moderno. O autor atribui o sucesso dos LLMs a ela somada ao volume de dados.  
**Relação:** duas ressalvas que o autor faz e que confundem: nem todo transformer é LLM (usam transformer em computer vision também) e nem todo LLM é transformer (existem LLMs recorrentes e convolucionais, feitos pra ser mais eficientes computacionalmente).  
**Exemplo:** o transformer original repetia os blocos encoder/decoder 6 vezes. O GPT-3 usa 96 camadas.

### 8. Encoder

**Equivalente:** codificador.  
**Definição:** o módulo que pega o texto de entrada e transforma numa série de vetores que carregam a informação de contexto.  
**Função:** monta a representação que o decoder vai usar. Ele enxerga o texto de entrada inteiro.  
**Relação:** é em cima dele que o BERT foi construído. No GPT ele nem existe.  
**Exemplo (figura 1.4):** numa tradução, o encoder pega "This is an example" e devolve embedding vectors pro decoder.

### 9. Decoder

**Equivalente:** decodificador.  
**Definição:** o módulo que recebe os vetores e vai gerando o texto de saída, uma palavra por vez.  
**Função:** é o único módulo que o GPT usa.  
**Relação:** de onde vem o comportamento autoregressive e a next-word prediction.  
**Exemplo (figura 1.4):** com "This is an example" mais a tradução parcial "Das ist ein", ele gera "Beispiel".

### 10. Decoder-only architecture

**Definição:** o GPT joga fora o encoder e fica só com a pilha de decoders. O processamento é unidirecional, da esquerda pra direita.  
**Função:** deixa a arquitetura mais simples e boa pra ir gerando texto de forma iterativa.  
**Relação:** dos três formatos, esse é o do GPT. O BERT é encoder-only e o transformer original é encoder-decoder.

### 11. Attention mechanism

**Equivalente:** mecanismo de atenção.  
**Definição:** o que dá ao modelo acesso seletivo à sequência de entrada inteira enquanto ele vai montando a saída palavra por palavra.  
**Função:** é a ideia central do transformer. O autor diz que está "no coração de todo LLM".  
**Relação:** o capítulo 1 não explica direito, adia pro capítulo 3 por causa da complexidade.  
**Exemplo:** em "O aluno que estava na sala entrou", pra escolher o verbo o modelo precisa olhar pra "aluno" e não pra "sala", mesmo com palavras no meio.

### 12. Self-attention

**Definição:** o tipo de attention que conecta as camadas do encoder e do decoder. Cada token pesa a importância dos outros tokens da mesma sequência em relação a ele.  
**Função:** é o que permite capturar dependências de longo alcance e relações de contexto dentro da entrada, o que melhora a coerência da saída.  
**Relação:** é o componente que vamos implementar na Sprint 3.  
**Exemplo:** em "O prêmio não coube na mala porque ela era pequena", é a self-attention que liga "ela" a "mala" e não a "prêmio".

### 13. Embedding vector

**Definição:** representação numérica do texto que guarda vários fatores diferentes em dimensões diferentes (é a definição que a legenda da figura 1.4 dá).  
**Função:** é o que sai do encoder e entra no decoder — a ponte entre texto e conta.  
**Relação:** o capítulo 1 só usa o termo. Como se constrói isso é o capítulo 2.

### 14. Transformer layer / block

**Equivalente:** camada ou bloco transformer.  
**Definição:** a unidade que se repete empilhada e dá profundidade ao modelo.  
**Relação:** o que tem dentro do bloco o capítulo 1 não fala, só o capítulo 4.  
**Exemplo:** 6 repetições no transformer original, 96 camadas no GPT-3.

### 15. Token

**Definição:** a unidade de texto que o modelo lê. O número de tokens de um dataset dá mais ou menos o número de palavras somado aos sinais de pontuação.  
**Função:** é a unidade em que tudo é medido — tamanho de dataset, custo, contexto.  
**Relação:** sai da tokenization. Como vira número é o capítulo 2.  
**Exemplo:** o CommonCrawl filtrado do GPT-3 tem 410 bilhões de tokens e ocupa uns 570 GB.

### 16. Pretraining

**Equivalente:** pré-treinamento.  
**Definição:** a primeira fase, em cima de um dataset grande, variado e sem label, pra o modelo pegar uma noção ampla de linguagem.  
**Função:** produz o foundation model, que já sai sabendo fazer text completion e um pouco de few-shot.  
**Relação:** é a primeira metade do esquema de dois estágios. É a parte cara.  
**Exemplo:** o do GPT-3 custou ~US$ 4,6 milhões. No projeto vamos fazer com dataset pequeno, rodando em hardware normal.

### 17. Fine-tuning

**Equivalente:** ajuste fino.  
**Definição:** pegar o modelo já pré-treinado e treinar de novo, agora num dataset menor e rotulado, específico de uma tarefa ou domínio.  
**Função:** especializa o modelo gastando bem menos recurso.  
**Relação:** segunda metade do esquema. Se divide em instruction e classification. O livro cita que modelo com fine-tuning em dado próprio pode ganhar de LLM genérico — exemplos: BloombergGPT (finanças) e LLMs pra question answering médico.

### 18. Downstream task

**Equivalente:** tarefa derivada.  
**Definição:** a tarefa específica pra qual o foundation model é adaptado depois, via fine-tuning.  
**Exemplo:** classificar texto ou seguir instrução, que segundo o autor são as mais comuns na prática e em pesquisa.

### 19. Autoregressive model

**Equivalente:** modelo autorregressivo.  
**Definição:** o modelo usa as saídas que ele mesmo já gerou como entrada das próximas predições.  
**Função:** no GPT cada palavra nova é escolhida com base na sequência inteira que veio antes, e é isso que deixa o texto coerente.  
**Relação:** cai direto da arquitetura decoder-only mais a next-word prediction.  
**Exemplo (figura 1.8):** a saída da iteração anterior volta como entrada da seguinte, e assim vai.

### 20. Text completion

**Definição:** conseguir terminar uma frase que o usuário começou, produzindo continuação plausível.  
**Função:** é o que o foundation model já sabe fazer antes de qualquer fine-tuning.  
**Exemplo:** entrada "Breakfast is the most" → "important meal of the day."

### 21. Zero-shot learning

**Definição:** dar conta de uma tarefa totalmente nova sem ter recebido nenhum exemplo dela.  
**Função:** resolve tarefa nova sem retreinar, sem fine-tuning e sem mexer na arquitetura.  
**Exemplo (figura 1.6):** entrada "Translate English to German: breakfast =>" e ele traduz, sem exemplo nenhum antes.

### 22. Few-shot learning

**Definição:** aprender a tarefa a partir de pouquíssimos exemplos que o usuário coloca na própria entrada.  
**Função:** ajuda quando o zero-shot não dá conta, sem precisar mexer no modelo.  
**Exemplo (figura 1.6):** "gaot => goat; sheo => shoe; pohne =>" → "phone".

> **Ponto que demorei pra separar:** zero-shot e few-shot acontecem só na entrada, o modelo não muda. Fine-tuning é outra coisa, ali os weights são realmente treinados de novo.

### 23. Emergent behavior

**Equivalente:** comportamento emergente.  
**Definição:** o modelo faz tarefa que ninguém treinou ele pra fazer. Não é ensinado, aparece por consequência de ter visto muito dado em muito contexto.  
**Função:** é o argumento pra usar um modelo geral em vez de vários especializados.  
**Relação:** vem da escala e da diversidade do dataset combinadas com um objetivo de treino simples.  
**Exemplo:** o caso clássico é tradução. O GPT traduz entre idiomas mesmo tendo sido treinado só pra prever a próxima palavra, e o livro diz que isso pegou os pesquisadores de surpresa.

### 24. Long-range dependencies

**Equivalente:** dependências de longo alcance.  
**Definição:** relação entre partes distantes da sequência de entrada.  
**Função:** é o problema que a self-attention resolve, e o que permite manter coerência em texto longo.  
**Exemplo:** um pronome no fim do parágrafo se referindo a um substantivo lá do começo.
