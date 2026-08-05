# Glossário - Capítulo 1
---

## 1. Machine Learning (aprendizado de máquina)

Algoritmos que aprendem as regras a partir dos dados, em vez de receber as regras prontas escritas por um programador. Você mostra exemplos, o algoritmo descobre o padrão sozinho.

É um subcampo da IA e representa uma inversão do modelo tradicional de programação. No jeito clássico, você escreve `se ... então ...` para cada situação. Em machine learning, você monta um conjunto de treino com exemplos e respostas corretas (os rótulos), e o algoritmo ajusta um modelo interno tentando minimizar o erro das próprias previsões. Depois de treinado, esse modelo é usado em dados novos, que ele nunca viu, para prever o rótulo desconhecido. O machine learning tradicional tem uma limitação importante: ele depende de extração manual de features, ou seja, um especialista humano precisa decidir quais características dos dados são relevantes.

Filtro de spam. Em vez de escrever manualmente "se contém a palavra prêmio, marque como spam", você alimenta o algoritmo com milhares de e-mails já rotulados como spam e não spam. Ele aprende sozinho quais combinações de características indicam spam e passa a classificar e-mails novos.

---

## 2. Neural Networks (redes neurais)

Um modelo formado por camadas de nós interligados, onde cada conexão tem um peso numérico. Entram números, eles são multiplicados e somados camada a camada, e saem números.

O nome vem da inspiração em neurônios biológicos, mas na prática o que acontece é multiplicação de matrizes seguida de uma função não linear de ativação. A rede tem uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. A não linearidade é o que permite modelar relações complexas: sem ela, empilhar camadas seria equivalente a ter uma única camada. O treinamento funciona comparando a saída da rede com a resposta esperada, calculando o erro e propagando esse erro de volta pelas camadas (backpropagation) para corrigir os pesos aos poucos.

Uma rede que recebe três números descrevendo um e-mail (quantidade de links, número de exclamações, tamanho do texto), passa esses valores por duas camadas ocultas e devolve um número entre 0 e 1 representando a probabilidade de ser spam.

---

## 3. Parâmetros

São os valores ajustáveis dentro da rede, os pesos das conexões. São exatamente o que o treinamento modifica e o que o modelo "sabe" depois de treinado.

Os parâmetros começam com valores aleatórios e vão sendo corrigidos a cada passo do treino, na direção que reduz o erro nas previsões. Quando o treino termina, o modelo é literalmente esse conjunto de números salvos em disco. A quantidade de parâmetros é a medida mais comum do tamanho de um modelo, e é a ela que o "large" de large language model se refere, junto com o tamanho do dataset. Vale a nota importante: parâmetros não armazenam os textos de treino, eles armazenam relações estatísticas extraídas desses textos.

Na equação `y = a * x + b`, os parâmetros são `a` e `b`, e treinar seria encontrar os valores de `a` e `b` que melhor se ajustam aos seus dados. Escalando isso: o GPT-3 tem 96 camadas de transformer e 175 bilhões de parâmetros.

---

## 4. Deep Learning (aprendizado profundo)

Machine learning feito com redes neurais de muitas camadas. É um subconjunto do machine learning, não algo paralelo a ele.

O "deep" se refere justamente à profundidade, três ou mais camadas ocultas empilhadas. A vantagem prática em relação ao machine learning tradicional é que ele dispensa a extração manual de features: a própria rede descobre quais características importam, e cada camada aprende representações progressivamente mais abstratas a partir da camada anterior. Por isso deep learning é especialmente bom com dados não estruturados como imagem, áudio e texto, que é exatamente o tipo de dado com que os LLMs trabalham. A coleta de rótulos, porém, continua necessária no aprendizado supervisionado.

Voltando ao spam: no machine learning tradicional, um especialista definiria as features (frequência de palavras como "grátis", número de exclamações, presença de links suspeitos) antes do treino. No deep learning, você entrega o texto do e-mail e a rede descobre por conta própria o que é relevante.

---

## 5. NLP (Natural Language Processing)

Processamento de linguagem natural, a área que faz o computador lidar com linguagem humana, tanto interpretando quanto gerando texto.

É o campo onde os LLMs se encaixam. As tarefas clássicas de NLP incluem tradução automática, classificação de texto, análise de sentimento, sumarização e resposta a perguntas. Antes dos LLMs, cada uma dessas tarefas costumava exigir um modelo próprio, treinado especificamente para ela: um modelo era bom em classificar, outro em traduzir, e nenhum deles transferia bem essa habilidade para tarefas fora do seu escopo. Métodos anteriores funcionavam bem em categorização e reconhecimento de padrões simples, mas falhavam em tarefas que exigiam compreensão contextual e geração de texto coerente.

Um modelo de NLP antigo conseguia classificar uma avaliação de produto como positiva ou negativa, mas não conseguia escrever um e-mail a partir de uma lista de palavras-chave, tarefa trivial para um LLM atual.

---

## 6. Tokens

São os pedaços em que o texto é quebrado antes de virar número. Um token nem sempre é uma palavra: pode ser uma palavra inteira, um fragmento dela ou um sinal de pontuação.

A tokenização é o primeiro passo do pré-processamento, antes de qualquer conta. Cada token recebe um número inteiro que corresponde à sua posição em um vocabulário fixo, definido durante o treino. A quebra em pedaços menores que a palavra existe para resolver um problema prático: se o vocabulário só tivesse palavras inteiras, qualquer termo raro, nome próprio ou erro de digitação ficaria de fora e o modelo não teria como representá-lo. Quebrando em subpalavras, dá para montar qualquer sequência a partir de peças conhecidas. Vale fixar que o modelo nunca vê letras nem palavras, ele vê apenas sequências de IDs de tokens, e é sobre esses tokens que o self-attention calcula a importância de um em relação ao outro. Token também é a unidade prática de medida em LLMs: limite de contexto, custo de API e tamanho de dataset são todos contados em tokens.

"This is an example" vira quatro tokens diretos, um por palavra. Já uma palavra incomum como "tokenization" pode ser quebrada em algo como "token" + "ization", duas peças que o vocabulário já conhece, mesmo que a palavra completa nunca tenha aparecido no treino.

---

## 7. Encode (codificar)

Transformar texto em números, mais especificamente em vetores, porque a rede neural só sabe fazer contas com números.

O texto precisa passar por etapas de pré-processamento até virar uma representação numérica que capture informação contextual, não apenas a identidade isolada de cada palavra. Essa representação em vetores é chamada de embedding, e a ideia é que ela capture vários fatores em dimensões diferentes. Na arquitetura transformer original, o encoder é justamente o módulo que processa o texto de entrada e produz esses vetores; o decoder é o módulo que faz o caminho inverso, pegando os vetores e gerando texto de saída uma palavra por vez. BERT é construído sobre o submódulo encoder, enquanto GPT usa apenas o decoder.

Em uma tradução, o encoder recebe "This is an example", codifica essa frase em vetores que representam seu significado e contexto, e o decoder usa esses vetores para gerar "Das ist ein Beispiel", palavra por palavra.

---

## 8. Transformer

A arquitetura de rede neural profunda sobre a qual quase todos os LLMs modernos são construídos. Sua característica central é conseguir olhar para todas as palavras da entrada ao mesmo tempo e decidir quais delas importam mais em cada momento.

Foi apresentada em 2017 no paper "Attention Is All You Need" e nasceu para tradução automática, inglês para alemão e francês. O desenho original tem dois submódulos: o encoder, que processa o texto de entrada e o transforma em vetores carregados de contexto, e o decoder, que consome esses vetores e gera o texto de saída uma palavra por vez. Ambos são formados por várias camadas ligadas pelo self-attention, o mecanismo que permite ao modelo pesar a importância de cada token em relação aos outros da sequência. É isso que resolve o problema de dependências longas, quando uma palavra no fim da frase depende de algo dito lá no começo. As variantes posteriores dividiram a arquitetura: o BERT usa apenas o encoder e se especializa em prever palavras mascaradas, útil para classificação; o GPT usa apenas o decoder, processa da esquerda para a direita e é feito para gerar texto. O detalhamento do self-attention fica para o capítulo 3. Vale registrar o alerta que o próprio livro faz, porque é fonte comum de confusão: nem todo transformer é um LLM, já que a arquitetura também é usada em visão computacional, e nem todo LLM é um transformer, porque existem LLMs construídos sobre arquiteturas recorrentes e convolucionais, propostas para reduzir custo computacional.

O transformer original repetia seus blocos de encoder e decoder seis vezes. O GPT-3 é decoder-only e empilha 96 camadas de transformer, usando cada palavra gerada como parte da entrada da próxima rodada, o que é chamado de comportamento autorregressivo.

---

## 9. Self-attention (auto-atenção)

Mecanismo que deixa cada posição da sequência olhar para todas as outras e decidir quais são relevantes para interpretá-la. É o coração do transformer.

O "self" está aí porque a comparação é da sequência com ela mesma: cada token calcula um peso de relevância em relação a todos os demais tokens da mesma entrada. Isso resolve o problema que travava as arquiteturas anteriores. As redes recorrentes processavam o texto palavra por palavra em ordem, carregando um estado interno que ia sendo comprimido a cada passo, então informação do começo da frase se degradava ao chegar no fim, o que se chama de problema de dependência de longa distância. Com self-attention, qualquer posição acessa qualquer outra diretamente, sem importar a distância entre elas, e como não há dependência sequencial o cálculo pode ser paralelizado, o que é o que tornou viável treinar em escala. Vale notar que esses pesos de atenção são calculados dinamicamente para cada entrada, eles não são parâmetros fixos guardados no modelo. O capítulo 3 inteiro do livro é dedicado a implementar isso.

Na frase "O carro que estava estacionado na garagem quebrou", o verbo "quebrou" precisa se conectar a "carro" e não a "garagem", que está bem mais perto. O self-attention é o que permite ao modelo atribuir peso alto justamente à palavra distante e correta.

---

## 10. Encoder (codificador)

O módulo que lê a entrada inteira de uma vez e a transforma em uma representação numérica carregada de contexto.

A característica que o define é ser bidirecional: cada token enxerga tanto o que vem antes quanto o que vem depois dele na sequência. Isso o torna bom em compreender e péssimo em gerar, porque gerar exige justamente não conhecer o futuro. Ele é formado por camadas empilhadas de self-attention, e a saída não é texto, são vetores. O BERT é a família encoder-only, e como não pode ser treinado por predição da próxima palavra, usa uma tarefa diferente: esconder palavras aleatórias no meio do texto e pedir que o modelo as adivinhe usando o contexto dos dois lados, o chamado masked word prediction. Isso o especializa em tarefas de compreensão como classificação de texto, análise de sentimento e busca semântica.

O livro cita que a X, antiga Twitter, usa modelos do tipo BERT para detectar conteúdo tóxico em publicações. É uma tarefa de classificar, não de escrever, então o encoder sozinho dá conta.

---

## 11. Decoder (decodificador)

O módulo que faz o caminho inverso do encoder: pega a representação numérica e produz texto de saída, uma palavra por vez.

Ele é a metade geradora do transformer. No desenho original, o decoder recebe duas coisas ao mesmo tempo: os vetores que o encoder produziu a partir do texto de entrada e o texto que ele mesmo já gerou até ali. A cada passo ele emite uma palavra, essa palavra volta a ser entrada no passo seguinte, e o ciclo se repete até a frase terminar. Esse funcionamento de reaproveitar a própria saída como entrada é chamado de autorregressivo, e é ele que dá coerência ao texto, já que cada nova palavra é escolhida em função de toda a sequência que a precede. O processamento é unidirecional, da esquerda para a direita: o decoder nunca enxerga palavras futuras, só o que já veio antes. O GPT abandona o encoder e usa apenas o decoder, apostando na predição da próxima palavra como tarefa única.

Na tradução de "This is an example", com a saída parcial já em "Das ist ein", o decoder usa os vetores do encoder mais essa saída parcial para produzir a última palavra, "Beispiel". No GPT não existe encoder nessa conta: você escreve o começo de uma frase e o decoder vai estendendo, iteração por iteração, cada rodada consumindo o resultado da anterior.

---

## 12. Autoregressive (autorregressivo)

O modelo usa a própria saída anterior como entrada do passo seguinte. Gera um token, anexa esse token à sequência e roda tudo de novo.

Esse laço é o que explica por que o texto aparece palavra por palavra em vez de sair pronto de uma vez. A cada iteração, o modelo recebe a sequência inteira acumulada até ali e produz apenas o próximo token, o que garante coerência: nenhuma palavra é escolhida no vazio, todas são condicionadas ao que já foi escrito. A propriedade tem consequências práticas diretas. Explica o efeito de digitação em streaming das interfaces de chat, explica por que gerar textos longos custa progressivamente mais, já que a entrada cresce a cada passo, e explica por que um erro cometido cedo tende a se propagar, pois vira contexto fixo para todo o resto da geração. É também a razão de o decoder ser unidirecional: se ele pudesse ver palavras futuras durante o treino, a tarefa de prever a próxima palavra ficaria trivial.

Dado "O gato subiu no", o modelo produz "telhado". A entrada do passo seguinte passa a ser "O gato subiu no telhado", e a partir dela ele produz a palavra seguinte, repetindo até atingir um limite ou um token de parada.

---

## 13. LLM (Large Language Model)

Uma rede neural profunda treinada em quantidades enormes de texto para entender, gerar e responder em linguagem humana.

O "large" tem duplo sentido: o tamanho do modelo em número de parâmetros (dezenas ou centenas de bilhões) e o tamanho do dataset, que pode incluir boa parte do texto público disponível na internet. A arquitetura usada é o transformer, que permite ao modelo dar atenção seletiva a diferentes partes da entrada ao fazer previsões. O treinamento acontece em duas fases: pretraining, onde o modelo aprende a prever a próxima palavra em textos brutos sem rótulos (os rótulos saem do próprio texto, por isso é chamado de self-supervised), e fine-tuning, onde ele é refinado com um conjunto menor e rotulado para tarefas específicas. Quando se diz que um LLM "entende" linguagem, o sentido é que ele processa e gera texto de forma coerente e contextualmente relevante, não que tenha consciência ou compreensão humana.

O GPT-3, precursor do modelo original do ChatGPT, é um LLM pré-treinado capaz de completar uma frase iniciada pelo usuário. Um comportamento curioso é que ele consegue traduzir textos mesmo nunca tendo sido treinado explicitamente para traduzir, algo chamado de comportamento emergente.

---

## 14. Self-supervised Learning (aprendizado auto-supervisionado)

O modelo cria os próprios rótulos a partir dos dados, sem nenhuma anotação humana. É o truque que torna o pretraining possível.

Ele fica no meio do caminho entre os dois esquemas clássicos. No aprendizado supervisionado, alguém precisa rotular cada exemplo à mão. No não supervisionado, não existe rótulo nenhum e o algoritmo só busca estrutura. No self-supervised existe rótulo, só que ele é extraído automaticamente do próprio dado. Em um LLM, isso significa esconder a próxima palavra de um trecho e usar a palavra que já estava ali como resposta correta, comparando com o que o modelo previu. Todo texto disponível vira automaticamente um conjunto de treino rotulado, sem custo de anotação, e é exatamente isso que viabiliza treinar sobre trilhões de palavras. O livro se refere a essa tarefa como next-word prediction.

A frase "O sol nasce no leste" sozinha já gera vários pares de treino: dado "O", prever "sol"; dado "O sol", prever "nasce"; dado "O sol nasce", prever "no", e assim por diante. Ninguém precisou escrever rótulo algum.

---

## 15. Pretraining (pré-treinamento)

A primeira e mais cara fase de treinamento de um LLM, feita sobre um volume gigantesco de texto bruto e sem rótulos, para o modelo desenvolver uma noção ampla de linguagem.

O "pre" indica que essa é a etapa inicial, cujo produto é chamado de base model ou foundation model, um modelo genérico que serve de matéria-prima para refinamentos posteriores. O detalhe importante é que aqui não existe rótulo criado por humano: o modelo usa self-supervised learning, ou seja, gera os próprios rótulos a partir do texto de entrada. A tarefa é prever a próxima palavra, e a resposta correta é simplesmente a palavra que já estava ali no texto original. Isso é o que torna viável treinar com trilhões de palavras, já que rotular esse volume manualmente seria impossível. O texto usado é chamado de raw text, dado bruto sem informação de rótulo, embora possa passar por filtragens como remoção de caracteres de formatação ou de documentos em idiomas desconhecidos. O custo é a barreira real dessa etapa: o livro estima o pretraining do GPT-3 em cerca de 4,6 milhões de dólares em computação, e observa que rodar isso em uma única GPU de consumidor levaria centenas de anos. É por isso que a maioria dos projetos parte de um modelo já pré-treinado e vai direto para o fine-tuning.

O GPT-3 foi pré-treinado dessa forma sobre textos da internet, livros, Wikipédia e artigos. O resultado é um modelo que completa uma frase pela metade e já apresenta capacidade de few-shot, aprendendo uma tarefa nova a partir de poucos exemplos dados no próprio prompt, sem treino adicional.

---

## 16. Fine-tuning (ajuste fino)

A segunda fase de treinamento, feita sobre um conjunto de dados bem menor, rotulado e específico, para especializar um modelo já pré-treinado em uma tarefa ou domínio.

Ao contrário do pretraining, aqui os rótulos existem e precisam ser fornecidos, no esquema de aprendizado supervisionado tradicional. O modelo não parte do zero: ele já chega com todos os parâmetros ajustados pelo pretraining e apenas os refina, o que torna o processo muito mais barato em dados e em computação. O livro trata de duas categorias principais. No instruction fine-tuning, o dataset é formado por pares de instrução e resposta, como um pedido de tradução acompanhado da tradução correta, e é o que transforma um modelo que apenas completa texto em um assistente que segue comandos. No classification fine-tuning, o dataset é formado por textos e suas classes correspondentes.

Pegar um LLM já pré-treinado e refiná-lo com um conjunto de e-mails rotulados como "spam" e "não spam" produz um classificador de spam. O mesmo modelo base, refinado com pares de instrução e resposta, viraria um assistente conversacional.

---

## 17. Emergent Behavior (comportamento emergente)

Capacidade que o modelo passa a exibir sem nunca ter sido treinado explicitamente para ela. Não estava no objetivo de treino, apareceu sozinha.

O termo vem da ideia de emergência: propriedades que surgem da combinação de escala, arquitetura e diversidade de dados, e não de uma instrução direta. O objetivo de treino de um LLM é uma única coisa, prever a próxima palavra, e ainda assim o modelo acaba capaz de executar tarefas que nunca constaram como meta. A explicação é que, para prever bem a próxima palavra em um corpus que contém textos traduzidos, código, tabelas e diálogos, o modelo precisa internalizar as estruturas por trás desses textos. Isso traz uma consequência prática incômoda: fica difícil prever de antemão o que um modelo maior vai saber fazer, e igualmente difícil avaliá-lo, já que a lista de habilidades não é conhecida antes do treino terminar.

O exemplo que o livro dá é a tradução: o GPT foi treinado apenas para prever a próxima palavra, sem nenhum dataset dedicado a pares de tradução, e mesmo assim traduz razoavelmente bem entre idiomas.

---

## 18. Zero-shot e Few-shot

Duas formas de pedir uma tarefa ao modelo pelo prompt. Zero-shot é pedir sem dar nenhum exemplo. Few-shot é dar alguns exemplos dentro do próprio prompt antes do pedido real.

O ponto central dos dois é que nada é retreinado: os parâmetros do modelo continuam exatamente iguais, o "aprendizado" acontece apenas dentro da janela de contexto daquela chamada, e some quando a conversa acaba. É por isso que o fenômeno é chamado de in-context learning. A capacidade zero-shot mede o quanto o modelo generaliza para tarefas que nunca viu descritas dessa forma, enquanto o few-shot serve para casos em que o formato de saída importa e é mais fácil demonstrar do que descrever. O contraste com fine-tuning é direto: fine-tuning altera os pesos de forma permanente e exige um dataset, few-shot custa apenas algumas linhas a mais no prompt. O paper que apresentou o GPT-3 se chama justamente "Language Models are Few-Shot Learners".

Zero-shot: "Traduza para o alemão: This is an example". Few-shot: dar antes três pares de frase em inglês e sua tradução em alemão e só então apresentar a frase que você quer traduzir, para o modelo pegar o padrão e o formato de resposta.
