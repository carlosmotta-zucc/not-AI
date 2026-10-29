# Glossário - Capítulo 4: Implementing a GPT Model from Scratch to Generate Text
---

## 1. GPT Configuration (configuração do GPT)

O dicionário de hiperparâmetros que define o tamanho e o comportamento de um modelo GPT antes de qualquer peso existir: vocabulário, contexto, largura do embedding, número de cabeças, número de blocos, taxa de dropout e presença de bias nas projeções de query, key e value.

É o ponto de partida do capítulo inteiro, porque toda a arquitetura é montada lendo essa configuração: a tabela de tokens usa vocab_size e emb_dim, a tabela de posições usa context_length e emb_dim, cada bloco usa n_heads e drop_rate, e o laço que empilha os blocos usa n_layers. A consequência prática é que o código do modelo não muda entre o GPT-2 small e o GPT-2 XL; só a configuração muda. Isso também deixa explícito que os componentes dos capítulos 2 e 3, embeddings e atenção multi-cabeça, passam a ser peças parametrizadas de um sistema maior.

O livro usa a GPT_CONFIG_124M: vocab_size 50257, context_length 1024, emb_dim 768, n_heads 12, n_layers 12, drop_rate 0.1 e qkv_bias False. Trocando só emb_dim, n_layers e n_heads para 1600, 48 e 25, o mesmo código monta o GPT-2 XL, com 1 637 792 000 parâmetros.

---

## 2. Logits

Os valores brutos, não normalizados, que o modelo produz na última camada, um por token do vocabulário, para cada posição da sequência de entrada.

Cada logit mede o quanto o modelo favorece um token como próximo da sequência naquela posição, mas ainda não é uma probabilidade: pode ser negativo, não tem limite e a linha não soma 1. Para virar distribuição de probabilidade, os logits passam pelo softmax, a mesma função usada no capítulo 3 para transformar pontuações de atenção em pesos. O shape dos logits é sempre (batch, num_tokens, vocab_size), ou seja, o modelo devolve uma previsão de próximo token para cada posição, não só para a última; o treino do capítulo 5 usa todas elas, e a geração de texto usa só a última.

No lote de dois textos do livro, "Every effort moves you" e "Every day holds a", com Token IDs [6109, 3626, 6100, 345] e [6109, 1110, 6622, 257], a saída do modelo tem shape (2, 4, 50257): dois textos, quatro posições, 50 257 logits por posição. O primeiro logit da primeira posição do primeiro texto, no GPTModel com seed 123, é 0.3613.

---

## 3. Vanishing Gradient (gradiente que desaparece)

O problema de treino em redes profundas em que o gradiente fica cada vez menor à medida que é propagado de volta da saída até as primeiras camadas, a ponto de essas camadas quase não aprenderem.

Na retropropagação, o gradiente de uma camada é obtido multiplicando os gradientes de todas as camadas que vêm depois dela, pela regra da cadeia. Se esses fatores forem pequenos, o produto encolhe exponencialmente com a profundidade; se forem grandes, cresce sem controle, e o problema simétrico se chama exploding gradient. Os dois tornam o treino instável ou lento. O capítulo 4 apresenta duas defesas contra isso, que aparecem juntas em todo bloco transformer: a layer normalization, que mantém as ativações numa escala estável, e as shortcut connections, que criam um caminho direto para o gradiente.

Na rede de cinco camadas do livro, com entrada [1., 0., -1.] e sem atalhos, o gradiente médio absoluto dos pesos cai de 0.00505 na última camada para 0.00020 na primeira, 25 vezes menor: a camada mais próxima da entrada praticamente não é atualizada.

---

## 4. Layer Normalization (normalização de camada)

A operação que ajusta as ativações de cada vetor para que tenham média 0 e variância 1 ao longo da dimensão do embedding, calculando média e variância separadamente para cada token.

O objetivo é estabilizar o treino: com as ativações sempre na mesma escala, os gradientes não explodem nem desaparecem por causa de valores que crescem ou encolhem de camada em camada. A normalização é feita na última dimensão (dim=-1), isto é, sobre os 768 valores de um único token, e não sobre o lote. Essa é a diferença em relação à batch normalization, que calcula estatísticas sobre os exemplos do lote: a layer normalization não depende do tamanho do batch, o que dá mais flexibilidade e estabilidade em treino distribuído e em ambientes com recursos limitados. No GPT, ela aparece duas vezes dentro de cada bloco transformer e uma vez no final, depois do último bloco. Na classe LayerNorm, a conta é (x - média) / sqrt(variância + eps), com eps = 1e-5 para evitar divisão por zero.

No exemplo do livro, a primeira linha da saída de uma camada Linear seguida de ReLU é [0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000], com média 0.1324 e variância 0.0231. Normalizada à mão, vira [0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719], com média 0 e variância 1. Essa conta manual do livro não usa eps e usa a variância padrão do PyTorch, com n - 1; a classe LayerNorm usa eps e divide por n.

---

## 5. Scale and Shift (escala e deslocamento)

Os dois vetores de parâmetros treináveis da layer normalization, do tamanho de emb_dim, que multiplicam e somam ao resultado normalizado: saída = scale * x_normalizado + shift.

Forçar média 0 e variância 1 em todas as camadas pode ser restritivo demais, porque nem sempre essa é a escala mais útil para a camada seguinte. Scale e shift devolvem ao modelo a liberdade de aprender, durante o treino, a escala e o centro mais adequados para cada dimensão. Eles começam neutros, scale com 1s e shift com 0s, então no início do treino a layer normalization faz só a normalização pura. Por terem só emb_dim valores cada, quase não pesam no total de parâmetros.

Com emb_dim 768, cada LayerNorm tem 768 valores de scale e 768 de shift, 1 536 parâmetros. O GPT-2 small tem 25 delas, duas por bloco nos 12 blocos mais a norma final, somando 38 400 parâmetros num modelo de 163 milhões.

---

## 6. Biased Variance (variância enviesada)

A variância calculada dividindo a soma dos quadrados dos desvios por n, o número de elementos, em vez de n - 1, que é a correção de Bessel usada na variância amostral.

A implementação de LayerNorm do livro usa unbiased=False por compatibilidade com as camadas de normalização do GPT-2, implementado originalmente em TensorFlow, cujo comportamento padrão é dividir por n. Para emb_dim grande a diferença entre dividir por n ou por n - 1 é desprezível, mas ela existe, e importa quando o objetivo é carregar pesos pré-treinados e reproduzir exatamente as saídas originais, que é o que o capítulo 5 faz com os pesos da OpenAI.

Com emb_dim 768, usar n - 1 em vez de n multiplicaria cada saída da LayerNorm por sqrt(767/768), aproximadamente 0.99935. No projeto, a LayerNorm com variância enviesada bate com a nn.LayerNorm do PyTorch com diferença máxima de 2.4e-07.

---

## 7. GELU (Gaussian Error Linear Unit)

A função de ativação usada no GPT-2, uma curva suave que se comporta como identidade para valores positivos grandes, tende a zero para negativos grandes e, entre esses extremos, deixa passar valores negativos pequenos.

A GELU é definida como x multiplicado pela função de distribuição acumulada da normal padrão; o GPT-2 usa uma aproximação com tangente hiperbólica: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))). A comparação natural é com a ReLU, que zera tudo que é negativo e tem uma quina em x = 0. A suavidade da GELU dá gradiente não nulo para quase todos os negativos, então neurônios que recebem entrada negativa continuam contribuindo para o aprendizado, ainda que menos que os de entrada positiva; na ReLU, a saída para qualquer entrada negativa é zero. Ela é a não linearidade dentro da feed forward network de cada bloco.

Em x = -1, a ReLU devolve 0.0000 e a GELU devolve -0.1588; em x = 1, a ReLU devolve 1.0000 e a GELU 0.8412. O mínimo da GELU fica perto de x = -0.75, onde ela vale -0.1700: é o único ponto negativo em que a derivada é zero.

---

## 8. Feed Forward Network (rede feed forward)

O pequeno módulo de duas camadas lineares com uma GELU entre elas, aplicado em cada bloco transformer depois da atenção, que expande a dimensão do embedding, aplica a não linearidade e contrai de volta.

A atenção mistura informação entre tokens diferentes; a feed forward network faz o contrário: processa cada posição de forma independente, com os mesmos pesos para todas as posições. A expansão para uma dimensão quatro vezes maior dá ao modelo um espaço mais rico para transformar a representação de cada token antes de devolvê-la ao tamanho original, que precisa ser mantido para que os blocos possam ser empilhados. No GPT-2 small, é o componente com mais parâmetros: 34.8% do total, contra 23.7% da tabela de tokens, 23.7% da cabeça de saída e 17.4% da atenção.

No GPT-2 small, a entrada (2, 3, 768) passa pela primeira Linear e vira (2, 3, 3072), passa pela GELU mantendo (2, 3, 3072) e volta a (2, 3, 768) na segunda Linear. São 2 · 768 · 3072 + 3072 + 768 = 4 722 432 parâmetros por bloco, 2.00 vezes os 2 360 064 da atenção multi-cabeça do mesmo bloco.

---

## 9. Shortcut Connection (conexão de atalho)

Também chamada de skip connection ou residual connection: a ligação que soma a entrada de uma camada à saída dela, x + f(x), criando um caminho alternativo que pula a transformação.

O efeito principal é sobre o gradiente: como a derivada de x + f(x) em relação a x tem sempre um termo igual a 1 vindo da soma, o gradiente consegue chegar às camadas iniciais sem ser multiplicado por todos os fatores pequenos das camadas intermediárias. É a solução direta para o vanishing gradient, introduzida originalmente nas redes residuais de visão computacional. A condição para funcionar é que entrada e saída tenham o mesmo shape, outro motivo para a atenção e a feed forward network preservarem emb_dim. No bloco transformer há duas conexões de atalho, uma em volta da atenção e outra em volta da feed forward network.

Na mesma rede de cinco camadas do livro, com atalhos, o gradiente médio absoluto da primeira camada sobe de 0.00020 para 0.22170, cerca de 1 100 vezes maior, e a razão entre a última e a primeira camada cai de 25 para 6.

---

## 10. Transformer Block (bloco transformer)

A unidade que se repete n_layers vezes no GPT, combinando atenção multi-cabeça mascarada, feed forward network, duas layer normalizations, dropout e duas conexões de atalho.

O fluxo dentro do bloco é: LayerNorm, atenção multi-cabeça causal, dropout e soma com a entrada; depois, LayerNorm, feed forward network, dropout e nova soma com o resultado anterior. A atenção permite que cada token incorpore informação dos tokens anteriores; a feed forward network transforma cada token individualmente com essa informação já incorporada. Como entrada e saída têm exatamente o mesmo shape, os blocos podem ser encadeados sem nenhum ajuste entre eles, e cada bloco refina a representação que recebeu do anterior. Todo o capítulo 3 converge aqui: a classe MultiHeadAttention vira uma das duas subcamadas do bloco.

No GPT-2 small, uma entrada (2, 4, 768) sai do bloco com shape (2, 4, 768). Cada bloco tem 7 085 568 parâmetros: 2 360 064 na atenção, 4 722 432 na feed forward network e 3 072 nas duas LayerNorms.

---

## 11. Pre-LayerNorm (normalização antes da subcamada)

A disposição em que a layer normalization é aplicada antes da atenção e antes da feed forward network, dentro de cada ramo do atalho, em vez de depois da soma do atalho.

A arquitetura transformer original usava Post-LayerNorm: primeiro a subcamada e a soma do atalho, depois a normalização. O GPT-2 e os LLMs posteriores passaram a normalizar antes, o que deixa o caminho do atalho limpo, sem nenhuma normalização atravessada nele, e melhora a dinâmica de treino em redes profundas. A consequência é que o fluxo residual, a soma acumulada que atravessa os blocos, nunca é normalizado dentro deles; por isso o GPT precisa de uma LayerNorm final antes da cabeça de saída.

No forward do bloco, a sequência é x = x + drop_shortcut(att(norm1(x))) e depois x = x + drop_shortcut(ff(norm2(x))): norm1 e norm2 só afetam o que entra na subcamada, e o x somado é o original.

---

## 12. Residual Stream (fluxo residual)

O tensor de shape (batch, num_tokens, emb_dim) que atravessa toda a pilha de blocos, ao qual cada subcamada soma sua contribuição pelas conexões de atalho.

O termo não aparece no livro; é o nome usado na literatura para o que as shortcut connections e o Pre-LayerNorm produzem juntos: em vez de cada bloco substituir a representação anterior, cada bloco acrescenta um ajuste a ela. A representação final de um token é a soma do embedding inicial, token mais posição, com todas as contribuições da atenção e das feed forward networks dos n_layers blocos. Como nada normaliza essa soma no caminho, sua escala cresce com a profundidade, e a LayerNorm final existe para corrigir isso antes da projeção para o vocabulário.

No modelo base do projeto, com emb_dim 256, o desvio padrão do fluxo residual na saída do último bloco sobe de 1.425 com 1 bloco para 1.610 com 12 blocos, enquanto o desvio dos logits, medido depois da LayerNorm final e da cabeça, fica em 0.577 em todos os casos.

---

## 13. GPTModel

A classe que monta o modelo completo: tabela de embeddings de tokens, tabela de embeddings posicionais, dropout, a pilha de n_layers blocos transformer, a LayerNorm final e a cabeça de saída.

No forward, os Token IDs viram embeddings de token, somados aos embeddings posicionais exatamente como no capítulo 2; o resultado passa pelo dropout, atravessa os blocos em sequência, é normalizado pela LayerNorm final e projetado pela cabeça de saída em logits do tamanho do vocabulário. É o ponto em que os três capítulos anteriores se conectam: tokenização e embeddings do capítulo 2, atenção do capítulo 3 e os componentes deste capítulo. Sem treino, os pesos são aleatórios, então o modelo já faz todas as contas certas, mas ainda não prevê nada útil.

Com torch.manual_seed(123) e a GPT_CONFIG_124M, o lote "Every effort moves you" e "Every day holds a" sai com logits de shape (2, 4, 50257). O livro só calcula a perda no capítulo 5; nos experimentos do projeto, o mesmo modelo com seed 123, sobre o lote do The Verdict, tem perda inicial de 10.975, perto de ln(50257) ≈ 10.825, a perda de uma distribuição uniforme sobre o vocabulário: o modelo ainda não sabe nada sobre linguagem.

---

## 14. Output Head (cabeça de saída)

A camada linear final, sem bias, que projeta cada vetor de emb_dim dimensões no espaço do vocabulário, produzindo um logit por token possível.

É o caminho inverso da tabela de embeddings de tokens: a tabela leva um Token ID para um vetor de 768 dimensões; a cabeça de saída leva um vetor de 768 dimensões para 50 257 pontuações, uma por Token ID. Ela recebe a saída da LayerNorm final e é a única parte do modelo que conhece o tamanho do vocabulário além da tabela de entrada. Junto com a tabela de embeddings, é um dos componentes mais pesados do modelo, porque sua matriz cresce com vocab_size, não com a profundidade.

No GPT-2 small, a cabeça de saída é uma nn.Linear(768, 50257, bias=False), com matriz de pesos de shape (50257, 768) e 38 597 376 parâmetros, 23.7% do modelo inteiro.

---

## 15. Weight Tying (compartilhamento de pesos)

A técnica de reutilizar a mesma matriz de pesos na tabela de embeddings de tokens e na cabeça de saída, em vez de manter duas matrizes independentes.

A ideia é possível porque as duas matrizes têm o mesmo shape, (vocab_size, emb_dim), e fazem operações inversas entre Token IDs e vetores. O GPT-2 original usa weight tying, e é por isso que é chamado de modelo de 124 milhões de parâmetros, embora a implementação do livro, com as duas matrizes separadas, tenha 163 milhões. O livro mantém as matrizes separadas porque, na experiência do autor, camadas separadas dão treino e desempenho melhores, o que também vale para LLMs modernos, e mostra o weight tying apenas na contagem; no capítulo 5, ao carregar os pesos da OpenAI, a mesma matriz é copiada para as duas posições.

O GPTModel do livro tem 163 009 536 parâmetros; subtraindo os 38 597 376 da cabeça de saída, que seriam compartilhados com a tabela de tokens, chega-se a 124 412 160. Em float32, 4 bytes por parâmetro, os 163 milhões ocupam 621.83 MB.

---

## 16. Autoregressive Generation (geração autoregressiva)

O processo de gerar texto um token por vez, em que cada token previsto é anexado à sequência de entrada e a sequência aumentada é usada para prever o próximo.

O GPT só sabe prever o próximo token; texto longo surge da repetição desse passo. Em cada iteração, o modelo recebe todos os Token IDs até ali, produz logits para todas as posições, e só os logits da última posição são usados, porque são os que preveem o token seguinte à sequência inteira. O token escolhido é concatenado à entrada e o laço recomeça, até atingir max_new_tokens. É a razão de ser da atenção causal do capítulo 3: no treino, cada posição aprende a prever seu próximo token vendo só o passado, exatamente a situação da geração.

O livro parte de "Hello, I am", Token IDs [15496, 11, 314, 716]. Na figura 4.18, que ilustra um modelo já treinado, a primeira iteração prevê o token 257, "a". No código, com o modelo sem treino, a primeira iteração prevê o token 27018, a sequência passa a [15496, 11, 314, 716, 27018], e depois de 6 iterações o resultado é "Hello, I am Featureiman Byeswickattribute argue": as contas estão certas, mas os pesos ainda são aleatórios.

---

## 17. Greedy Decoding (decodificação gulosa)

A estratégia de escolher, em cada passo da geração, sempre o token com o maior logit, sem nenhuma aleatoriedade.

A escolha é feita com torch.argmax na última posição. O livro aplica o softmax antes do argmax para deixar claro que os logits correspondem a probabilidades, mas observa que esse passo é redundante: o softmax é monotônico, então o token de maior logit é sempre o de maior probabilidade. A decodificação gulosa é determinística, a mesma entrada sempre gera a mesma saída; o capítulo 5 troca o argmax por amostragem, com temperatura e top-k, para dar variedade ao texto gerado.

Nos logits de exemplo do capítulo 5 para o token seguinte a "every effort moves you", o maior logit é 6.75, na posição de "forward"; a decodificação gulosa escolhe "forward" sempre, enquanto a amostragem com temperatura 1 escolhe "forward" em 582 de 1000 sorteios e "toward" em 343.

---

## 18. Context Window Cropping (corte da janela de contexto)

A operação, feita a cada passo da geração, de manter só os últimos context_length tokens da sequência antes de passá-la ao modelo.

O modelo tem uma tabela de embeddings posicionais de tamanho fixo, context_length linhas, então não consegue processar sequências mais longas do que isso: não existe embedding para a posição 1025 num modelo com contexto 1024. Na geração autoregressiva a sequência cresce um token por iteração, e em algum momento ultrapassaria esse limite. O corte idx[:, -context_size:] descarta os tokens mais antigos e mantém a janela mais recente, com o custo de o modelo deixar de ver o início do texto.

Num modelo com contexto 1024 que já gerou uma sequência de 1030 tokens, a próxima iteração passa ao modelo só os tokens das posições 7 a 1030; os 6 primeiros deixam de influenciar a previsão.
