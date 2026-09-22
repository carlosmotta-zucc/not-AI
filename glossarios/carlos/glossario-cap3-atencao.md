# Glossário - Capítulo 3: Coding Attention Mechanisms
---

## 1. Encoder-Decoder RNN (RNN codificador-decodificador)

A arquitetura usada para tarefas como tradução automática antes dos transformers, dividida em dois submódulos: um encoder, que lê a sequência de entrada inteira, e um decoder, que gera a saída a partir do que o encoder produziu.

O encoder processa o texto de entrada passo a passo, atualizando a cada token um estado interno, até condensar a frase inteira num único vetor final. O decoder recebe apenas esse vetor final e, a partir dele, gera a tradução token por token. O ponto fraco dessa arquitetura é justamente esse gargalo: o decoder nunca enxerga os estados intermediários do encoder, só o resumo final, então frases longas ou com dependências distantes perdem informação antes mesmo de a geração começar. Foi essa limitação que motivou o mecanismo de atenção.

O livro usa o exemplo de tradução "Kannst du mir helfen diesen Satz zu übersetzen" para o inglês: uma tradução palavra por palavra produz "Can you me help this sentence to translate", gramaticalmente incorreta, porque certas palavras do inglês dependem de palavras que aparecem em posições diferentes na frase alemã, algo que um RNN codificador-decodificador tem dificuldade de preservar até o fim da sequência.

---

## 2. Hidden State (estado oculto)

O vetor interno que um RNN atualiza a cada passo, tentando acumular toda a informação relevante da sequência processada até ali.

No encoder, o hidden state funciona como uma memória: a cada token de entrada, ele é atualizado e, ao final da sequência, deveria conter o significado da frase inteira. É esse hidden state final, e só ele, que o decoder recebe para começar a gerar a saída. O problema é de capacidade: comprimir uma frase inteira, por mais longa que seja, num único vetor de tamanho fixo é uma tarefa que degrada com o comprimento da entrada, porque a informação mais antiga vai sendo sobrescrita pelas atualizações seguintes.

O livro trata o hidden state final do encoder como equivalente a um vetor de embedding: ele resume a frase alemã inteira num único ponto do espaço vetorial, que o decoder usa como único ponto de partida para produzir a tradução palavra por palavra.

---

## 3. Bahdanau Attention (atenção de Bahdanau)

O mecanismo, proposto em 2014, que modificou o RNN codificador-decodificador para que o decoder deixasse de depender só do hidden state final e passasse a acessar seletivamente diferentes partes da sequência de entrada a cada passo de geração.

A ideia central é que nem todo token de entrada é igualmente relevante para gerar um token de saída específico. Bahdanau resolveu isso dando ao decoder acesso a todos os hidden states do encoder, ponderados por uma importância calculada dinamicamente a cada passo, em vez de só o hidden state final. Essa ponderação por importância é a semente conceitual de tudo o que o capítulo desenvolve depois: três anos mais tarde, pesquisadores descobriram que nem precisavam do RNN, e chegaram ao self-attention e à arquitetura transformer.

No exemplo do livro, ao gerar a segunda palavra da tradução em inglês, o decoder consulta todos os hidden states da frase alemã de entrada, e a espessura da linha que liga cada palavra de entrada à palavra de saída representa o quanto aquele token específico pesou na decisão.

---

## 4. Self-Attention (autoatenção)

O mecanismo em que cada posição de uma sequência calcula sua relevância em relação a todas as outras posições da própria sequência, incluindo ela mesma, para construir uma representação enriquecida de cada elemento.

O prefixo "self" marca a diferença em relação à atenção de Bahdanau: ali a atenção conectava duas sequências diferentes, entrada em alemão e saída em inglês sendo gerada; aqui a atenção conecta a sequência com ela mesma. O objetivo é calcular, para cada token de entrada x(i), um vetor de contexto z(i) que combine informação de todos os outros tokens da sequência, não só do próprio token. Esse vetor de contexto pode ser entendido como um embedding enriquecido: carrega a identidade do token original mais o contexto de tudo ao redor dele. É o componente central dos transformers e a base de todo o resto do capítulo, primeiro numa versão simplificada, sem pesos treináveis, depois numa versão treinável, depois com máscara causal, depois com múltiplas cabeças.

Na frase "Your journey starts with one step", calcular o vetor de contexto z(2) para o token "journey" significa combinar o embedding de "journey" com informação vinda de "Your", "starts", "with", "one" e "step", cada um pesado por sua relevância para "journey" especificamente.

---

## 5. Attention Score (pontuação de atenção)

O valor bruto e não normalizado que mede a afinidade entre dois tokens de uma sequência, calculado como o produto escalar entre seus vetores.

O produto escalar é uma forma compacta de multiplicar dois vetores elemento a elemento e somar os produtos: quanto mais alinhados os dois vetores estiverem no espaço, maior o produto escalar, e maior a pontuação de atenção entre eles. Na versão simplificada do self-attention, sem pesos treináveis, a pontuação é calculada diretamente entre os embeddings de entrada. Na versão treinável, a pontuação passa a ser calculada entre a query de um token e a key de outro, e a matriz completa de pontuações, uma por par de tokens, é o que o livro chama de ω. A pontuação de atenção ainda não é usada diretamente: precisa ser normalizada pelo softmax antes de virar peso de atenção.

Para a query "journey" (x^2) contra os seis tokens de "Your journey starts with one step", a versão sem pesos treináveis produz as pontuações tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]); o segundo valor, 1.4950, é a pontuação do token contra ele mesmo.

---

## 6. Softmax

A função que converte um vetor de números reais quaisquer numa distribuição de probabilidade: todos os valores de saída ficam entre 0 e 1 e a soma deles é exatamente 1.

No self-attention, o softmax é aplicado sobre as pontuações de atenção de um token contra todos os outros, transformando-as em pesos de atenção. É essa propriedade de somar 1 que permite interpretar os pesos como "quanto de cada token entra na combinação final": um peso de 0.3 significa que aquele token contribui com 30% do vetor de contexto resultante. O softmax também tem um efeito colateral importante quando os valores de entrada crescem muito: ele se aproxima de uma função degrau, produzindo gradientes muito pequenos durante o treino; é esse efeito colateral que justifica o escalonamento usado na scaled dot-product attention.

Aplicado às seis pontuações de "journey" (tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])), o softmax devolve seis números positivos que somam 1, com o maior peso caindo sobre a própria posição de "journey", que tinha a maior pontuação da lista.

---

## 7. Attention Weight (peso de atenção)

A pontuação de atenção depois de normalizada pelo softmax: o valor que efetivamente determina o quanto cada token contribui para o vetor de contexto de outro token.

É importante não confundir peso de atenção com peso do modelo (weight parameter). Os pesos do modelo, como as matrizes Wq, Wk e Wv, são parâmetros fixos depois do treino, aprendidos uma vez e reaproveitados para qualquer entrada nova. Os pesos de atenção são dinâmicos: mudam a cada nova sequência de entrada, porque dependem do conteúdo específico daquela sequência, não de algo memorizado permanentemente. A matriz completa de pesos de atenção de uma sequência é quadrada, uma linha e uma coluna por token, e cada linha soma 1.

Para "journey", o peso de atenção calculado com pesos treináveis e escalonamento é tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]); o maior peso, 0.2264, recai sobre o próprio "journey", seguido de 0.2199 para "starts".

---

## 8. Context Vector (vetor de contexto)

O vetor final produzido pelo self-attention para um token: a soma dos vetores de valor de toda a sequência, cada um ponderado pelo respectivo peso de atenção.

É o resultado que todo o mecanismo existe para produzir. Diferente do embedding de entrada, que representa um token isoladamente, o vetor de contexto é um embedding enriquecido: carrega informação de todos os outros tokens da sequência, proporcional à relevância de cada um. Na versão simplificada, o vetor de contexto é a soma ponderada dos próprios embeddings de entrada; na versão treinável, é a soma ponderada dos vetores de valor (value), não dos embeddings originais. É esse vetor de contexto, um por posição da sequência, que segue adiante no pipeline da LLM, primeiro pela máscara causal e depois pelas múltiplas cabeças de atenção.

O vetor de contexto z(2), calculado para "journey" com pesos treináveis, é tensor([0.3061, 0.8210]); rodando o mesmo cálculo para as seis posições da frase, o resultado é uma matriz 6x2, uma linha de vetor de contexto por token de entrada.

---

## 9. Query, Key e Value (consulta, chave e valor)

As três projeções treináveis de cada token de entrada, usadas para calcular pontuações e pesos de atenção e, a partir deles, o vetor de contexto.

Os termos vêm do vocabulário de bancos de dados e sistemas de busca. A query representa o token atual, aquele para o qual se está calculando o vetor de contexto, e funciona como uma pergunta: "o que é relevante para mim?" A key existe para cada token da sequência e serve para ser comparada com a query, como um índice de busca. A value também existe para cada token e representa o conteúdo real que será recuperado, uma vez que o modelo já decidiu, via comparação entre query e keys, quais tokens são mais relevantes. Na prática, o produto escalar entre a query de um token e a key de outro produz a pontuação de atenção entre os dois, e a soma ponderada das values pelos pesos de atenção produz o vetor de contexto.

Para calcular z(2) na frase de exemplo, "journey" fornece a query; todos os seis tokens, incluindo "journey", fornecem suas próprias keys e values, porque a comparação precisa considerar a sequência inteira, não só o token atual.

---

## 10. Weight Matrices Wq, Wk, Wv (matrizes de peso treináveis)

As três matrizes de parâmetros que transformam cada embedding de entrada x(i) nos vetores de query, key e value correspondentes, via multiplicação de matriz.

Cada matriz projeta o vetor de entrada, de dimensão d_in, para um vetor de saída de dimensão d_out: query_i = x_i @ Wq, e o mesmo padrão vale para Wk e Wv. É a existência dessas três matrizes, inicializadas aleatoriamente e ajustadas durante o treino, que separa o self-attention simplificado, sem pesos treináveis, calculado direto sobre os embeddings de entrada, do self-attention usado de fato em LLMs. Nos modelos GPT, d_in e d_out costumam ser iguais; o livro usa valores diferentes (d_in=3, d_out=2) só para deixar mais fácil acompanhar o cálculo passo a passo. Na implementação com nn.Linear em vez de nn.Parameter, essas três matrizes viram três camadas lineares sem bias, o que é matematicamente equivalente a uma multiplicação de matriz e tem inicialização de pesos mais estável.

Com d_in=3 e d_out=2, cada uma das três matrizes Wq, Wk e Wv tem shape (3, 2); projetar os seis tokens de "Your journey starts with one step" produz seis vetores de query, seis de key e seis de value, cada um com duas posições.

---

## 11. Scaled Dot-Product Attention (atenção de produto escalar escalonado)

O nome técnico do self-attention usado em LLMs: as pontuações de atenção são calculadas por produto escalar entre query e key, e depois divididas pela raiz quadrada da dimensão das keys antes do softmax.

O escalonamento existe por uma razão específica de treino, não conceitual: quando a dimensão do embedding é grande, nos LLMs estilo GPT costuma passar de mil, produtos escalares tendem a ter magnitude alta, e valores altos empurram o softmax para perto de uma função degrau, o que produz gradientes próximos de zero durante a retropropagação e trava o aprendizado. Dividir pela raiz quadrada de d_k mantém as pontuações numa faixa que preserva gradientes úteis. É por esse escalonamento, e só por ele, que o mecanismo ganha o nome scaled dot-product attention: a fórmula é a mesma do self-attention treinável, só com essa normalização extra antes do softmax.

No cálculo de "journey", a pontuação bruta contra "journey" é 1.8524; dividida pela raiz de d_k=2 (aproximadamente 1.4142) e depois passada pelo softmax junto com as outras cinco pontuações, ela vira o peso de atenção 0.2264.

---

## 12. Causal Attention (atenção causal)

Uma forma especializada de self-attention, também chamada de masked attention, em que cada token só pode considerar tokens anteriores e ele mesmo, nunca tokens futuros da sequência.

É a modificação necessária para tarefas de geração de texto: como o modelo precisa prever a próxima palavra a partir apenas do que já foi escrito, permitir que ele veja tokens futuros durante o treino tornaria a tarefa trivial e inútil para geração, porque a resposta certa estaria disponível como entrada. A diferença em relação ao self-attention padrão, que dá acesso à sequência inteira de uma vez, é exatamente essa restrição temporal: para o token na posição i, a atenção causal zera as pontuações de atenção para todas as posições depois de i, antes mesmo de normalizar pelo softmax.

Na matriz de pesos de atenção da frase de exemplo, a atenção causal impede que o token "journey" (posição 2) tenha qualquer peso não nulo sobre "starts", "with", "one" ou "step", que vêm depois dele na sequência; "journey" só pode atender a si mesmo e a "Your".

---

## 13. Causal Mask (máscara causal)

A matriz triangular usada para implementar a atenção causal: zera as pontuações ou pesos de atenção acima da diagonal principal, correspondentes a posições futuras.

Existem duas formas equivalentes de aplicar essa máscara. A primeira, mais intuitiva, calcula o softmax normalmente sobre todas as pontuações, multiplica o resultado por uma máscara triangular de 0s e 1s (torch.tril), zerando os pesos futuros, e depois renormaliza cada linha para voltar a somar 1. A segunda, mais eficiente e usada na implementação final, preenche as pontuações futuras com menos infinito antes do softmax (torch.triu, com masked_fill_): como e elevado a menos infinito é zero, o softmax já produz zero diretamente nessas posições, sem precisar de uma etapa extra de renormalização. As duas formas chegam ao mesmo resultado numérico; a segunda só evita o passo redundante de dividir e depois corrigir.

Para uma sequência de seis tokens, a máscara triangular inferior usada na primeira abordagem é uma matriz 6x6 de 1s abaixo da diagonal, incluindo a diagonal, e 0s acima dela; aplicada aos pesos de atenção de "starts" (posição 3), ela zera qualquer peso sobre "with", "one" e "step", preservando só os pesos sobre "Your", "journey" e o próprio "starts".

---

## 14. Dropout (na atenção)

Uma técnica de regularização em que unidades escolhidas aleatoriamente são zeradas durante o treino, para impedir que o modelo dependa demais de um subconjunto específico de conexões.

No mecanismo de atenção dos transformers, o dropout costuma ser aplicado logo depois de calcular os pesos de atenção, antes de multiplicá-los pelos vetores de value. Ele só é ativo durante o treino; na inferência, fica desligado e todos os pesos são usados normalmente. Um detalhe importante: quando uma fração dos pesos é zerada, os pesos restantes são multiplicados por 1 dividido por (1 menos a taxa de dropout), para compensar a redução e manter a soma total da atenção equivalente à que existiria sem o dropout. O livro usa uma taxa de 50% só para fins didáticos, deixando claro que valores como 0.1 ou 0.2 são mais realistas para o treino de uma LLM de verdade.

Aplicando dropout de 50% a uma matriz 6x6 de 1s, o resultado tem aproximadamente metade das posições zeradas e a outra metade com valor 2 (1 dividido por 0.5), em vez de 1.

---

## 15. Multi-Head Attention (atenção multi-cabeça)

A execução de várias instâncias do mecanismo de atenção causal em paralelo, cada uma com sua própria projeção aprendida de query, key e value, cujos resultados são combinados no final.

A ideia é que uma única cabeça de atenção só consegue capturar um tipo de relação entre os tokens de cada vez; rodando várias cabeças em paralelo, cada uma com suas próprias matrizes Wq, Wk e Wv, o modelo pode atender simultaneamente a aspectos diferentes da mesma sequência, por exemplo relações sintáticas numa cabeça e relações semânticas noutra, embora o livro não force nenhuma especialização explícita, ela surge do treino. É essa capacidade de atender a múltiplos subespaços de representação ao mesmo tempo que dá ao mecanismo de atenção dos transformers boa parte do seu poder.

O menor modelo GPT-2 usa 12 cabeças de atenção com embedding de contexto de 768 posições; o maior modelo GPT-2 usa 25 cabeças com embedding de 1600 posições, mostrando que o número de cabeças escala junto com o tamanho do modelo.

---

## 16. Attention Head e head_dim (cabeça de atenção)

Cada instância paralela de atenção causal dentro de um módulo multi-cabeça, responsável por uma fatia da dimensão total de saída, chamada head_dim.

A dimensão de saída total, d_out, é dividida igualmente entre o número de cabeças, num_heads: head_dim = d_out / num_heads, e por isso d_out precisa ser divisível por num_heads. Cada cabeça produz seu próprio vetor de contexto de tamanho head_dim, e os vetores de todas as cabeças são concatenados ao final para reconstituir a dimensão d_out original. Isso significa que aumentar o número de cabeças sem aumentar d_out não aumenta a capacidade total do modelo, só reparte a mesma capacidade em fatias mais finas; para manter a capacidade por cabeça, d_out também precisa crescer junto com num_heads.

Com d_out=4 e num_heads=2, cada cabeça trabalha com head_dim=2: duas cabeças produzem dois vetores de contexto de duas posições cada, que concatenados formam o vetor de contexto final de quatro posições por token.

---

## 17. MultiHeadAttentionWrapper vs. MultiHeadAttention (empilhamento vs. divisão de pesos)

Duas formas equivalentes em resultado, mas diferentes em eficiência, de implementar atenção multi-cabeça: empilhar módulos de atenção causal independentes, ou usar um único conjunto maior de matrizes e depois dividir o resultado entre as cabeças.

Na primeira abordagem, o MultiHeadAttentionWrapper, cada cabeça é um objeto CausalAttention separado, com suas próprias matrizes Wq, Wk e Wv de tamanho d_out por cabeça, e o forward roda cada cabeça sequencialmente, concatenando as saídas no final. Isso significa repetir a multiplicação de matriz mais cara do mecanismo, a projeção de query, key e value, uma vez por cabeça. Na segunda abordagem, a classe MultiHeadAttention, existe uma única matriz Wq (e o mesmo para Wk e Wv) de tamanho d_out total; a projeção é feita uma única vez para a sequência inteira, e só depois o resultado é remodelado (view) e transposto para separar as cabeças, calcular a atenção em lote para todas elas de uma vez, e depois recombinar. O resultado numérico final é idêntico ao da primeira abordagem, mas com muito menos multiplicações de matriz, o que é o motivo de ser essa a versão usada de fato na LLM do projeto.

Com num_heads=2 e CausalAttention de d_out=2 por cabeça, o MultiHeadAttentionWrapper produz um vetor de contexto final de 4 posições (2x2, concatenado); a classe MultiHeadAttention equivalente, para produzir a mesma saída de 4 posições, usaria d_out=4 já dividido internamente em duas cabeças de head_dim=2, chegando ao mesmo resultado com uma fração das multiplicações de matriz.

---

## 18. Output Projection Layer (camada de projeção de saída)

Uma camada linear adicional, aplicada depois de concatenar as saídas de todas as cabeças de atenção, que mistura a informação das diferentes cabeças antes de o vetor de contexto seguir adiante na rede.

Sem essa camada, o vetor de contexto final seria simplesmente a concatenação bruta das saídas de cada cabeça, sem nenhuma interação entre elas depois da divisão. A camada de projeção de saída (self.out_proj, uma nn.Linear de d_out para d_out) permite que o modelo aprenda a recombinar essas fatias de forma útil. O livro é direto ao dizer que essa camada não é estritamente necessária para o mecanismo de atenção funcionar, mas é praticamente padrão nas arquiteturas de LLM em uso, o que justifica incluí-la mesmo assim.

Na classe MultiHeadAttention, depois que os vetores de contexto de todas as cabeças são recombinados de volta ao shape (batch, num_tokens, d_out), esse tensor ainda passa por self.out_proj antes de ser devolvido pelo forward: uma multiplicação de matriz quadrada, d_out por d_out, que não altera a dimensão, só recombina o conteúdo.
