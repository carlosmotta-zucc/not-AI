# Glossário - Capítulo 5: Pretraining on Unlabeled Data
---

## 1. Pretraining on Unlabeled Data (pré-treinamento em dados não rotulados)

O treino em que o modelo aprende a prever o próximo token de um texto bruto, sem nenhuma anotação humana: o rótulo de cada posição é o token que vem logo depois dela no próprio texto.

É o estágio que transforma o GPTModel do capítulo 4, com pesos aleatórios, num modelo que produz texto com alguma estrutura. Ele usa tudo o que veio antes sem mudar nada: os pares entrada-alvo deslocados de uma posição do capítulo 2, a atenção causal do capítulo 3, que impede cada posição de enxergar o próprio alvo, e os logits do capítulo 4. O que o capítulo 5 acrescenta são as três peças que faltavam: um número que mede a qualidade da previsão, um laço que ajusta os pesos para melhorar esse número, e maneiras de gerar texto com o modelo treinado e de guardá-lo em arquivo. Para o treino caber num laptop, o livro reduz o context_length da GPT_CONFIG_124M de 1024 para 256.

O corpus do livro é o conto The Verdict, com 20 479 caracteres e 5 145 tokens. Com 90% para treino, ele rende 9 lotes de 2 sequências de 256 tokens, e os 10% restantes formam 1 lote de validação. Em 10 épocas sobre esse texto, a perda de treino do livro cai de 9.781 para 0.391.

---

## 2. Target Tokens (tokens-alvo)

Os Token IDs que o modelo deveria prever: a própria sequência de entrada deslocada uma posição para a frente.

Cada posição da entrada tem o seu alvo, então uma sequência de T tokens fornece T exemplos de treino de uma vez. É por isso que o treino aproveita os logits de todas as posições, enquanto a geração do capítulo 4 usa só os da última. Os alvos saem do mesmo DataLoader de janela deslizante do capítulo 2 e não exigem trabalho humano, o que é o sentido de "unlabeled" no título do capítulo: o texto rotula a si mesmo.

No exemplo do livro, as entradas são [[16833, 3626, 6100], [40, 1107, 588]], "every effort moves" e "I really like", e os alvos são [[3626, 6100, 345], [1107, 588, 11311]], " effort moves you" e " really like chocolate". Sem treino, o argmax dos logits da primeira sequência decodifica " Armed heNetflix" no lugar de " effort moves you".

---

## 3. Target Probability (probabilidade do token-alvo)

A probabilidade que o softmax dos logits atribui, em cada posição, ao token que de fato vem a seguir no texto.

É a única coluna da distribuição que entra no cálculo da perda: treinar o modelo é fazer esse número subir, e como o softmax soma 1, ele só sobe tirando probabilidade dos outros tokens. Num modelo sem treino a distribuição é quase uniforme, e a probabilidade do alvo fica perto de 1/50257, cerca de 0.00002. O livro a extrai indexando o tensor de probabilidades com os Token IDs dos alvos, probas[text_idx, [0, 1, 2], targets[text_idx]].

Nas duas sequências do exemplo do livro, as probabilidades dos alvos são [7.4541e-05, 3.1061e-05, 1.1563e-05] e [1.0337e-05, 5.6776e-05, 4.7559e-06]: todas da ordem de 1/50257, algumas um pouco acima e outras um pouco abaixo do chute uniforme.

---

## 4. Average Log Probability (log-probabilidade média)

A média dos logaritmos das probabilidades dos tokens-alvo; com o sinal trocado, é a perda que o treino minimiza.

O logaritmo resolve um problema prático: a probabilidade de um texto inteiro é o produto das probabilidades de cada token, e um produto de números da ordem de 0.00001 some na precisão do float. Com o log, o produto vira soma. Como o log de uma probabilidade é sempre negativo ou zero, o objetivo de levar a probabilidade do alvo a 1 equivale a levar a log-probabilidade média a 0; por convenção, em vez de maximizar um número negativo, o treino minimiza o seu oposto, a negative average log probability. O log também pune o erro confiante: errar com probabilidade 0.5 custa 0.69, errar com 0.0001 custa 9.2.

No exemplo do livro, os logs das seis probabilidades dos alvos são [-9.5042, -10.3796, -11.3677, -11.4798, -9.7764, -12.2561]. A média é -10.7940 e a perda, 10.7940.

---

## 5. Cross-Entropy Loss (perda de entropia cruzada)

A medida da diferença entre a distribuição prevista pelo modelo e a distribuição verdadeira, que para um alvo único se reduz à log-probabilidade negativa média do token-alvo.

É a função de perda do capítulo e o ponto de partida da retropropagação. A função cross_entropy do PyTorch faz em uma chamada os passos que o livro abre um a um: softmax dos logits, seleção da probabilidade do alvo, logaritmo, média e troca de sinal. Ela espera uma linha de logits por exemplo, então os tensores são achatados antes: os logits de (batch, tokens, vocabulário) para (batch · tokens, vocabulário), e os alvos de (batch, tokens) para um vetor. Cada posição de cada sequência vira uma classificação independente entre 50 257 classes.

No exemplo do livro, os logits de shape (2, 3, 50257) viram (6, 50257), os alvos viram um vetor de 6 Token IDs, e a cross_entropy devolve 10.7940, o mesmo valor calculado à mão. No modelo base do projeto, sem treino, a perda de validação é 10.992, perto de ln(50257) ≈ 10.825, a perda de quem distribui a probabilidade por igual.

---

## 6. Perplexity (perplexidade)

A exponencial da perda de entropia cruzada, lida como o número efetivo de tokens entre os quais o modelo hesita a cada passo.

Ela carrega a mesma informação da perda, numa escala mais fácil de interpretar: perplexidade 1 é certeza no token correto, e perplexidade igual ao tamanho do vocabulário é a dúvida de quem sorteia uniformemente. É calculada com torch.exp(loss) e só vale para a perda média por token com logaritmo natural. Ela não diz se o token certo seria o escolhido, só quanta probabilidade ele recebeu.

A perda de 10.7940 do exemplo do livro dá perplexidade 48 725.8: o modelo sem treino hesita entre praticamente o vocabulário inteiro. No treino de referência do projeto, a perplexidade de validação cai de 54 888 na primeira avaliação, logo depois do primeiro passo, para 464 no fim da décima época.

---

## 7. Training and Validation Sets (conjuntos de treino e de validação)

As duas partes em que o texto é dividido: a de treino, usada para ajustar os pesos, e a de validação, guardada para medir a perda em texto que o modelo nunca viu.

A perda de treino sozinha não distingue um modelo que aprendeu a língua de um que decorou o texto; só a perda de validação mostra se o que foi aprendido serve fora dos dados de treino. O livro corta o texto com train_ratio 0.90 e monta dois DataLoaders: o de treino com shuffle e drop_last ligados, o de validação com os dois desligados. As funções calc_loss_batch e calc_loss_loader calculam a perda de um lote e a média sobre os lotes de um DataLoader, e são usadas tanto no laço de treino quanto na avaliação.

Com o The Verdict, o livro obtém 9 lotes de treino e 1 de validação, e as perdas do modelo sem treino são 10.98758347829183 no treino e 10.98110580444336 na validação: quase iguais, porque um modelo que não aprendeu nada trata os dois textos do mesmo jeito. No modelo base do projeto, com contexto 128, são 36 janelas de treino e 4 de validação.

---

## 8. Backpropagation (retropropagação)

O cálculo, pela regra da cadeia e da saída para a entrada, do gradiente da perda em relação a cada peso do modelo.

É o que liga a perda aos pesos: o gradiente diz, para cada peso, em que direção e com que intensidade a perda muda se ele mudar. No PyTorch é uma chamada, loss.backward(), que percorre o grafo de operações montado durante a passada para a frente. Os gradientes se acumulam em vez de serem substituídos, e por isso o laço de treino chama optimizer.zero_grad() antes de cada lote. É aqui que as defesas do capítulo 4 mostram para que servem: as shortcut connections e a layer normalization existem para o gradiente chegar com tamanho útil às primeiras camadas.

No modelo base do projeto, uma chamada de backward calcula 28 920 832 derivadas, uma por parâmetro, a um custo de cerca de duas passadas para a frente. O teste do treino confere que todos os tensores de parâmetros recebem gradiente e mudam depois de uma época.

---

## 9. AdamW

O otimizador usado no capítulo: uma variante do Adam em que o weight decay é aplicado diretamente aos pesos, separado do cálculo do gradiente.

O otimizador é quem transforma gradiente em pesos novos, na chamada optimizer.step(). O Adam guarda, para cada peso, duas médias móveis, a do gradiente e a do quadrado do gradiente, e usa a razão entre elas para dar a cada peso um passo de tamanho parecido, qualquer que seja a escala do seu gradiente. Isso importa num GPT, em que a linha de embedding de um token raro recebe gradiente poucas vezes e a cabeça de saída recebe em toda posição. O livro cria o otimizador com torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1). Essas médias fazem parte do estado do treino, e é por isso que o capítulo salva o otimizador junto com o modelo.

As duas médias custam memória: no projeto, o estado de treino tem pesos, gradientes e as duas médias, 16 bytes por parâmetro em float32, o que dá 441 MB para o modelo base de 28.9 milhões de parâmetros e 2 478 MB para o modelo do livro.

---

## 10. Learning Rate (taxa de aprendizado)

O hiperparâmetro que define o tamanho do passo dado pelo otimizador em cada atualização dos pesos.

Pequena demais, o modelo quase não sai do lugar no número de passos disponível; grande demais, os passos atravessam a região de perda baixa em vez de descer nela. No AdamW ela é, aproximadamente, o quanto cada peso anda por passo. O livro usa 0.0004 do começo ao fim do treino; aquecimento e decaimento da taxa ficam para o apêndice D.

Nos experimentos do projeto, com 10 épocas e tudo o mais igual, a taxa 0.00001 termina com perda de validação 10.374, quase a do início; a do livro, 0.0004, chega a 6.135; a de 0.001 chega a 6.118 já na sexta época, e depois piora; a de 0.004 não passa de 6.473.

---

## 11. Weight Decay (decaimento de pesos)

A regularização que encolhe todos os pesos um pouco a cada passo do otimizador, para desencorajar valores grandes.

A ideia é limitar a complexidade do modelo: pesos menores tendem a produzir funções mais suaves, menos ajustadas aos detalhes do conjunto de treino. No AdamW o encolhimento é um fator aplicado direto ao peso, 1 - lr · weight_decay, fora da conta do gradiente; no Adam com penalidade L2, o termo entraria no gradiente e seria normalizado junto com ele. É, com o dropout, uma das duas regularizações presentes no treino do capítulo.

Com lr 0.0004 e weight_decay 0.1, o fator por passo é 0.99996, e em 180 passos os pesos encolhem 0.7% por causa dele. Nos experimentos do projeto, weight decay 0, 0.1 e 1.0 dão perdas de validação mínimas de 6.138, 6.135 e 6.126, diferenças menores que a variação entre seeds: num treino tão curto ele quase não age.

---

## 12. Epoch, Batch and Step (época, lote e passo)

As três unidades de contagem do treino: o lote é o grupo de sequências processado de uma vez, o passo é uma atualização dos pesos feita a partir de um lote, e a época é uma passada completa por todos os lotes do conjunto de treino.

O número de passos é o que de fato mede quanto o modelo foi treinado: épocas · lotes por época. Com o tamanho do lote fixo, mais épocas significam mais passos sobre os mesmos dados, e num corpus pequeno isso leva a decorar o texto. O livro registra o passo global em global_step e os tokens processados em tokens_seen, e avalia as perdas a cada eval_freq passos.

No treino do livro são 9 lotes por época e 10 épocas, 90 passos, numerados de 0 a 89; o log vai de "Ep 1 (Step 000000)" a "Ep 10 (Step 000085)", uma linha a cada 5 passos. No treino de referência do projeto são 18 lotes por época e 180 passos, e na avaliação do passo 175 o modelo já processou 45 056 tokens, quase dez vezes o conto.

---

## 13. Training Loop (laço de treino)

A sequência de operações repetida a cada lote para ajustar os pesos: zerar os gradientes, calcular a perda, retropropagar e dar o passo do otimizador.

É o centro do capítulo, implementado na função train_model_simple. Dois laços aninhados percorrem as épocas e os lotes; dentro deles ficam as quatro operações que treinam e duas que só observam: a cada eval_freq passos, as perdas de treino e de validação são medidas, e no fim de cada época o modelo gera um trecho de texto a partir do mesmo contexto, para acompanhar a evolução a olho. O modelo fica em model.train() durante os passos, com o dropout ligado.

No livro, a amostra do fim da primeira época é "Every effort moves you,,,,,,,,,,,,." e a da décima é "Every effort moves you know," was one of the axioms he laid down across the Sevres and silver of an exquisitely appointed luncheon-table...". O laço do projeto, train_model, reproduz esse treino em CPU: as perdas impressas e as amostras conferem em 45 de 45 comparações.

---

## 14. Evaluation Mode (modo de avaliação)

A configuração em que o modelo é medido ou usado, e não treinado: model.eval() desliga o dropout, e torch.no_grad() desliga o registro das operações para o cálculo de gradiente.

São dois interruptores diferentes. O eval muda o comportamento do modelo: com o dropout ligado, a mesma entrada daria saídas diferentes a cada chamada, e a perda de validação mudaria de uma medida para a outra. O no_grad muda só o custo: sem ele, cada avaliação guardaria as ativações para um backward que nunca vai acontecer. A função evaluate_model do livro aplica os dois, mede as perdas e devolve o modelo ao modo de treino com model.train(). A geração de texto usa o mesmo par.

No projeto, a decodificação gulosa de 30 tokens rodada duas vezes com o modelo em train() dá textos diferentes em 28 das 30 posições; com o modelo em eval(), as duas execuções são idênticas.

---

## 15. Overfitting (sobreajuste)

A situação em que o modelo continua melhorando nos dados de treino enquanto para de melhorar, ou piora, nos dados que não viu.

Aparece como um afastamento entre as duas curvas de perda: a de treino segue caindo e a de validação estaciona. Com um corpus muito pequeno e várias épocas, o sobreajuste chega à memorização, em que o modelo reproduz trechos do texto de treino palavra por palavra. O livro aponta esse efeito no próprio treino e observa que ele seria reduzido com muito mais dados; decodificar com sorteio em vez de argmax diminui a recitação literal, mas não a causa.

No livro, a perda de treino termina em 0.391 e a de validação em 6.452, e o texto gerado contém "quite insensible to the irony", uma frase do conto. No projeto, o mesmo treino levado a 30 épocas chega a perda de treino 0.043 e validação 7.067, pior que a mínima de 6.106 alcançada na época 11, e o modelo recita 32 de 32 tokens dos trechos de treino.

---

## 16. Decoding Strategy (estratégia de decodificação)

A regra que transforma a distribuição de probabilidades do próximo token em um token escolhido.

O modelo só produz a distribuição; o que se faz com ela é uma decisão separada, tomada na hora de gerar e sem nenhum efeito sobre os pesos. O capítulo 4 usa a mais simples, a gulosa, que pega sempre o token mais provável e por isso gera sempre o mesmo texto. O capítulo 5 apresenta as alternativas com sorteio, temperatura e top-k, e as reúne numa função generate. A estratégia só redistribui a probabilidade que o treino produziu: ela decide o quanto do que o modelo sabe, e do que ele não sabe, aparece no texto.

No livro, depois do treino, a decodificação gulosa a partir de "Every effort moves you" gera "know," was one of the axioms he laid down across the Sevres and silver of an exquisitely appointed lun", um trecho do conto; com top_k 25 e temperatura 1.4, o texto deixa de ser cópia.

---

## 17. Multinomial Sampling (amostragem multinomial)

A escolha do próximo token por sorteio, em que cada token tem chance de ser escolhido igual à sua probabilidade.

Substitui o torch.argmax por torch.multinomial(probas, num_samples=1). O token mais provável continua saindo na maioria das vezes, mas não sempre, o que dá variedade ao texto e faz duas gerações com o mesmo contexto serem diferentes. O custo é que tokens improváveis também saem, na proporção da probabilidade que têm, e num vocabulário de 50 257 tokens a soma de muitas probabilidades pequenas pode ser grande. O sorteio depende do gerador de números aleatórios: com a mesma seed, sai o mesmo texto.

No vocabulário de 9 palavras do livro, com logits [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79], o argmax escolhe sempre "forward". Em 1 000 sorteios com seed 123 saem 582 vezes "forward", 343 "toward", 73 "closer" e 2 "inches".

---

## 18. Temperature Scaling (escala de temperatura)

A divisão dos logits por um número positivo, a temperatura, antes do softmax, para controlar o quanto a distribuição fica concentrada ou espalhada.

Temperatura abaixo de 1 amplia as diferenças entre os logits e concentra a probabilidade nos tokens mais prováveis, aproximando a amostragem da escolha gulosa; acima de 1, reduz as diferenças e aproxima a distribuição da uniforme; igual a 1, não muda nada. Ela muda a forma da distribuição, nunca a ordem dos tokens, e por isso não altera o argmax. Só faz sentido combinada com a amostragem multinomial. Na função generate do livro, temperatura 0 é o código para usar o argmax.

No vocabulário de 9 palavras, "forward" tem probabilidade 0.57 com temperatura 1, 0.99 com 0.1 e 0.24 com 5. Com temperatura 5, "pizza", que tem um dos menores logits, passa a ter probabilidade 0.043 e é sorteada 32 vezes em 1 000: a frase "every effort moves you pizza" deixa de ser impossível.

---

## 19. Top-k Sampling (amostragem top-k)

A restrição do sorteio aos k tokens de maior logit, com todos os outros levados a probabilidade zero.

É implementada mascarando os logits fora dos k maiores com menos infinito antes do softmax, o mesmo truque da máscara causal do capítulo 3: a exponencial de menos infinito é zero, e os k restantes já saem somando 1. O top-k corta o que a temperatura só atenua: tokens que o modelo considera improváveis deixam de poder ser sorteados, qualquer que seja a temperatura. Os dois são usados juntos, o top-k para delimitar os candidatos e a temperatura para repartir a probabilidade entre eles.

Com k = 3 no vocabulário de 9 palavras, sobram os logits 6.75, 6.28 e 4.51, de "forward", "toward" e "closer"; depois do softmax, as probabilidades são 0.5775, 0.3610 e 0.0615, e as outras seis palavras ficam com 0.

---

## 20. state_dict

O dicionário do PyTorch que associa o nome de cada camada aos seus tensores de parâmetros; é a forma recomendada de salvar um modelo.

Salvar o state_dict com torch.save(model.state_dict(), "model.pth") guarda só os números, não o código da classe. Para usar o arquivo é preciso primeiro criar um modelo com a mesma arquitetura e então copiar os pesos para dentro dele com model.load_state_dict(torch.load("model.pth", map_location=device)), seguido de model.eval() para a inferência. O map_location diz em que dispositivo os tensores devem ser carregados, independentemente de onde foram salvos. O arquivo não diz de que configuração os pesos saíram: quem o carrega precisa saber.

No livro, o modelo é recriado com GPTModel(GPT_CONFIG_124M) antes de receber os pesos; se a configuração usada fosse outra, com outro emb_dim ou outro número de blocos, os shapes não bateriam e o load_state_dict falharia.

---

## 21. Checkpoint (ponto de salvamento)

O arquivo que guarda o estado de um treino num certo momento, com os pesos do modelo e o estado do otimizador, para que o treino possa ser retomado depois.

Para só gerar texto, os pesos bastam. Para continuar treinando, não: o AdamW guarda para cada peso as médias móveis que usa para ajustar o passo, e sem elas o otimizador recomeça do zero e os passos seguintes saem diferentes dos de um treino que não parou. O livro salva os dois num dicionário só, com as chaves model_state_dict e optimizer_state_dict, e na carga recria o modelo e o otimizador antes de chamar load_state_dict em cada um. O exercício 5.4 pede exatamente isso: carregar modelo e otimizador numa sessão nova e treinar mais uma época.

No livro o arquivo se chama model_and_optimizer.pth.

---

## 22. Pretrained Weights (pesos pré-treinados)

Os pesos de um modelo já treinado por outra pessoa ou instituição, carregados numa arquitetura compatível em vez de serem obtidos por treino próprio.

Treinar o GPT-2 exige recursos fora do alcance de um laptop, e a OpenAI publicou os pesos; o capítulo termina carregando-os no GPTModel do livro. Isso só funciona porque a arquitetura implementada é a mesma, e exige duas mudanças na configuração: context_length volta a 1024 e qkv_bias passa a True, porque o GPT-2 original usa bias nas projeções de query, key e value. A função load_weights_into_gpt copia tensor por tensor, e a auxiliar assign confere se os shapes coincidem antes de cada cópia. A matriz de embeddings de tokens da OpenAI é copiada também para a cabeça de saída, que é o weight tying do capítulo 4. É o ponto de partida do fine-tuning dos capítulos 6 e 7.

O arquivo de configuração do modelo menor traz n_vocab 50257, n_ctx 1024, n_embd 768, n_head 12 e n_layer 12, e a matriz wte tem shape (50257, 768). Os quatro tamanhos publicados são 124M, 355M, 774M e 1558M.
