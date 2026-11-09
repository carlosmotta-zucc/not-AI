# Glossário — Capítulo 5: Pretraining on Unlabeled Data

João Antônio Sarturi Popp — Inteligência Artificial e Sistemas Inteligentes — UNOESC

---

## 1. Pretraining (no cap. 5)

**Equivalente:** Pré-treinamento.

**Definição:** Já tinha aparecido no cap. 1 como o primeiro estágio, o treino em texto bruto. Aqui ele vira código: pré-treinar é mudar os pesos do modelo pra minimizar a loss de treino (termo 2). O "unlabeled" do título quer dizer que ninguém rotula nada: a resposta certa é sempre o próximo token do próprio texto.

**Função:** É o estágio 2 da figura 5.1 (passo 4), que inclui o código de treino (passo 5), a avaliação do modelo (passo 6) e o salvar/carregar pesos (passo 7). É o que transforma o `GPTModel` do cap. 4, que só gerava lixo, num modelo que escreve texto com gramática.

**Relação:** Os dados vêm do pipeline do cap. 2 (`create_dataloader_v1`, pares entrada-alvo deslocados de uma posição) e a arquitetura é o `GPTModel` do cap. 4 sem mudança nenhuma. Na configuração, a única mudança é o `context_length`, que cai de 1.024 pra 256 pra dar pra treinar num notebook. Do lado do `drop_rate` o livro anota que é possível e comum usar dropout 0, mas mantém 0.1. Como pré-treinar de verdade é caro, o capítulo termina carregando os pesos que a OpenAI já treinou (termo 19).

**Exemplo:** Quadro de custo do livro: o Llama 2 de 7 bilhões de parâmetros levou 184.320 horas de GPU A100 e processou 2 trilhões de tokens. Com um servidor 8 × A100 na AWS a uns US$ 30 por hora, dá 184.320 ÷ 8 × 30 = US$ 691.200, que o livro arredonda pra cerca de US$ 690.000. O nosso, com o "The Verdict", treina em uns 5 minutos.

---

## 2. Loss Function

**Equivalente:** Função de perda (o livro também chama de *text generation loss*, perda de geração de texto).

**Definição:** Função que calcula a diferença entre a saída prevista pelo modelo e a saída desejada, ou seja, o quanto as previsões estão longe dos alvos. Aqui a saída prevista são as probabilidades que o modelo deu pros tokens-alvo.

**Função:** É o número que diz se o treino está funcionando. Serve pra duas coisas: avaliar a qualidade do texto gerado com um número, em vez de só olhar e achar "coerente", e ser o ponto de partida do backpropagation (termo 3), que usa a loss pra ajustar os pesos.

**Relação:** O objetivo do treino é aumentar a probabilidade do softmax justamente na posição do token-alvo de cada vetor de logits (figura 5.6). A loss usada no capítulo é a cross entropy (termo 4). Os alvos são as entradas deslocadas uma posição pra frente, os mesmos pares entrada-alvo do cap. 2.

**Exemplo:** `inputs = [[16833, 3626, 6100], [40, 1107, 588]]` ("every effort moves" / "I really like") e `targets = [[3626, 6100, 345], [1107, 588, 11311]]` (" effort moves you" / " really like chocolate"). Sem treino, o argmax da saída do primeiro texto decodifica "Armed heNetflix" em vez de "effort moves you".

---

## 3. Backpropagation

**Equivalente:** Retropropagação.

**Definição:** Técnica padrão de treino de redes neurais profundas. É o processo que atualiza os pesos pra que o modelo dê valores maiores nos IDs de token que a gente quer gerar.

**Função:** É o mecanismo que de fato faz o modelo aprender. Precisa de uma função de perda (termo 2) pra saber o quanto as previsões estão erradas.

**Relação:** No código aparece como `loss.backward()` dentro do loop de treino (termo 9). Observação minha: a rigor o `backward` só calcula os gradientes, e quem mexe nos pesos é o otimizador com `optimizer.step()` (termo 12); o livro chama o processo todo de backpropagation. É o mesmo gradiente do vanishing gradient do cap. 4: LayerNorm e shortcut existem pra ele chegar nas primeiras camadas.

**Exemplo:** O livro manda pros apêndices A.3 a A.7 pros detalhes. Na prática já tinha aparecido no cap. 4, no `loss.backward()` da `ExampleDeepNeuralNetwork` com `nn.MSELoss()`, usado só pra medir os gradientes.

---

## 4. Cross Entropy Loss

**Equivalente:** Perda de entropia cruzada (o livro trata como sinônimo de *negative average log probability*, média negativa da log-probabilidade).

**Definição:** Medida da diferença entre duas distribuições de probabilidade: a verdadeira (os tokens do dataset) e a prevista pelo modelo. Pra tokens, ela é a média negativa da log-probabilidade dos alvos. A figura 5.7 faz isso em 6 passos: logits → probabilidades (softmax) → probabilidades dos alvos → logaritmo → média → troca de sinal. Montando em fórmula (montagem minha), com N tokens no batch e p o softmax dos logits:

$$
\text{loss} = -\frac{1}{N}\sum_{i=1}^{N} \ln p_i(\text{alvo}_i)
$$

**Função:** É a loss do treino de LLM. Como probabilidade fica entre 0 e 1, o log é sempre menor ou igual a 0 e só chega em 0 quando a probabilidade é 1 (observação minha). O objetivo seria levar a log-probabilidade média até 0, mas em deep learning o costume é levar o negativo dela até 0, por isso o sinal trocado. O livro só diz que é mais fácil otimizar com o log do que com as probabilidades direto e deixa o porquê pra um apêndice.

**Relação:** O `torch.nn.functional.cross_entropy` faz os 6 passos sozinho, por isso recebe os logits crus (cap. 4) e não probabilidades. Exige os tensores achatados (termo 7).

**Exemplo:** As probabilidades dos alvos do texto 1 são `[7.4541e-05, 3.1061e-05, 1.1563e-05]`, perto de 1/50.257 ≈ 0,00002 porque os pesos estão aleatórios. O log das 6 probabilidades dá `[-9.5042, -10.3796, -11.3677, -11.4798, -9.7764, -12.2561]`, a média dá −10,7940 e a loss fica 10,7940, o mesmo valor do `cross_entropy`. Conta minha: com a mesma probabilidade pra todos os tokens, a loss seria $\ln 50257 \approx 10{,}82$, então uma loss na faixa de 10,8 a 11 é a de um modelo que ainda está chutando.

---

## 5. Perplexity

**Equivalente:** Perplexidade.

**Definição:** Medida usada junto com a cross entropy pra avaliar modelos de linguagem. Mede o quanto a distribuição prevista pelo modelo bate com a distribuição real das palavras no dataset. É a exponencial da loss:

$$
\text{perplexidade} = e^{\text{loss}}
$$

**Função:** Dá uma leitura mais interpretável da incerteza do modelo: é o tamanho efetivo do vocabulário entre o qual o modelo está em dúvida a cada passo. Como na loss, quanto menor, melhor.

**Relação:** Sai direto da cross entropy (termo 4) com `torch.exp(loss)`, então não muda nada no treino, é só outra forma de ler o mesmo número. Ela não aparece no `train_model_simple`; o livro só mostra num quadro (observação minha).

**Exemplo:** Com a loss de 10,7940, `torch.exp(loss)` dá `tensor(48725.8203)`: o modelo estaria em dúvida entre uns 48.725 tokens, praticamente o vocabulário inteiro. Conta minha com o fim do treino de 10 épocas: treino $e^{0{,}391} \approx 1{,}48$ e validação $e^{6{,}452} \approx 634$.

---

## 6. Training Set e Validation Set

**Equivalente:** Conjunto de treino e conjunto de validação.

**Definição:** Divisão do texto em duas partes: 90% pra treinar (`train_ratio = 0.90`) e os 10% restantes só pra avaliar o modelo durante o treino.

**Função:** A loss de validação mostra se o modelo está aprendendo algo que vale fora do texto de treino ou só decorando. É comparando as duas que se percebe o overfitting (termo 13). Os pesos só são atualizados com o conjunto de treino.

**Relação:** Cada parte vira um data loader com o `create_dataloader_v1` do cap. 2, com `max_length` e `stride` iguais ao `context_length` (256), então as janelas não se sobrepõem. O de treino usa `shuffle=True` e `drop_last=True`; o de validação, `shuffle=False` e `drop_last=False`.

**Exemplo:** O "The Verdict" tem 20.479 caracteres e 5.145 tokens. A divisão é feita no texto, por caractere (`split_idx = int(train_ratio * len(text_data))`). Com batch size 2 saem 9 batches de treino `[2, 256]` e só 1 de validação. O livro comenta que, em LLM de verdade, batch de 1.024 ou mais não é raro.

---

## 7. calc_loss_batch e calc_loss_loader

**Equivalente:** N/A — nome próprio (funções do livro; em português, "calcular loss do batch" e "calcular loss do loader").

**Definição:** A `calc_loss_batch` calcula a cross entropy de um batch: manda entrada e alvo pro `device`, roda o modelo e chama `cross_entropy(logits.flatten(0, 1), target_batch.flatten())`. O `flatten` junta dimensões: os logits `[batch, tokens, vocab_size]` viram `[batch × tokens, vocab_size]` e os alvos viram `[batch × tokens]`, que é o formato que o `cross_entropy` espera. A `calc_loss_loader` chama a primeira pra cada batch de um data loader, soma com `loss.item()` e divide pelo número de batches.

**Função:** A loss da `calc_loss_batch` é a que vai pro `loss.backward()` no treino. A `calc_loss_loader` mede o desempenho no conjunto inteiro (treino ou validação); o `num_batches` deixa usar só alguns batches, pra avaliar mais rápido no meio do treino.

**Relação:** O `.to(device)` coloca os dados no mesmo dispositivo do modelo. Com `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, o mesmo código treina na GPU se tiver uma com CUDA. Se `num_batches` passar do total do loader, o `min` corta; loader vazio devolve `nan`. Observação minha: ao explicar o shape `[2, 3, 50257]` das probabilidades, o livro diz que o último número é a "dimensão de embedding, determinada pelo tamanho do vocabulário". Isso está impreciso: 50.257 é o tamanho do vocabulário (a saída do `out_head`), e a dimensão de embedding é 768.

**Exemplo:** No exemplo pequeno do livro, logits `[2, 3, 50257]` e alvos `[2, 3]` viram `[6, 50257]` e `[6]`: 2 textos × 3 tokens = 6 previsões. Com o modelo sem treino, dentro de `torch.no_grad()`: `Training loss: 10.98758347829183` e `Validation loss: 10.98110580444336`. Se o modelo aprendesse a gerar os próximos tokens como aparecem no texto, a loss iria pra perto de 0.

---

## 8. torch.no_grad()

**Equivalente:** Sem gradiente (desliga o *gradient tracking*, rastreamento de gradiente).

**Definição:** Bloco `with torch.no_grad():` que desliga o rastreamento de gradiente no código de dentro. Observação minha: o rastreamento é o PyTorch ir guardando as operações feitas com os parâmetros pra conseguir calcular os gradientes no `backward`.

**Função:** Economizar computação quando não vai ter `backward`. Aparece na avaliação, na geração e no cálculo das losses do modelo antes de treinar. Os comentários do livro dizem exatamente isso: não é necessário fora do treino e só aumenta o custo.

**Relação:** No cap. 4 eu tinha deixado como observação minha que essa linha desliga o gradiente; agora o livro confirma. Não confundir com `model.eval()` (termo 11): o `no_grad` desliga o gradiente e o `eval` desliga o dropout. Na avaliação durante o treino os dois aparecem juntos.

**Exemplo:** `with torch.no_grad(): train_loss = calc_loss_loader(train_loader, model, device)`. Dentro do loop de treino ele não aparece, porque ali a loss precisa do gradiente pro `loss.backward()`.

---

## 9. Training Loop

**Equivalente:** Laço (loop) de treinamento; época (*epoch*) é uma volta completa dele.

**Definição:** O procedimento padrão de treino de redes neurais no PyTorch (figura 5.11), em 8 passos: (1) percorrer as épocas; (2) percorrer os batches de cada época; (3) zerar os gradientes da iteração anterior; (4) calcular a loss do batch atual; (5) backward pra calcular os gradientes; (6) atualizar os pesos com os gradientes; (7) imprimir as losses de treino e validação; (8) gerar um texto de exemplo. O 7 e o 8 são opcionais, só pra acompanhar o progresso. Uma época é uma passada completa pelo conjunto de treino, e o número de batches por época é o tamanho do conjunto dividido pelo tamanho do batch.

**Função:** É a parte que efetivamente treina. Os passos 3, 5 e 6 são três linhas: `optimizer.zero_grad()`, `loss.backward()` e `optimizer.step()`. O livro reforça que não tem nada de especial pra LLM: é o loop comum de deep learning, com cross entropy e AdamW.

**Relação:** O `backward` é o backpropagation (termo 3) e o `step` aplica a regra do AdamW (termo 12). Observação minha: o `zero_grad` precisa vir antes porque o PyTorch soma os gradientes novos em cima dos antigos; sem ele, cada batch carregaria os gradientes dos anteriores. Muitas épocas num dataset pequeno levam o modelo a decorar o texto (termo 13); o livro comenta que o comum é treinar num dataset bem maior por uma época só. Técnicas mais avançadas (learning rate warmup, cosine annealing, gradient clipping) ficam pro apêndice D.

**Exemplo:** Pra cada batch, nessa ordem: `optimizer.zero_grad()` → `loss = calc_loss_batch(...)` → `loss.backward()` → `optimizer.step()`. Conta minha: 9 batches por época × 10 épocas = 90 atualizações; o `global_step` vai de 0 a 89, e o último impresso é o 85 porque a avaliação só roda de 5 em 5.

---

## 10. train_model_simple

**Equivalente:** N/A — nome próprio (função do livro; em português, "treinar modelo simples").

**Definição:** A função principal de pré-treino (listagem 5.3), que implementa a figura 5.11 e devolve as listas `train_losses`, `val_losses` e `track_tokens_seen`. Usa duas auxiliares: a `evaluate_model`, que calcula a loss de treino e de validação usando só `eval_iter` batches, e a `generate_and_print_sample`, que converte o texto inicial com `text_to_token_ids`, gera 50 tokens com o `generate_text_simple` e imprime com `token_ids_to_text`.

**Função:** Coloca o modelo em `model.train()` no começo de cada época e faz a atualização de pesos de cada batch. A cada `eval_freq` passos (`global_step % eval_freq == 0`) chama a `evaluate_model`, que dá a estimativa numérica do progresso; no fim de cada época chama a `generate_and_print_sample`, que dá um exemplo concreto de texto. Também conta os tokens vistos com `input_batch.numel()`.

**Relação:** Usa a `calc_loss_batch` (termo 7). As duas auxiliares começam com `model.eval()` + `torch.no_grad()` (termos 8 e 11) e terminam voltando pro `train()`. O `global_step` começa em −1 pra que o primeiro batch seja o passo 0 e já entre na avaliação. O `context_size` da geração vem do próprio modelo, `model.pos_emb.weight.shape[0]`. Com `eval_iter=5`, no treino entram 5 dos 9 batches, e na validação o `min` reduz pra 1 (conta minha). O exercício 5.4 manda rodar essa função de novo depois de carregar o checkpoint (termo 18).

**Exemplo:** `train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=10, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)` leva uns 5 minutos num MacBook Air. A saída intercala as duas auxiliares: `Ep 2 (Step 000015): Train loss 5.961, Val loss 6.616` e depois `Every effort moves you, and, and, and, ...`. Conta minha: 2 × 256 = 512 tokens por batch, 4.608 por época e 46.080 no total.

---

## 11. model.train() e model.eval()

**Equivalente:** Modo de treino e modo de avaliação.

**Definição:** Os dois modos do modelo no PyTorch. O `model.eval()` desliga componentes aleatórios usados só no treino, como o dropout; o `model.train()` liga de novo.

**Função:** Na avaliação o dropout fica desligado pra ter resultados estáveis e reproduzíveis; no treino ele precisa estar ligado pra regularizar.

**Relação:** Mexe nas três camadas de dropout do `GPTModel` (cap. 4). Não desliga o gradiente, isso é o `torch.no_grad()` (termo 8). Depois de carregar pesos salvos (termo 18), usa-se `eval()` pra gerar texto e `train()` pra continuar treinando.

**Exemplo:** Depois do treino, antes de testar as estratégias de geração, o livro faz `model.to("cpu")` (inferência com modelo pequeno não precisa de GPU) e `model.eval()`.

---

## 12. AdamW

**Equivalente:** N/A — nome próprio (otimizador; variante do Adam). Junto com ele: *weight decay* (decaimento de pesos) e *learning rate* (taxa de aprendizado).

**Definição:** O otimizador usado no treino. Os otimizadores Adam são escolha popular pra redes profundas, e o AdamW é uma variante que melhora a forma de aplicar o weight decay, técnica que penaliza pesos grandes pra reduzir a complexidade do modelo e evitar overfitting. Isso dá uma regularização mais eficaz e melhor generalização, por isso ele é muito usado em LLM.

**Função:** É quem executa o `optimizer.step()`: recebe os gradientes e decide quanto mexer em cada peso. É adaptativo: guarda dados históricos de cada parâmetro e usa pra ajustar dinamicamente a learning rate de cada um. Observação minha: o livro usa o `lr` sem definir; ele é o tamanho do passo de cada atualização (grande demais deixa o treino instável, pequeno demais deixa lento), e no AdamW é o valor base que o otimizador ajusta.

**Relação:** Recebe `model.parameters()`, que devolve todos os pesos treináveis do modelo. O weight decay tem o mesmo objetivo do dropout, mas age nos pesos e não nas ativações (observação minha); mesmo com ele, o modelo decorou o texto (termo 13), porque o dataset é pequeno demais. Como o AdamW guarda estado extra, precisa ser salvo junto com o modelo pra continuar o treino (termo 18). Conferi no PyTorch: o estado de cada parâmetro é `step`, `exp_avg` e `exp_avg_sq`, os dois últimos do tamanho do próprio parâmetro.

**Exemplo:** `torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)`. Detalhe que achei testando (observação minha): na seção 5.4, ao recarregar o checkpoint, o livro cria o otimizador com `lr=5e-4`, mas o `optimizer.load_state_dict(...)` sobrescreve os hiperparâmetros com os salvos. Rodei no PyTorch: o lr é 0,0005 antes do load e volta pra 0,0004 depois, então o `5e-4` ali não tem efeito.

---

## 13. Overfitting

**Equivalente:** Sobreajuste (no capítulo, aparece como memorização do texto de treino).

**Definição:** Quando o modelo se ajusta demais aos dados de treino e passa a decorá-los em vez de generalizar. No gráfico, a loss de treino continua caindo enquanto a de validação para de cair e fica bem maior.

**Função:** Não é componente, é o problema que a validação serve pra detectar. Explica por que o modelo treinado gera trechos copiados palavra por palavra do texto.

**Relação:** Aparece comparando treino e validação (termo 6). Dropout e weight decay (termo 12) servem pra reduzir, mas aqui a causa é dataset muito pequeno treinado por várias épocas. As estratégias de decodificação (termos 14 a 16) atacam o efeito na geração, deixando o texto mais original. O livro diz que, treinando nos 60.000 livros do Project Gutenberg, isso não acontece.

**Exemplo:** Figura 5.12: as duas losses caem juntas no começo e passam a divergir depois da segunda época. No fim: treino 0,391 e validação 6,452. Dá pra confirmar a memorização procurando dentro do "The Verdict" o trecho "quite insensible to the irony", que o modelo gerou.

---

## 14. Probabilistic Sampling

**Equivalente:** Amostragem probabilística (o livro chama essas técnicas de *decoding strategies*, estratégias de decodificação).

**Definição:** Trocar o `argmax` (decodificação gulosa, cap. 4) por um sorteio do próximo token proporcional à probabilidade de cada um, com `torch.multinomial(probas, num_samples=1)`.

**Função:** Não entra no treino, só na geração. Com `argmax` o modelo sempre gera o mesmo texto pro mesmo início, e depois do treino esse texto é justamente o trecho decorado. Sorteando, o token mais provável sai na maioria das vezes, mas não sempre.

**Relação:** É a base da temperatura (termo 15) e do top-k (termo 16). Usa as mesmas probabilidades do softmax, só muda o critério de escolha.

**Exemplo:** Vocabulário de 9 palavras, contexto "every effort moves you" e logits `[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]`. O argmax dá sempre "forward". Sorteando 1.000 vezes: 582 × forward, 343 × toward, 73 × closer e 2 × inches.

---

## 15. Temperature Scaling

**Equivalente:** Escalonamento por temperatura.

**Definição:** Dividir os logits por um número T maior que 0 antes do softmax:

$$
p_i = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}
$$

**Função:** Controla o quanto a distribuição fica espalhada. T maior que 1 deixa as probabilidades mais uniformes (mais variedade, mas mais texto sem sentido); T menor que 1 deixa mais concentrada no token mais provável; T = 1 é o mesmo que não usar.

**Relação:** Só faz sentido com amostragem (termo 14): com T muito pequeno, o `multinomial` escolhe quase sempre o mais provável e fica parecido com o `argmax`. No `generate` (termo 17), T = 0 virou o código pra "usar argmax".

**Exemplo:** Exercício 5.1 (resposta minha). Rodei o `print_sampled_tokens` com seed 123; com T = 1 bateu exatamente com o livro (582 / 343 / 73 / 2), então confio nos outros. "pizza" sai 0 vez com T = 1, 0 com T = 0,1 e 32 com T = 5. O jeito mais rápido e exato é nem sortear: ler direto a probabilidade do softmax com temperatura. Pra "pizza" dá 0,0001 com T = 1, praticamente 0 com T = 0,1 e 0,043 com T = 5, que são os cerca de 4% que o livro cita.

---

## 16. Top-k Sampling

**Equivalente:** Amostragem top-k (só entre os k mais prováveis).

**Definição:** Restringir o sorteio aos k tokens de maior logit. Os outros recebem −inf antes do softmax, então ficam com probabilidade 0, e as que sobram somam 1.

**Função:** Corrige o defeito da temperatura alta: mantém a variedade, mas impede que tokens muito improváveis, tipo "pizza", sejam sorteados e estraguem a frase.

**Relação:** É o mesmo truque da máscara com −inf da atenção causal do cap. 3. Em código, `torch.topk` pega os k maiores e `torch.where(logits < top_logits[-1], -inf, logits)` mascara o resto. No `generate` (termo 17) é combinado com temperatura e `multinomial`. Exercício 5.2 (resposta minha): temperatura e top-k baixos quando precisa de resposta consistente e previsível (pergunta factual, código); altos quando a variedade importa mais (texto criativo, brainstorm).

**Exemplo:** Com k = 3: top logits `[6.75, 6.28, 4.51]` nas posições `[3, 7, 0]` (forward, toward, closer). Depois da máscara e do softmax: `[0.0615, 0, 0, 0.5775, 0, 0, 0, 0.3610, 0]`.

---

## 17. generate

**Equivalente:** N/A — nome próprio (função do livro; em português, "gerar").

**Definição:** Versão nova do `generate_text_simple` (listagem 5.4), com três parâmetros a mais: `temperature`, `top_k` e `eos_id`. O começo do loop é o mesmo (cortar o contexto, rodar o modelo, pegar o último vetor de logits); muda a escolha do token.

**Função:** Ordem dentro do loop: se tem `top_k`, filtra os logits; se `temperature > 0`, divide por ela, aplica o softmax e sorteia com `multinomial`; senão, usa `argmax` como antes. Se o token gerado for o `eos_id` (token de fim de sequência), para antes de completar os `max_new_tokens`.

**Relação:** Junta os termos 14, 15 e 16. Exercício 5.3 (resposta minha): pra ficar determinístico como o `generate_text_simple`, basta `temperature=0.0` (o padrão), com qualquer `top_k`, porque cai no argmax; ou `top_k=1` com qualquer temperatura, porque só sobra um token com probabilidade diferente de zero (a não ser que dois logits empatem exatamente). Observação minha: do jeito que está no livro, só funciona com 1 texto por batch. Testei com 2 e dá erro de shape no `logits < min_val` e erro de "tensor ambíguo" no `if idx_next == eos_id`.

**Exemplo:** Modelo treinado no "The Verdict" com `top_k=25` e `temperature=1.4`: "Every effort moves you stand to work on surprise, a one of us had gone with random". Bem diferente do trecho decorado que o `generate_text_simple` gerava ("Every effort moves you know," was one of the axioms he laid down...).

---

## 18. state_dict e Checkpoint

**Equivalente:** Dicionário de estado e ponto de salvamento.

**Definição:** O `state_dict` é um dicionário que liga cada camada aos seus parâmetros, e salvar ele é a forma recomendada de salvar um modelo no PyTorch. Checkpoint, aqui, é salvar no mesmo arquivo o `state_dict` do modelo e o do otimizador.

**Função:** Pré-treinar é caro, então salvar evita treinar tudo de novo a cada sessão. Pra só usar o modelo, basta o `state_dict` dele. Pra continuar o treino, o livro recomenda salvar também o do otimizador: sem os dados do AdamW, o otimizador reseta e o modelo pode aprender mal ou nem convergir.

**Relação:** O `state_dict` só tem os pesos, não a arquitetura, então pra carregar precisa criar antes um `GPTModel` com a mesma configuração (observação minha). A extensão `.pth` é só convenção, e o `map_location=device` carrega os tensores no dispositivo certo. Depois de carregar: `model.eval()` pra inferência ou `model.train()` pra continuar treinando (termo 11). Carregar o otimizador também restaura o lr salvo (termo 12).

**Exemplo:** Só o modelo: `torch.save(model.state_dict(), "model.pth")` e depois `model.load_state_dict(torch.load("model.pth", map_location=device))`. Checkpoint: `torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, "model_and_optimizer.pth")` e, na volta, `model.load_state_dict(checkpoint["model_state_dict"])` e `optimizer.load_state_dict(checkpoint["optimizer_state_dict"])`. O exercício 5.4 é isso: carregar numa sessão nova e treinar mais uma época.

---

## 19. Pretrained Weights (OpenAI GPT-2)

**Equivalente:** Pesos pré-treinados (do GPT-2 da OpenAI).

**Definição:** Os pesos dos modelos GPT-2 que a OpenAI liberou publicamente, nos tamanhos 124M, 355M, 774M e 1558M. "Pesos" aqui são os parâmetros guardados nos atributos `.weight` das camadas `Linear` e `Embedding`, os mesmos que o `model.parameters()` acessa no treino.

**Função:** Substituem o pré-treino num corpus grande, que custaria de dezenas a centenas de milhares de dólares. Segundo o livro, dão ao modelo um bom ponto de partida pro fine-tuning (cap. 1).

**Relação:** A OpenAI salvou os pesos em TensorFlow, por isso precisa instalar o `tensorflow` (e o `tqdm`, pra barra de progresso do download). A `download_and_load_gpt2(model_size="124M", models_dir="gpt2")`, do `gpt_download.py` do repositório do livro, devolve dois dicionários: `settings`, equivalente ao nosso `GPT_CONFIG_124M` (`n_vocab`, `n_ctx`, `n_embd`, `n_head`, `n_layer`), e `params`, com os tensores (`wte`, `wpe`, `blocks`, `g`, `b`). Pra encaixar, o livro cria o `NEW_CONFIG` com `context_length` de volta pra 1.024 e `qkv_bias=True`, porque a OpenAI usou bias nas camadas de query, key e value. O livro diz que LLMs atuais não usam mais esse bias porque ele não melhora o desempenho; aqui ele entra só pra bater com os pesos.

**Exemplo:** `params["wte"].shape` dá `(50257, 768)`, igual ao nosso `tok_emb`. São 7 arquivos baixados, o maior com 498 MB. Conta minha: o `qkv_bias=True` acrescenta 3 × 768 × 12 = 27.648 parâmetros ao 124M; contando com weight tying, o total fica 124.439.808. Fazendo a mesma conta pros outros tamanhos dá 354.823.168, 774.030.080 e 1.557.611.200, que batem com os nomes 355M, 774M e 1558M.

---

## 20. load_weights_into_gpt e assign

**Equivalente:** N/A — nome próprio (funções do livro; em português, "carregar pesos no GPT" e "atribuir").

**Definição:** A `assign(left, right)` confere se os dois tensores têm o mesmo shape (senão dá `ValueError` com "Shape mismatch") e devolve o `right` como `torch.nn.Parameter`. A `load_weights_into_gpt(gpt, params)` usa o `assign` pra trocar, um por um, os pesos aleatórios do nosso `GPTModel` pelos da OpenAI.

**Função:** Fazer a ponte entre a nomenclatura da OpenAI e a nossa. O `GPTModel` nasce com pesos aleatórios, prontos pro pré-treino; depois dessa função, ele vira o GPT-2 já treinado, sem a gente treinar nada.

**Relação:** O mapeamento passa por todos os componentes do cap. 4: `wpe` e `wte` → `pos_emb` e `tok_emb`; em cada block, `c_attn` (Q, K e V juntos, separados em 3 com `np.split`) → `W_query`, `W_key` e `W_value`; `attn.c_proj` → `out_proj`; `mlp.c_fc` e `mlp.c_proj` → `ff.layers[0]` e `ff.layers[2]`; `g` e `b` de `ln_1`/`ln_2` → `scale` e `shift` de `norm1`/`norm2`; `g` e `b` finais → `final_norm`; e `wte` de novo → `out_head`, que é o weight tying do GPT-2 original (cap. 4). Observação minha sobre o `.T`: a OpenAI guarda as matrizes como [entrada, saída] e a `nn.Linear` como [saída, entrada]. Nas matrizes da feed forward (768 × 3.072) o `assign` acusaria erro sem o `.T`, mas nas quadradas (Q, K, V e `out_proj`, 768 × 768) passaria sem erro e o modelo ficaria errado sem aviso. Outra observação minha: como o `assign` cria um tensor novo a cada chamada, `out_head` e `tok_emb` começam com os mesmos valores, mas continuam sendo dois parâmetros separados no nosso modelo.

**Exemplo:** Depois de `load_weights_into_gpt(gpt, params)`, o `generate(model=gpt, ..., max_new_tokens=25, top_k=50, temperature=1.5)` gera "Every effort moves you toward finding an ideal new way to practice something! What makes us want to be on top of that?". O livro usa isso como teste: se tivesse um erro no mapeamento, o modelo não geraria texto coerente.