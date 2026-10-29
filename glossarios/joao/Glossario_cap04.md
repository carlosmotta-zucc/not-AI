# Glossário — Capítulo 4: Implementing a GPT Model from Scratch to Generate Text

João Antônio Sarturi Popp — Inteligência Artificial e Sistemas Inteligentes — UNOESC

---

## 1. Transformer

**Equivalente:** Transformer (termo mantido em inglês; arquitetura transformadora).

**Definição:** Arquitetura de rede neural profunda apresentada em 2017 no artigo "Attention Is All You Need". O original tinha dois submódulos, encoder e decoder, e foi feito pra tradução automática. O GPT (*generative pretrained transformer*) usa só a parte do decoder, por isso o livro chama de arquitetura *decoder-only*.

**Função:** É a base de todo o nosso modelo. Na prática, quando o livro monta o GPT, o "transformer" aparece como o transformer block (termo 15) repetido várias vezes, um em cima do outro.

**Relação:** A peça central dele é o self-attention, que já foi implementado na forma de `MultiHeadAttention`. O transformer original aplicava a LayerNorm depois da atenção e da feed forward (Post-LayerNorm, termo 16), diferente do que a gente faz no GPT.

**Exemplo:** O GPT-3 tem basicamente a mesma arquitetura do GPT-2, só que escalada de 1,5 bilhão pra 175 bilhões de parâmetros e treinada com mais dados. Por isso o livro usa o GPT-2: roda num notebook, enquanto o GPT-3 precisa de cluster de GPU (segundo a Lambda Labs, levaria 355 anos pra treinar numa única GPU V100).

---

## 2. GPT_CONFIG_124M

**Equivalente:** N/A — nome próprio (dicionário de configuração do livro; em português, "configuração do GPT de 124 milhões").

**Definição:** Dicionário Python com os hiperparâmetros do menor GPT-2, que o livro usa em todas as classes dessa parte.

**Função:** Centraliza a configuração: `vocab_size` = 50257 (vocabulário do tokenizer BPE), `context_length` = 1024 (máximo de tokens que o modelo aceita, limitado pelos embeddings posicionais), `emb_dim` = 768, `n_heads` = 12, `n_layers` = 12 (quantidade de transformer blocks), `drop_rate` = 0.1 e `qkv_bias` = False (sem bias nas `nn.Linear` de query, key e value, seguindo as LLMs modernas).

**Relação:** É passado como `cfg` pra `FeedForward` (termo 13), `TransformerBlock` (termo 15) e `GPTModel` (termo 18). O exercício 4.2 mostra a vantagem disso: dá pra montar GPT-2 medium, large e XL só trocando o dicionário, sem mexer em código.

**Exemplo:** Exercício 4.2: medium usa embedding de 1.024, 24 blocks e 16 cabeças; large usa 1.280, 36 e 20; XL usa 1.600, 48 e 25. Detalhe: o artigo original do GPT-2 fala em 117 milhões de parâmetros, mas esse número foi corrigido depois pra 124 milhões.

---

## 3. Parameters

**Equivalente:** Parâmetros.

**Definição:** Os pesos treináveis do modelo, ou seja, as variáveis internas que são ajustadas e otimizadas no treino pra minimizar uma função de perda (*loss function*).

**Função:** É o que o modelo "aprende" a partir dos dados de treino. Quando se fala que o GPT-2 tem 124 milhões de parâmetros, é a soma de todos os elementos de todas as matrizes de pesos.

**Relação:** Não confundir com os pesos de atenção, que são calculados na hora pra cada entrada. A contagem real de parâmetros do nosso modelo (163 milhões) só fecha com 124 milhões quando se considera o weight tying (termo 20).

**Exemplo:** Uma camada representada por uma matriz de pesos 2.048 × 2.048 tem 2.048 × 2.048 = 4.194.304 parâmetros, um por elemento da matriz.

---

## 4. DummyGPTModel

**Equivalente:** N/A — nome próprio (classe do livro; em português, "modelo GPT de mentira" ou de espaço reservado).

**Definição:** Versão simplificada do GPT, que o livro chama de *GPT backbone*, com a estrutura toda montada mas usando duas classes vazias no lugar das peças que ainda não tinham sido feitas: `DummyTransformerBlock` e `DummyLayerNorm`, que só devolvem o `x` que recebem.

**Função:** Dar a visão de cima pra baixo antes de entrar nos detalhes. É o passo 1 da figura 4.3: primeiro o esqueleto, depois LayerNorm, GELU, feed forward e shortcut, depois o transformer block e, por fim, o GPT completo.

**Relação:** Já tem o fluxo final do forward: embedding de token + embedding posicional → dropout → blocks → norma final → camada de saída. O `GPTModel` (termo 18) é essa mesma classe com as peças reais no lugar dos placeholders.

**Exemplo:** Com o tokenizer `tiktoken.get_encoding("gpt2")`, o batch "Every effort moves you" e "Every day holds a" vira `tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]])`. Mesmo com placeholders o modelo já roda e devolve um tensor de shape `torch.Size([2, 4, 50257])`.

---

## 5. Logits

**Equivalente:** Logits (termo mantido em inglês; saídas brutas, não normalizadas).

**Definição:** As saídas do modelo antes de qualquer normalização. Pra cada token de entrada sai um vetor com 50.257 posições, uma pra cada token do vocabulário. O livro descreve como as "probabilidades não normalizadas" do próximo token.

**Função:** São o resultado final do forward do GPT. Ainda não são probabilidades (podem ser negativas e não somam 1), então precisam passar pelo softmax antes de escolher o próximo token.

**Relação:** Saem da camada de saída (termo 19). No `generate_text_simple` (termo 24) só o último vetor de logits é usado. Como o softmax preserva a ordem, o maior logit já indica o token escolhido (termo 25).

**Exemplo:** No `GPTModel`, o primeiro vetor do batch começa com `[0.3613, 0.4222, -0.0711, ...]`. A quantidade de vetores de saída é igual à de tokens de entrada, então pra "Every effort moves you" a saída corresponde a "effort moves you forward": o "Every" não aparece e o último vetor é o que deveria virar "forward".

---

## 6. Vanishing Gradient

**Equivalente:** Desaparecimento do gradiente (o livro cita junto o *exploding gradient*, explosão do gradiente).

**Definição:** Problema em que os gradientes, que guiam a atualização dos pesos no treino, vão ficando cada vez menores conforme são propagados de trás pra frente pelas camadas, até que as primeiras camadas praticamente não aprendem.

**Função:** Não é um componente, é o problema que justifica duas peças da arquitetura. Gradiente que some ou explode deixa o treino instável e dificulta achar pesos que minimizem a perda.

**Relação:** A LayerNorm (termo 7) ataca a instabilidade normalizando as ativações, e as shortcut connections (termo 14) atacam o desaparecimento criando um caminho mais curto pro gradiente.

**Exemplo:** Na `ExampleDeepNeuralNetwork` de 5 camadas sem shortcut, usando `nn.MSELoss()` com alvo 0 e `loss.backward()`, a média absoluta do gradiente cai de 0.0050 na última camada (`layers.4`) pra 0.0002 na primeira (`layers.0`).

---

## 7. Layer Normalization (LayerNorm)

**Equivalente:** Normalização de camada.

**Definição:** Ajusta as ativações (saídas) de uma camada pra terem média 0 e variância 1 (variância unitária). Subtrai a média e divide pela raiz da variância (desvio padrão), na última dimensão do tensor:

$$
\text{LayerNorm}(x) = \text{scale} \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \text{shift}
$$

**Função:** Deixa o treino mais estável e eficiente e acelera a convergência pra bons pesos. No nosso GPT ela aparece antes da atenção (`norm1`), antes da feed forward (`norm2`) e uma última vez antes da camada de saída (`final_norm`).

**Relação:** Usa `dim=-1` porque a última dimensão é a de embedding; isso funciona igual pra tensor 2D e pro 3D do GPT `[batch_size, num_tokens, emb_dim]`, sem trocar `dim=1` por `dim=2`. O `keepdim=True` mantém o número de dimensões (sai 2×1 em vez de vetor). Observação minha: o texto diz que no GPT-2 a LayerNorm vai "antes e depois" da atenção, mas no código do `TransformerBlock` o que existe é uma antes da atenção e outra antes da feed forward; a "depois" da atenção é, na prática, a `norm2`.

**Exemplo:** Saída de `nn.Sequential(nn.Linear(5, 6), nn.ReLU())`: primeira linha com média 0.1324 e variância 0.0231. Depois de normalizar vira `[0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719]`, com média -5.9605e-08 (praticamente 0, só não é exato por erro de precisão numérica) e variância 1. O `torch.set_printoptions(sci_mode=False)` tira a notação científica.

---

## 8. eps, scale e shift

**Equivalente:** Épsilon, escala e deslocamento.

**Definição:** `eps` é uma constante pequena (1e-5) somada à variância; `scale` e `shift` são dois parâmetros treináveis da classe `LayerNorm`, do mesmo tamanho do embedding.

**Função:** O `eps` evita divisão por zero. O `scale` e o `shift` deixam o modelo ajustar a normalização sozinho no treino, se isso melhorar o desempenho, em vez de ficar preso a média 0 e variância 1.

**Relação:** São criados como `nn.Parameter(torch.ones(emb_dim))` e `nn.Parameter(torch.zeros(emb_dim))`. Começando em 1 e 0, a saída inicial é exatamente a normalização pura (observação minha, sai direto da fórmula do termo 7).

**Exemplo:** No GPT de 124M, cada LayerNorm tem 768 + 768 = 1.536 parâmetros (conta minha). Cada block tem duas, mais a final.

---

## 9. Biased Variance

**Equivalente:** Variância enviesada.

**Definição:** Cálculo de variância dividindo por n (número de elementos) e não por n − 1. Ou seja, sem a correção de Bessel, que é a que normalmente se usa pra estimativa de variância de amostra.

**Função:** É o detalhe do `unbiased=False` no `x.var(...)` da `LayerNorm`. Foi escolhido pra ficar compatível com as camadas de normalização do GPT-2 original, que foi implementado em TensorFlow, e esse é o comportamento padrão do TensorFlow.

**Relação:** Pra LLM a diferença é desprezível, porque n é a dimensão do embedding (768), e dividir por 768 ou por 767 quase não muda nada.

**Exemplo:** `var = x.var(dim=-1, keepdim=True, unbiased=False)` dentro do `forward` da `LayerNorm`.

---

## 10. Batch Normalization

**Equivalente:** Normalização em lote.

**Definição:** Método tradicional de normalização em redes neurais que normaliza ao longo da dimensão do batch, enquanto a LayerNorm normaliza ao longo da dimensão das features.

**Função:** Não é usado no nosso modelo. Entra só como comparação pra justificar a escolha da LayerNorm.

**Relação:** Como a LayerNorm (termo 7) normaliza cada entrada sem depender do tamanho do batch, ela é mais flexível e estável quando o hardware ou o caso de uso obriga a mudar o batch size no treino ou na inferência (termo 22). Isso ajuda principalmente em treino distribuído e em ambiente com pouco recurso.

**Exemplo:** Na figura 4.6, `dim=0` calcula uma média por coluna (atravessando as amostras, o que lembra a lógica do batch norm) e `dim=-1` calcula uma média por linha, que é o que a LayerNorm faz.

---

## 11. ReLU

**Equivalente:** Unidade linear retificada (*rectified linear unit*).

**Definição:** Função de ativação linear por partes: se a entrada é positiva, devolve ela mesma; se é negativa, devolve zero.

**Função:** Foi a ativação padrão em deep learning por ser simples e funcionar bem. Aqui ela aparece só no exemplo da LayerNorm e como comparação com a GELU.

**Relação:** Tem um "bico" no zero, o que às vezes dificulta a otimização em redes muito profundas, e zera tudo que é negativo. A GELU (termo 12) resolve as duas coisas.

**Exemplo:** A saída `tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000], ...])` não tem nenhum valor negativo justamente por causa da ReLU.

---

## 12. GELU

**Equivalente:** Unidade linear de erro gaussiano (*Gaussian error linear unit*).

**Definição:** Função de ativação suave e não linear. A versão exata é GELU(x) = x · Φ(x), em que Φ é a função de distribuição acumulada da normal padrão. Na prática se usa uma aproximação mais barata, encontrada por ajuste de curva, que foi a usada no treino do GPT-2 original:

$$
\text{GELU}(x) \approx 0{,}5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \cdot \left(x + 0{,}044715 \cdot x^3\right)\right]\right)
$$

**Função:** É a ativação dentro da feed forward do transformer block. Por ser suave, permite ajustes mais finos nos parâmetros durante o treino. E, como não zera os negativos, neurônios com entrada negativa ainda contribuem pro aprendizado, só que menos que os positivos.

**Relação:** Fica entre as duas `nn.Linear` da `FeedForward` (termo 13). O livro cita também a SwiGLU como outra ativação usada em LLMs, mas não implementa.

**Exemplo:** No gráfico da figura 4.8 (`torch.linspace(-3, 3, 100)`), a GELU parece uma ReLU arredondada, com gradiente diferente de zero pra quase todos os valores negativos, exceto perto de x ≈ −0,75.

---

## 13. FeedForward Network

**Equivalente:** Rede feed forward (rede de propagação direta).

**Definição:** Pequena rede neural dentro do transformer block, formada por uma `nn.Linear`, uma `GELU` e outra `nn.Linear`. A primeira expande o embedding 4 vezes (768 → 3.072) e a segunda contrai de volta (3.072 → 768).

**Função:** Melhora a capacidade do modelo de aprender e generalizar. A expansão pra um espaço maior permite explorar uma representação mais rica. Como entrada e saída têm o mesmo tamanho, dá pra empilhar várias sem ajustar dimensões.

**Relação:** Faz par com a atenção dentro do block (termo 15): a atenção analisa a relação entre os tokens, e a feed forward modifica cada posição individualmente. Exercício 4.1 (conta minha): a feed forward tem 4.722.432 parâmetros e a `MultiHeadAttention` tem 2.360.064, então a feed forward tem praticamente o dobro.

**Exemplo:** `ffn = FeedForward(GPT_CONFIG_124M)` com `x = torch.rand(2, 3, 768)` devolve `torch.Size([2, 3, 768])`. Por dentro (figura 4.9), o shape passa por `(2, 3, 3072)` antes de voltar.

---

## 14. Shortcut Connection

**Equivalente:** Conexão de atalho (também chamada de conexão residual ou *skip connection*).

**Definição:** Conexão que soma a entrada de uma camada (ou bloco) com a saída dela, criando um caminho alternativo que pula uma ou mais camadas. Surgiu nas redes residuais de visão computacional.

**Função:** Preserva o fluxo do gradiente no backward, criando um caminho mais curto pra ele e evitando o vanishing gradient (termo 6). O livro coloca como peça central de modelos grandes como LLMs.

**Relação:** No exemplo só é somada quando os shapes batem (`x.shape == layer_output.shape`); no transformer block sempre batem, porque atenção e feed forward mantêm a dimensão. Ali ela aparece duas vezes: em volta da atenção e em volta da feed forward (termo 15).

**Exemplo:** Mesma rede de 5 camadas, mesma seed, agora com `use_shortcut=True`: gradiente de 1.3259 em `layers.4` e 0.2217 em `layers.0`. A última camada ainda tem gradiente maior, mas ele estabiliza e não some nas primeiras camadas. No código é só `x = x + layer_output`.

---

## 15. Transformer Block

**Equivalente:** Bloco transformer.

**Definição:** O bloco fundamental do GPT, que junta atenção multi-cabeça mascarada, LayerNorm, dropout, feed forward com GELU e shortcut connections numa classe só (`TransformerBlock`).

**Função:** É o componente principal do GPT, repetido `n_layers` vezes (12 no GPT-2 de 124M, 48 no de 1.542M). Cada vetor sai com o mesmo tamanho que entrou, mas re-codificado com informação contextual da sequência inteira.

**Relação:** O forward tem duas metades iguais: `shortcut = x` → `norm1` → `att` → `drop_shortcut` → `x + shortcut`, e depois `shortcut = x` → `norm2` → `ff` → `drop_shortcut` → `x + shortcut`. A `MultiHeadAttention` entra com `d_in = d_out = emb_dim`. A posição da LayerNorm define que é Pre-LayerNorm (termo 16).

**Exemplo:** `block = TransformerBlock(GPT_CONFIG_124M)` com `x = torch.rand(2, 4, 768)` dá entrada e saída `torch.Size([2, 4, 768])`. Manter o shape não é coincidência: é o que permite empilhar os blocks e manter a relação um-pra-um entre vetor de entrada e de saída.

---

## 16. Pre-LayerNorm e Post-LayerNorm

**Equivalente:** LayerNorm antes e LayerNorm depois.

**Definição:** Pre-LayerNorm é aplicar a normalização antes da atenção e antes da feed forward; Post-LayerNorm é aplicar depois delas.

**Função:** Nosso `TransformerBlock` usa Pre-LayerNorm, com o dropout vindo depois de cada componente. O livro diz que a Post-LayerNorm, usada em arquiteturas antigas como o transformer original, costuma ter dinâmica de treino pior.

**Relação:** É uma diferença de arquitetura entre o GPT e o transformer original (termo 1). Muda só a ordem das chamadas no forward, as peças são as mesmas.

**Exemplo:** No forward do block, o `self.norm1(x)` vem antes do `self.att(x)` e o `self.norm2(x)` vem antes do `self.ff(x)`. Se fosse Post-LayerNorm, a normalização ficaria depois.

---

## 17. Dropout (no GPT)

**Equivalente:** Dropout (termo mantido em inglês; descarte aleatório de unidades).

**Definição:** Mesmo conceito já visto na atenção: zerar unidades aleatórias durante o treino pra evitar overfitting. No GPT a taxa vem do `drop_rate` = 0.1, ou seja, 10% das unidades descartadas.

**Função:** Regularizar o modelo inteiro, não só a atenção. Aparece em três lugares: logo depois de somar os embeddings (`drop_emb`), depois da atenção e da feed forward antes de somar o atalho (`drop_shortcut`) e dentro da `MultiHeadAttention`, sobre os pesos de atenção.

**Relação:** Só vale no treino. Na geração de texto o modelo vai pro `model.eval()`, que desliga o dropout (termo 22). Exercício 4.3 (resposta minha): trocar o `drop_rate` único por três chaves no dicionário, por exemplo `drop_rate_emb`, `drop_rate_shortcut` e `drop_rate_attn`, e passar cada uma pra camada correspondente.

**Exemplo:** No `GPTModel`: `self.drop_emb = nn.Dropout(cfg["drop_rate"])`. No `TransformerBlock`: `self.drop_shortcut = nn.Dropout(cfg["drop_rate"])`, usado duas vezes no forward.

---

## 18. GPTModel

**Equivalente:** N/A — nome próprio (classe do livro; em português, "modelo GPT").

**Definição:** Implementação completa do GPT-2 de 124 milhões de parâmetros: o `DummyGPTModel` com `TransformerBlock` e `LayerNorm` reais no lugar dos placeholders.

**Função:** Recebe um batch de IDs de tokens e devolve os logits. O `__init__` cria os embeddings de token e de posição, a pilha `nn.Sequential` de `n_layers` blocks, a `LayerNorm` final e a camada de saída. O forward calcula os embeddings, soma a posição, aplica dropout, passa pelos blocks, normaliza e gera os logits.

**Relação:** Junta tudo: termos 7, 15, 17 e 19. O `torch.arange(seq_len, device=in_idx.device)` garante que o tensor de posições fica no mesmo dispositivo (CPU ou GPU) dos dados de entrada.

**Exemplo:** `model = GPTModel(GPT_CONFIG_124M)` com o batch de "Every effort moves you" / "Every day holds a" devolve `torch.Size([2, 4, 50257])`: 2 textos, 4 tokens cada, 50.257 logits por token. A classe fica curta justamente porque a complexidade está toda no `TransformerBlock`.

---

## 19. Language Model Head

**Equivalente:** Cabeça de modelo de linguagem (o livro chama de *linear output layer* ou *output head*, no código `out_head`).

**Definição:** A última camada do GPT, uma `nn.Linear(emb_dim, vocab_size, bias=False)` que projeta cada vetor de 768 dimensões no espaço do vocabulário, de 50.257 dimensões. Observação minha: "Language Model Head" é o nome comum na literatura, o livro não usa esse termo, usa "camada de saída linear".

**Função:** Transforma a saída dos transformer blocks em logits (termo 5), um valor pra cada token possível, que é o que permite prever o próximo token.

**Relação:** Vem logo depois da `final_norm`, que estabiliza as saídas dos blocks. O peso dela tem shape `[50257, 768]`, o mesmo do embedding de token, e é isso que torna possível o weight tying (termo 20).

**Exemplo:** `model.out_head.weight.shape` dá `torch.Size([50257, 768])`. Isso são 50.257 × 768 = 38.597.376 parâmetros (conta minha), quase um quarto do modelo inteiro só nessa camada.

---

## 20. Weight Tying

**Equivalente:** Compartilhamento (amarração) de pesos.

**Definição:** Técnica do GPT-2 original que reaproveita os pesos da camada de embedding de token na camada de saída, em vez de ter duas matrizes separadas.

**Função:** Explica por que o nosso modelo conta 163.009.536 parâmetros e não 124 milhões: o GPT-2 original não conta a camada de saída duas vezes. Reduz memória e custo computacional.

**Relação:** Só funciona porque `tok_emb.weight` e `out_head.weight` têm o mesmo shape, `[50257, 768]` (termo 19). Mesmo assim, o nosso `GPTModel` usa camadas separadas, porque segundo o autor isso dá treino e desempenho melhores, e as LLMs modernas também fazem assim.

**Exemplo:** `total_params - sum(p.numel() for p in model.out_head.parameters())` dá 124.412.160, que bate com o tamanho original do GPT-2.

---

## 21. numel() e Tamanho do Modelo

**Equivalente:** Número de elementos (*number of elements*) e tamanho em memória.

**Definição:** `numel()` é o método do PyTorch que devolve quantos elementos tem um tensor. Somando de todos os tensores de parâmetros, tem-se o total de parâmetros do modelo.

**Função:** Medir o modelo e estimar quanto de memória ele ocupa. Assumindo float32, cada parâmetro ocupa 4 bytes.

**Relação:** É a ferramenta usada pra achar a diferença do weight tying (termo 20) e pra resolver o exercício 4.1 (termo 13).

**Exemplo:** `sum(p.numel() for p in model.parameters())` dá 163.009.536; vezes 4 bytes, dividido por 1024 × 1024, dá 621,83 MB, mesmo sendo uma LLM pequena. Conferi a conta separando por partes (conta minha): embedding de token 38.597.376 + embedding posicional 786.432 + 12 blocks × 7.085.568 + LayerNorm final 1.536 + camada de saída 38.597.376 = 163.009.536.

---

## 22. Inference

**Equivalente:** Inferência.

**Definição:** A fase em que o modelo é usado pra gerar saída a partir de uma entrada, sem atualizar os pesos, ao contrário do treino. Observação minha: o livro usa o termo várias vezes ("treino e inferência"), mas não para pra definir formalmente; essa definição é minha juntando os trechos.

**Função:** No nosso caso, inferência é a geração de texto. O livro prepara o modelo pra isso com `model.eval()`, que desliga componentes aleatórios usados só no treino, como o dropout.

**Relação:** No `generate_text_simple` (termo 24) o modelo é chamado dentro de `with torch.no_grad():`. O livro não explica essa linha aqui; pelo que já se sabe de PyTorch, ela desliga o cálculo de gradiente, que não é necessário quando não se está treinando (observação minha). É também na inferência que pesa a vantagem da LayerNorm sobre o batch norm (termo 10).

**Exemplo:** Sequência usada no livro antes de gerar: `model.eval()` e depois `generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M["context_length"])`.

---

## 23. Autoregressive Generation

**Equivalente:** Geração autoregressiva.

**Definição:** Gerar texto um token por vez, em que cada token gerado é adicionado à entrada e passa a ser usado pra prever o próximo. O livro chama esse tipo de modelo de autoregressivo lá na visão geral da arquitetura GPT: modelos que usam as próprias saídas anteriores como entrada das previsões seguintes.

**Função:** É o jeito como o GPT produz texto. A cada iteração o contexto cresce, o que ajuda o texto a ficar coerente, porque cada palavra nova é escolhida com base em tudo que veio antes.

**Relação:** Combina com a atenção causal: o modelo só pode olhar os tokens anteriores, que é justamente o que ele tem na hora de gerar. Em código, isso é o loop do `generate_text_simple` (termo 24).

**Exemplo:** Figura 4.16: começando em "Hello, I am", a 1ª iteração adiciona "a", a 2ª "model", a 3ª "ready", e na 6ª fica "Hello, I am a model ready to help.". Em IDs: `[15496, 11, 314, 716]` → `[257]` → `[2746]` → `[3492]` ...

---

## 24. generate_text_simple

**Equivalente:** N/A — nome próprio (função do livro; em português, "gerar texto simples").

**Definição:** Função que implementa o loop de geração: repete `max_new_tokens` vezes a previsão de um token e o acrescenta ao contexto.

**Função:** A cada passo: corta o contexto pros últimos `context_size` tokens (`idx[:, -context_size:]`), roda o modelo, pega só o último vetor de logits (`logits[:, -1, :]`, de `(batch, n_tokens, vocab_size)` pra `(batch, vocab_size)`), aplica softmax, escolhe o maior com `argmax` e concatena com `torch.cat((idx, idx_next), dim=1)`.

**Relação:** É a geração autoregressiva (termo 23) em código, usando a escolha gulosa (termo 25). O corte do contexto existe porque o modelo só suporta até `context_length` tokens; se ele suporta 5 e o contexto tem 10, só os últimos 5 entram.

**Exemplo:** `tokenizer.encode("Hello, I am")` dá `[15496, 11, 314, 716]`; o `.unsqueeze(0)` adiciona a dimensão de batch, shape `[1, 4]`. Com 6 tokens novos sai um tensor de tamanho 10, que decodificado dá "Hello, I am Featureiman Byeswickattribute argue". Sai lixo porque o modelo ainda não foi treinado, os pesos estão aleatórios.

---

## 25. Greedy Decoding

**Equivalente:** Decodificação gulosa.

**Definição:** Estratégia de escolha em que o modelo sempre pega o token mais provável, ou seja, a posição do maior valor (`torch.argmax`).

**Função:** É o critério de escolha do `generate_text_simple`. O livro mostra o softmax antes do argmax só pra deixar claro o caminho logits → probabilidades → token.

**Relação:** Como o softmax é monotônico (preserva a ordem das entradas), a posição do maior valor é a mesma nos logits e nas probabilidades. Então o softmax ali é redundante: `torch.argmax` direto nos logits (termo 5) dá exatamente o mesmo resultado.

**Exemplo:** Figura 4.17: o último vetor de logits vira probabilidades tipo `[0.0001, ..., 0.0200, ..., 0.0001]`; se o maior está na posição 257, o token escolhido é o ID 257, que decodificado é "a".