# Glossário — Capítulo 3: Coding Attention Mechanisms

João Antônio Sarturi Popp — Inteligência Artificial e Sistemas Inteligentes — UNOESC

---

## 1. Encoder–Decoder RNN

**Equivalente:** RNN codificador–decodificador.

**Definição:** Arquitetura com dois submódulos: o encoder lê e processa o texto de entrada inteiro, e o decoder gera o texto de saída. Antes dos transformers, era a opção mais usada pra tradução automática. RNN é uma rede em que a saída de um passo entra como entrada do passo seguinte, por isso serve pra dados sequenciais como texto.

**Função:** Não faz parte da nossa LLM. Aparece no capítulo só como o "antes": é o problema que motivou os mecanismos de atenção. O próprio livro diz que não precisa entender RNN a fundo pra construir a LLM.

**Relação:** O encoder resume a entrada inteira num hidden state (termo 2) e o decoder só enxerga esse resumo. Essa limitação levou à atenção de Bahdanau (termo 4) e depois ao self-attention (termo 5).

**Exemplo:** Traduzir "Kannst du mir helfen diesen Satz zu uebersetzen" palavra por palavra dá "Can you me help this sentence to translate", que está errado. A tradução certa ("Can you help me to translate this sentence") exige olhar palavras que aparecem antes ou depois na frase original.

---

## 2. Hidden State

**Equivalente:** Estado oculto.

**Definição:** Os valores internos das camadas ocultas de uma RNN, atualizados a cada passo da sequência. O livro diz que dá pra pensar nele como um vetor de embedding (cap. 2).

**Função:** No encoder–decoder RNN, o hidden state final funciona como a "memória" que tenta guardar o significado da frase inteira pra passar pro decoder.

**Relação:** É o gargalo do encoder–decoder: o decoder não acessa os hidden states anteriores do encoder, só o atual. Com isso se perde contexto, principalmente em frases complexas com dependências longas. O vetor de contexto (termo 6) resolve isso de outro jeito, combinando todos os tokens pra cada posição.

**Exemplo:** Na figura 3.4, a frase alemã entra token por token e, no fim, o encoder tem um único hidden state representando "Kannst du mir..." inteiro. É só com ele que o decoder começa a gerar "Can you help...".

---

## 3. Attention Mechanism

**Equivalente:** Mecanismo de atenção.

**Definição:** Mecanismo que transforma os elementos de entrada em vetores de contexto enriquecidos, que incorporam informação de todas as entradas, dando mais importância a umas do que a outras.

**Função:** É o assunto do capítulo e o passo 2 do estágio 1 da construção da LLM (figura 3.1). O capítulo constrói quatro versões em sequência: self-attention simplificado, self-attention com pesos treináveis, atenção causal e atenção multi-cabeça.

**Relação:** Nasceu como correção do encoder–decoder RNN (termo 4) e, na LLM, aparece na forma de self-attention (termo 5). A "importância" de cada entrada é dada pelos pesos de atenção (termo 10).

**Exemplo:** Na figura 3.5, ao gerar a segunda palavra da tradução, o decoder acessa todos os tokens de entrada, e a espessura da linha pontilhada mostra o quanto cada token de entrada importa pra aquele token de saída.

---

## 4. Bahdanau Attention

**Equivalente:** Atenção de Bahdanau (nome do primeiro autor do artigo).

**Definição:** Mecanismo de atenção para RNNs, de 2014, que modifica o encoder–decoder pra que o decoder possa acessar seletivamente diferentes partes da entrada a cada passo de decodificação.

**Função:** Não é implementada no livro (é um método de RNN, fora do escopo). Entra como origem histórica: três anos depois veio o transformer original, com um self-attention inspirado nela, mostrando que RNN não era necessária pra PLN.

**Relação:** Resolve o gargalo do hidden state (termo 2). A diferença pro self-attention (termo 5) é que ela relaciona duas sequências diferentes (entrada e saída), enquanto o self-attention relaciona a sequência com ela mesma.

**Exemplo:** O livro usa a figura 3.5, mas avisa que ela mostra só a ideia geral de atenção, não a implementação exata de Bahdanau.

---

## 5. Self-Attention

**Equivalente:** Autoatenção.

**Definição:** Mecanismo que permite que cada posição da sequência de entrada considere a relevância de (ou "preste atenção a") todas as outras posições da mesma sequência ao calcular a representação dela.

**Função:** É a peça central de toda LLM baseada em transformer, como os GPT. O objetivo é calcular, pra cada entrada x(i), um vetor de contexto z(i). São três passos (figura 3.12): pontuações de atenção → pesos de atenção → vetores de contexto.

**Relação:** O "self" é o que diferencia da atenção tradicional (termo 4), que liga duas sequências. O livro faz primeiro uma versão simplificada, sem pesos treináveis, direto nos embeddings; depois acrescenta Wq, Wk e Wv (termo 12), e isso vira o scaled dot-product attention (termo 14).

**Exemplo:** Com a frase "Your journey starts with one step" já em embeddings 3D (tensor `inputs`, 6×3), tomando "journey" (x(2)) como query, o self-attention calcula z(2) misturando os 6 tokens, inclusive o próprio "journey".

---

## 6. Context Vector

**Equivalente:** Vetor de contexto.

**Definição:** Um embedding enriquecido: pra um elemento x(i), é um vetor que contém informação dele e de todos os outros elementos da sequência, calculado como soma ponderada pelos pesos de atenção.

**Função:** É a saída do mecanismo de atenção. Cria representações enriquecidas de cada token, o que é essencial pra LLM entender a relação e a relevância das palavras entre si.

**Relação:** Na versão simplificada é a soma ponderada dos próprios embeddings de entrada; na versão treinável é a soma ponderada dos vetores value (termo 11). Os pesos vêm do termo 10.

**Exemplo:** Versão simplificada: z(2) = `tensor([0.4419, 0.6515, 0.5683])` (3D, igual à entrada). Versão treinável com d_out=2: z(2) = `tensor([0.3061, 0.8210])`. Pra todos os tokens de uma vez: `all_context_vecs = attn_weights @ inputs`, que dá uma matriz 6×3.

---

## 7. Dot Product

**Equivalente:** Produto escalar.

**Definição:** Forma concisa de multiplicar dois vetores elemento a elemento e somar os produtos, resultando num único número.

**Função:** É a medida de similaridade usada pra calcular as pontuações de atenção: quanto maior o produto escalar, mais alinhados os vetores e maior a atenção entre os dois elementos.

**Relação:** Gera a pontuação de atenção (termo 8). Com várias entradas, o livro troca os loops `for` por multiplicação de matrizes (`inputs @ inputs.T`), que dá o mesmo resultado e é mais rápida.

**Exemplo:** O livro mostra que somar `inputs[0][idx] * query[idx]` num loop dá `tensor(0.9544)`, exatamente igual a `torch.dot(inputs[0], query)`.

---

## 8. Attention Score

**Equivalente:** Pontuação de atenção (símbolo ω).

**Definição:** Valor intermediário, ainda não normalizado, que mede a afinidade entre a query e cada elemento da entrada.

**Função:** É o passo 1 do self-attention. Na versão simplificada é o produto escalar entre embeddings; na versão treinável é o produto escalar entre a query de um token e a key de outro.

**Relação:** Vem do produto escalar (termo 7) e ainda precisa passar pelo softmax (termo 9) pra virar peso de atenção (termo 10). Na versão treinável, antes do softmax, é dividida por √d_k (termo 14).

**Exemplo:** Simplificado, pra query "journey": `tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])`. Treinável: ω22 = `query_2.dot(keys_2)` = 1.8524, e o vetor completo é `tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])`.

---

## 9. Softmax

**Equivalente:** Softmax (termo mantido em inglês; também chamada de função exponencial normalizada).

**Definição:** Função que converte um vetor de números numa distribuição de probabilidade: todos os valores ficam positivos e somam 1.

**Função:** Normaliza as pontuações de atenção pra virarem pesos de atenção. O livro prefere o softmax a simplesmente dividir pela soma porque lida melhor com valores extremos e tem propriedades de gradiente melhores no treino.

**Relação:** Liga o termo 8 ao termo 10. Duas propriedades dele voltam depois: com valores de entrada grandes ele se comporta quase como função degrau (motivo do escalonamento, termo 14), e e^(−∞) = 0, que é o truque da máscara causal (termo 16).

**Exemplo:** Dividir as pontuações de "journey" pela soma dá `[0.1455, 0.2278, ...]`; com softmax dá `[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]`. A `softmax_naive` (`torch.exp(x) / torch.exp(x).sum(dim=0)`) pode ter overflow/underflow, por isso na prática se usa `torch.softmax(attn_scores, dim=-1)`, que normaliza cada linha.

---

## 10. Attention Weight

**Equivalente:** Peso de atenção (símbolo α).

**Definição:** A pontuação de atenção depois de normalizada: valores positivos que, em cada linha, somam 1. Determinam o quanto o vetor de contexto depende de cada parte da entrada.

**Função:** É o fator que pondera cada vetor na soma que gera o vetor de contexto.

**Relação:** Aqui é fácil confundir: peso de atenção não é o mesmo que *weight parameter* (os valores de Wq, Wk, Wv). Os weight parameters são coeficientes aprendidos no treino que definem as conexões da rede; os pesos de atenção são dinâmicos e específicos do contexto, mudam pra cada entrada. É sobre a matriz de pesos de atenção que atuam a máscara causal (termo 16) e o dropout (termo 18).

**Exemplo:** Versão treinável, query "journey": `tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])`. Pra frase inteira, a matriz é 6×6, uma linha por query, e `attn_weights.sum(dim=-1)` dá 1.0000 em todas as linhas.

---

## 11. Query, Key e Value

**Equivalente:** Consulta, chave e valor.

**Definição:** Três vetores obtidos de cada token de entrada, multiplicando o embedding pelas matrizes Wq, Wk e Wv. Os nomes vêm de recuperação de informação e bancos de dados.

**Função:** A query representa o item atual que o modelo está tentando entender e serve pra "sondar" o resto da sequência. A key é como a chave de um índice: cada token tem uma, e ela é comparada com a query. A value é o conteúdo real do token, recuperado depois que se sabe quais keys são mais relevantes.

**Relação:** query · key = pontuação de atenção (termo 8); pesos de atenção × values = vetor de contexto (termo 6). Mesmo calculando só z(2), precisa das keys e values de todos os tokens, porque todos entram no cálculo dos pesos.

**Exemplo:** Pra "journey": `query_2 = x_2 @ W_query` = `tensor([0.4306, 1.4551])`. `keys = inputs @ W_key` e `values = inputs @ W_value` têm shape `[6, 2]`: os 6 tokens projetados de 3D pra 2D.

---

## 12. Trainable Weight Matrices (Wq, Wk, Wv)

**Equivalente:** Matrizes de pesos treináveis.

**Definição:** Três matrizes de parâmetros, de shape (d_in, d_out), que projetam os embeddings de entrada em queries, keys e values. São atualizadas durante o treino.

**Função:** São a diferença principal entre o self-attention simplificado e o usado de verdade em LLMs: é por causa delas que o módulo de atenção consegue aprender a produzir "bons" vetores de contexto (o treino em si fica pro cap. 5).

**Relação:** Geram os vetores do termo 11. Na `SelfAttention_v1` são `nn.Parameter(torch.rand(d_in, d_out))`; na `SelfAttention_v2` viram `nn.Linear` (termo 13). Em modelos GPT normalmente d_in = d_out; o livro usa d_in=3 e d_out=2 só pra facilitar acompanhar as contas.

**Exemplo:** `torch.manual_seed(123)` e `W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)`. O `requires_grad=False` é só pra limpar a saída do exemplo; pra treinar teria que ser `True`.

---

## 13. nn.Linear

**Equivalente:** N/A — nome próprio (classe do PyTorch que implementa uma camada linear).

**Definição:** Camada do PyTorch que, com o bias desligado, faz na prática uma multiplicação de matriz.

**Função:** Substitui o `nn.Parameter` manual nas matrizes Wq, Wk e Wv. A vantagem, segundo o livro, é o esquema de inicialização de pesos otimizado, que deixa o treino mais estável e eficaz.

**Relação:** É usada na `SelfAttention_v2`, na `CausalAttention` e na `MultiHeadAttention`, com `bias=qkv_bias` (padrão `False`). Também é a camada de projeção de saída (termo 25). Detalhe do exercício 3.1: `nn.Linear` guarda a matriz de pesos transposta.

**Exemplo:** A `SelfAttention_v1` e a `SelfAttention_v2` dão saídas diferentes porque a inicialização é diferente; a v2 dá `[-0.0739, 0.0713]` pro primeiro token. O exercício 3.1 pede pra copiar os pesos da v2 pra v1 e mostrar que aí as saídas ficam iguais.

---

## 14. Scaled Dot-Product Attention

**Equivalente:** Atenção por produto escalar escalonado.

**Definição:** O self-attention com pesos treináveis usado no transformer original, nos GPT e na maioria das LLMs. As pontuações (query · key) são divididas pela raiz quadrada da dimensão das keys (d_k) antes do softmax.

**Função:** O escalonamento melhora o treino evitando gradientes pequenos. Com embeddings grandes (mais de 1.000 dimensões nos GPT), os produtos escalares ficam grandes, o softmax vira quase uma função degrau e os gradientes na retropropagação chegam perto de zero, o que desacelera ou trava o aprendizado.

**Relação:** É o self-attention (termo 5) + matrizes treináveis (termo 12) + divisão por √d_k. É a base da `CausalAttention` e da `MultiHeadAttention`. O "scaled" do nome vem justamente dessa divisão.

**Exemplo:** `d_k = keys.shape[-1]` (= 2) e `attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)`. O ω22 = 1.8524, dividido por √2, entra no softmax e vira o peso 0.2264. Elevar a 0.5 é o mesmo que tirar raiz quadrada.

---

## 15. Causal Attention

**Equivalente:** Atenção causal (também chamada de atenção mascarada, *masked attention*).

**Definição:** Forma especializada de self-attention que restringe o modelo a considerar só as entradas anteriores e a atual ao processar cada token. O self-attention padrão, ao contrário, acessa a sequência inteira de uma vez.

**Função:** É essencial pra modelagem de linguagem: como cada palavra prevista deve depender só das anteriores, o modelo não pode ver tokens futuros ao calcular os vetores de contexto. Garante a ordem temporal durante a geração de texto.

**Relação:** Implementada com a máscara causal (termo 16) mais o dropout (termo 18) na classe `CausalAttention`. Várias `CausalAttention` em paralelo formam a multi-head attention (termo 20).

**Exemplo:** Na figura 3.19, na linha de "journey" sobram só os pesos de "Your" e do próprio "journey" (0.55 e 0.44); "starts", "with", "one" e "step" são mascarados. A linha de "Your" fica só com 1.0 nele mesmo.

---

## 16. Causal Attention Mask

**Equivalente:** Máscara de atenção causal.

**Definição:** Matriz que marca as posições acima da diagonal (tokens futuros) pra que elas não entrem no cálculo dos pesos de atenção.

**Função:** O livro mostra dois jeitos. (1) Aplicar o softmax, multiplicar os pesos por uma máscara `torch.tril` de 1s e 0s pra zerar acima da diagonal e renormalizar cada linha dividindo pela soma. (2) O mais eficiente: antes do softmax, preencher as pontuações acima da diagonal com −∞ (`torch.triu(..., diagonal=1)` + `masked_fill`); como e^(−∞) = 0, o softmax já devolve zero ali e as linhas já somam 1, sem renormalizar.

**Relação:** Usa a propriedade do softmax (termo 9). Pegadinha que eu anotei (observação minha, não está no livro): no jeito (2) as pontuações recebem −∞, não zero; se recebessem zero, o softmax daria e^0 = 1 e o token futuro ainda ganharia peso. A renormalização do jeito (1) não causa vazamento de informação (termo 17).

**Exemplo:** Pra 6 tokens, `torch.tril(torch.ones(6, 6))` dá 1s na diagonal e abaixo, 0s acima. Os dois jeitos chegam no mesmo resultado: a linha de "journey" fica `[0.5517, 0.4483, 0, 0, 0, 0]`.

---

## 17. Information Leakage

**Equivalente:** Vazamento de informação.

**Definição:** Quando informação de tokens futuros (que deveriam estar mascarados) acaba influenciando o token atual.

**Função:** É a dúvida que o livro levanta sobre o jeito (1) da máscara: como os valores futuros entraram no denominador do primeiro softmax, parece que eles poderiam "vazar".

**Relação:** O livro mostra que não vaza: depois de mascarar e renormalizar, a distribuição fica como se o softmax tivesse sido calculado só nas posições não mascaradas desde o começo. É isso que garante que a atenção causal (termo 15) funcione de verdade.

**Exemplo:** Linha de "journey" depois de zerar: `[0.2041, 0.1659, 0, ...]`. Dividindo pela soma (0.3700) vira `[0.5517, 0.4483, 0, ...]`, exatamente o que o jeito (2), com −∞, entrega.

---

## 18. Dropout

**Equivalente:** Dropout (termo mantido em inglês; descarte aleatório de unidades).

**Definição:** Técnica de deep learning em que unidades escolhidas aleatoriamente são ignoradas (zeradas) durante o treino. É usada só no treino e desligada depois.

**Função:** Reduz overfitting, fazendo o modelo não depender demais de um conjunto específico de unidades. Na atenção dos GPT, costuma ser aplicado depois de calcular os pesos de atenção ou depois de multiplicá-los pelos values; o livro usa o primeiro, que é o mais comum.

**Relação:** Atua sobre os pesos de atenção já mascarados (termo 16), como uma segunda máscara, aleatória (figura 3.22). Os valores que sobram são escalados por 1/(1 − taxa), pra manter o equilíbrio geral dos pesos e a influência média da atenção igual no treino e na inferência.

**Exemplo:** `torch.nn.Dropout(0.5)` numa matriz 6×6 de 1s: mais ou menos metade vira 0 e o resto vira 2 (1/0.5). Os 50% são só didáticos; no treino do GPT o livro vai usar 0.1 ou 0.2. Na `CausalAttention`: `attn_weights = self.dropout(attn_weights)`.

---

## 19. register_buffer

**Equivalente:** N/A — nome próprio (método do `nn.Module` do PyTorch; em português, "registrar buffer").

**Definição:** Método usado no `__init__` da `CausalAttention` pra guardar a máscara causal dentro do módulo, como buffer.

**Função:** Buffers são movidos automaticamente pro dispositivo certo (CPU ou GPU) junto com o modelo, então não preciso garantir na mão que a máscara está no mesmo device dos parâmetros, o que evita erro de *device mismatch*. Não é estritamente necessário, mas é vantajoso.

**Relação:** Guarda a máscara do termo 16, criada uma vez com tamanho context_length × context_length. No forward ela é cortada pro tamanho real da entrada com `[:num_tokens, :num_tokens]`.

**Exemplo:** `self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))` e depois `attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)`. O `_` no fim de `masked_fill_` indica operação in-place, sem cópia extra de memória.

---

## 20. Multi-Head Attention

**Equivalente:** Atenção multi-cabeça.

**Definição:** Divisão do mecanismo de atenção em várias "cabeças", cada uma operando de forma independente. Na prática: rodar a atenção várias vezes, em paralelo, com projeções lineares aprendidas diferentes, e combinar as saídas.

**Função:** Cada cabeça aprende aspectos diferentes dos dados, e o modelo passa a atender ao mesmo tempo a informações de diferentes subespaços de representação em diferentes posições, o que melhora o desempenho em tarefas complexas. É a classe de atenção final, a que vai entrar na LLM.

**Relação:** Uma `CausalAttention` sozinha é *single-head* (um só conjunto de pesos de atenção). O livro implementa a versão multi-cabeça de dois jeitos: empilhando (termo 22) e dividindo pesos (termo 23).

**Exemplo:** GPT-2 menor (117 milhões de parâmetros): 12 cabeças e embedding de contexto de tamanho 768. GPT-2 maior (1,5 bilhão): 25 cabeças e embedding de tamanho 1.600. Nos GPT, d_in = d_out.

---

## 21. Attention Head e head_dim

**Equivalente:** Cabeça de atenção e dimensão por cabeça.

**Definição:** Cabeça é cada instância de atenção que roda de forma independente dentro do módulo multi-cabeça. head_dim é o tamanho da fatia de d_out que cada cabeça recebe: head_dim = d_out / num_heads.

**Função:** Na `MultiHeadAttention`, define como d_out é repartido entre as cabeças. Por isso tem o `assert d_out % num_heads == 0`: se não for divisível, não dá pra dividir.

**Relação:** O significado de d_out muda entre as duas implementações. No Wrapper (termo 22), d_out é por cabeça e a saída final tem d_out × num_heads; na `MultiHeadAttention` (termo 23), d_out já é o total e cada cabeça fica com head_dim.

**Exemplo:** No exemplo do livro da `MultiHeadAttention`, d_out=2 e num_heads=2, então head_dim = 1. No GPT-2 menor, 768 / 12 = 64 (conta minha, a partir dos números do livro).

---

## 22. MultiHeadAttentionWrapper

**Equivalente:** N/A — nome próprio (classe do livro; em português, "invólucro" de atenção multi-cabeça).

**Definição:** Classe que implementa multi-head attention empilhando várias instâncias de `CausalAttention`, cada uma com seus próprios pesos, e concatenando as saídas.

**Função:** É a forma intuitiva, pra entender o conceito. O problema é que as cabeças são processadas em sequência (`[head(x) for head in self.heads]`) e a projeção de query, key e value é repetida pra cada cabeça.

**Relação:** Cada cabeça tem suas próprias Wq, Wk e Wv (Wv1 e Wv2 na figura 3.24). A versão eficiente é a `MultiHeadAttention` (termo 23). No exercício 3.2, pra ter saída 2D mantendo num_heads=2, é só passar d_out=1, sem mexer na classe (resposta minha).

**Exemplo:** Com num_heads=2 e d_out=2 por cabeça, `torch.cat(..., dim=-1)` gera vetores de contexto de 4 dimensões; com o batch de 2 textos de 6 tokens, o shape é `torch.Size([2, 6, 4])`.

---

## 23. Weight Splits (MultiHeadAttention)

**Equivalente:** Divisão de pesos (classe `MultiHeadAttention`).

**Definição:** Jeito eficiente de fazer multi-head attention numa classe só: usa uma matriz Wq maior (idem Wk e Wv), faz uma única multiplicação pra obter Q e depois divide Q em Q1, Q2, ... reorganizando o tensor.

**Função:** Evita repetir, pra cada cabeça, a multiplicação de matriz das projeções, que é um dos passos mais caros computacionalmente. É a classe que vai ser usada na LLM.

**Relação:** Implementa o mesmo conceito do Wrapper (termo 22), só que integrado. A divisão usa `.view`: (b, num_tokens, d_out) → (b, num_tokens, num_heads, head_dim), e `.transpose(1, 2)` → (b, num_heads, num_tokens, head_dim), pra calcular todas as cabeças em lote (termo 24). No fim faz o caminho inverso (`.transpose(1, 2)`, `.contiguous().view`) até (b, num_tokens, d_out) e passa pela `out_proj` (termo 25). As saídas numéricas das duas classes no livro são diferentes; o que elas têm em comum é o conceito.

**Exemplo:** `MultiHeadAttention(d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2)` no batch dá shape `torch.Size([2, 6, 2])`: quem controla a dimensão de saída é d_out, não num_heads.

---

## 24. Batched Matrix Multiplication

**Equivalente:** Multiplicação de matrizes em lote.

**Definição:** Quando o operador `@` recebe tensores com mais de duas dimensões, o PyTorch multiplica as duas últimas dimensões e repete isso pra cada índice das dimensões da frente (batch e cabeças).

**Função:** Permite calcular as pontuações de todas as cabeças numa operação só, em vez de um loop por cabeça. Na `MultiHeadAttention`: `attn_scores = queries @ keys.transpose(2, 3)`.

**Relação:** Só funciona porque o `.transpose(1, 2)` do termo 23 coloca num_heads antes de num_tokens. É a mesma ideia do `inputs @ inputs.T` que substituiu os loops `for` no self-attention simplificado (termo 7).

**Exemplo:** Um tensor `a` de shape (1, 2, 3, 4) = (b, num_heads, num_tokens, head_dim). `a @ a.transpose(2, 3)` dá o mesmo resultado que fazer `first_head @ first_head.T` e `second_head @ second_head.T` separadamente; a primeira linha da primeira cabeça é `[1.3208, 1.1631, 1.2879]` nos dois casos.

---

## 25. Output Projection Layer

**Equivalente:** Camada de projeção de saída.

**Definição:** Camada linear (`self.out_proj = nn.Linear(d_out, d_out)`) aplicada depois de juntar as saídas de todas as cabeças.

**Função:** Combina as saídas das cabeças. O livro diz que ela não é estritamente necessária, mas é comum em muitas arquiteturas de LLM, por isso foi incluída.

**Relação:** Existe na `MultiHeadAttention` (termo 23), mas não na `CausalAttention` nem no Wrapper. Não muda o shape: entra (b, num_tokens, d_out) e sai (b, num_tokens, d_out).

**Exemplo:** Última linha do forward antes do `return`: `context_vec = self.out_proj(context_vec)`, logo depois do `.contiguous().view(b, num_tokens, self.d_out)`.
