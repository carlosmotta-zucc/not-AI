# Sprint 2 - Frente B: resultados dos experimentos

Gerado por `experimentos/sprint02/frente_b_experimentos.py`. Nao editar a mao: rode o script novamente.

Corpus: The Verdict, 20 479 caracteres.

Seed fixa em 42 no inicio e antes de cada bloco que sorteia (tabelas de embeddings) -- sem isso nenhuma tabela deste relatorio seria reproduzivel.

## E1 - Contexto e stride contra quantidade de amostras

**Pergunta:** o numero de amostras de treino cresce so com o contexto, ou o stride pesa tanto quanto? E quanto de sobreposicao cada combinacao paga?

**Configuracao:** The Verdict tokenizado pelo BPE, context_length em 8, 16, 32, 64, 128, 256, cada um com stride em {1, context_length/2, context_length}.

O corpus tem 5 145 Token IDs pelo BPE.

| context_length | stride | amostras | tokens vistos por epoca | fracao de sobreposicao |
| -------------- | ------ | -------- | ----------------------- | ---------------------- |
| 8              | 1      | 5 137    | 41 096                  | 0.88                   |
| 8              | 4      | 1 285    | 10 280                  | 0.50                   |
| 8              | 8      | 643      | 5 144                   | 0.00                   |
| 16             | 1      | 5 129    | 82 064                  | 0.94                   |
| 16             | 8      | 642      | 10 272                  | 0.50                   |
| 16             | 16     | 321      | 5 136                   | 0.00                   |
| 32             | 1      | 5 113    | 163 616                 | 0.97                   |
| 32             | 16     | 320      | 10 240                  | 0.50                   |
| 32             | 32     | 160      | 5 120                   | 0.00                   |
| 64             | 1      | 5 081    | 325 184                 | 0.98                   |
| 64             | 32     | 159      | 10 176                  | 0.50                   |
| 64             | 64     | 80       | 5 120                   | 0.00                   |
| 128            | 1      | 5 017    | 642 176                 | 0.99                   |
| 128            | 64     | 79       | 10 112                  | 0.50                   |
| 128            | 128    | 40       | 5 120                   | 0.00                   |
| 256            | 1      | 4 889    | 1 251 584               | 1.00                   |
| 256            | 128    | 39       | 9 984                   | 0.50                   |
| 256            | 256    | 20       | 5 120                   | 0.00                   |

## E2 - Batch size contra shape e numero de batches

**Pergunta:** quantos lotes uma epoca produz para cada batch_size, e quantas amostras drop_last descarta em cada caso?

**Configuracao:** The Verdict tokenizado pelo BPE, context_length=128, stride=128 (fixos), batch_size em 1, 2, 4, 8, 16, 32, drop_last=True.

Amostras na epoca (independe de batch_size): 40.

| batch_size | batches por epoca | shape do batch | amostras descartadas |
| ---------- | ----------------- | -------------- | -------------------- |
| 1          | 40                | (1, 128)       | 0                    |
| 2          | 20                | (2, 128)       | 0                    |
| 4          | 10                | (4, 128)       | 0                    |
| 8          | 5                 | (8, 128)       | 0                    |
| 16         | 2                 | (16, 128)      | 8                    |
| 32         | 1                 | (32, 128)      | 8                    |

Shape do primeiro lote real com batch_size=1: (1, 128) -- confere com a tabela.

## E3 - Dimensao do embedding contra custo

**Pergunta:** quanto custa, em parametros, memoria e tempo de forward, cada aumento de emb_dim? O custo depende mais da dimensao ou do tamanho do vocabulario?

**Configuracao:** emb_dim em 16, 32, 64, 128, 256, 768 (768 e a dimensao do GPT-2 small, incluida para comparacao), vocabulario do BPE (50 257 entradas) e vocabulario proprio (1 151 entradas), context_length=128, forward sobre lote (8, 128), seed 42.

Vocabulario do BPE:

| emb_dim | vocab_size | parametros (tokens) | parametros (posicoes) | memoria estimada | tempo de um forward |
| ------- | ---------- | ------------------- | --------------------- | ---------------- | ------------------- |
| 16      | 50 257     | 804 112             | 2 048                 | 3.1 MB           | 0.20 ms             |
| 32      | 50 257     | 1 608 224           | 4 096                 | 6.2 MB           | 0.09 ms             |
| 64      | 50 257     | 3 216 448           | 8 192                 | 12.3 MB          | 0.47 ms             |
| 128     | 50 257     | 6 432 896           | 16 384                | 24.6 MB          | 0.41 ms             |
| 256     | 50 257     | 12 865 792          | 32 768                | 49.2 MB          | 0.43 ms             |
| 768     | 50 257     | 38 597 376          | 98 304                | 147.6 MB         | 0.88 ms             |

Vocabulario proprio:

| emb_dim | vocab_size | parametros (tokens) | parametros (posicoes) | memoria estimada | tempo de um forward |
| ------- | ---------- | ------------------- | --------------------- | ---------------- | ------------------- |
| 16      | 1 151      | 18 416              | 2 048                 | 0.1 MB           | 0.11 ms             |
| 32      | 1 151      | 36 832              | 4 096                 | 0.2 MB           | 0.08 ms             |
| 64      | 1 151      | 73 664              | 8 192                 | 0.3 MB           | 0.22 ms             |
| 128     | 1 151      | 147 328             | 16 384                | 0.6 MB           | 0.13 ms             |
| 256     | 1 151      | 294 656             | 32 768                | 1.2 MB           | 0.15 ms             |
| 768     | 1 151      | 883 968             | 98 304                | 3.7 MB           | 0.87 ms             |

## E4 - Efeito da posicao

**Pergunta:** duas ocorrencias do mesmo Token ID tem o mesmo vetor de token? E o mesmo vetor final, depois da soma posicional?

**Configuracao:** primeiros 128 Token IDs de The Verdict pelo BPE, emb_dim=256, seed 42. Token ID 1807 (`' thought'`) se repete nas posicoes 4 e 66.

| vetor                               | distancia entre as duas ocorrencias |
| ----------------------------------- | ----------------------------------- |
| token embedding (antes da posicao)  | 0.000000                            |
| input embedding (depois da posicao) | 23.141924                           |

## E5 - Dimensoes ao longo do pipeline

**Pergunta:** os shapes fecham do texto bruto ate o tensor de entrada do modelo, para uma configuracao de referencia?

**Configuracao:** context_length=128, stride=128, batch_size=8, emb_dim=256, seed=42

| etapa                                 | objeto         | shape             |
| ------------------------------------- | -------------- | ----------------- |
| texto bruto                           | str            | 20 479 caracteres |
| tokenizacao (BPE)                     | list[int]      | 5 145 Token IDs   |
| janela deslizante                     | GPTDataset     | 40 amostras       |
| uma amostra                           | (x, y)         | (128,) cada       |
| lote                                  | Tensor long    | (8, 128)          |
| token embedding                       | Tensor float32 | (8, 128, 256)     |
| positional embedding                  | Tensor float32 | (128, 256)        |
| input embedding (entrada da Sprint 3) | Tensor float32 | (8, 128, 256)     |

Parametros treinaveis do bloco de embeddings: 12 898 560.

