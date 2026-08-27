# Sprint 2 — Frente B: análise

**Do Token ID ao lote.** Responde às questões 5 a 10 da seção 5 do enunciado da Sprint 2, a partir dos números medidos em [`experimentos/sprint02/resultados/frente-b.md`](../../experimentos/sprint02/resultados/frente-b.md).

Corpus: o mesmo da Frente A — _The Verdict_, 20 479 caracteres, 5 145 Token IDs pelo BPE do GPT-2.

---

## O que foi implementado

| Etapa                          | Onde                              | O que faz                                                        |
| ------------------------------- | ---------------------------------- | ------------------------------------------------------------------ |
| 4.1a Janela deslizante          | `src/dataset/sequences.py`        | Recorta uma lista de Token IDs em pares (entrada, alvo), em Python puro |
| 4.1b Dataset e lotes            | `src/dataset/loader.py`           | Empacota os pares em tensores (`GPTDataset`) e monta os lotes (`create_dataloader`) |
| 4.2 Embedding de token          | `src/embeddings/token_embeddings.py` | Troca cada Token ID por um vetor treinável                       |
| 4.3 Embedding posicional        | `src/embeddings/positional.py`    | Soma a posição ao vetor do token (`PositionalEmbedding`, `InputEmbedding`) |

Os testes em `tests/test_dataset.py` (15) e `tests/test_embeddings.py` (11) — 26 no total, todos passando — fixam cada afirmação feita abaixo.

### Uma decisão que diverge do livro

**Pedaço incompleto no fim, e texto curto demais, não são erro.** O livro não trata explicitamente o caso em que o texto não tem Token IDs suficientes para uma janela. Aqui a regra é explícita e vale nos dois extremos: uma janela final que não tem espaço para um alvo completo é descartada, nunca preenchida com um token de padding artificial — um alvo de padding ensinaria o modelo a prever algo que nunca foi linguagem. No limite, se não sobrar nenhuma janela completa, o resultado é um dataset de tamanho zero, não uma exceção. `expected_sample_count` (`src/dataset/sequences.py`) é a mesma função que calcula o número de amostras nas tabelas abaixo e que decide quantas janelas `GPTDataset` de fato constrói — não há duas fórmulas para divergir.

---

## Questão 5 — Qual é a função dos embeddings?

Fechar a lacuna que a Frente A deixou em aberto na questão 4: o Token ID é um rótulo nominal, e nenhuma conta feita diretamente sobre ele — soma, distância, ordem — significa nada. O embedding troca esse rótulo por um vetor de `emb_dim` posições, e é sobre esse vetor, não sobre o ID, que o resto do modelo opera.

A tabela do E3 deixa a mecânica visível: a camada de embedding é literalmente uma tabela com `vocab_size` linhas, uma por entrada do vocabulário, e `emb_dim` colunas. Para `emb_dim=256` sobre o vocabulário do BPE, essa tabela tem 12 865 792 parâmetros treináveis — mais que o dobro do vocabulário próprio inteiro (294 656, para o mesmo `emb_dim`) só porque o BPE tem 50 257 entradas contra 1 151. O ID de um token não faz nada além de escolher qual dessas linhas ler; o vetor em si começa aleatório (por isso a seed fixa em 42 é obrigatória para qualquer tabela deste relatório) e é ajustado pelo treino.

O E4 prova a consequência direta dessa mecânica: a mesma linha da tabela sai idêntica sempre que o mesmo ID aparece. O Token ID 1807 (`" thought"`), que se repete nas posições 4 e 66 dos primeiros 128 tokens de _The Verdict_, produz vetores de token com distância **0,000000** entre si — é a mesma linha da mesma tabela, lida duas vezes. A função do embedding, isolada da posição, é essa: dar a cada token um ponto num espaço vetorial de `emb_dim` dimensões que o treino pode mover livremente, ao contrário do índice fixo e arbitrário que o vocabulário atribuiu.

---

## Questão 6 — Por que é necessário representar a posição dos tokens?

Porque a tabela de embeddings, sozinha, não sabe em que posição da sequência um token apareceu — ela é indexada só pelo ID. A camada de atenção da Sprint 3 processa todas as posições de uma janela em paralelo, sem recorrência e sem convolução; nada no caminho até ali carrega informação de ordem, a não ser que alguma etapa a insira de propósito.

O E4 mede exatamente essa lacuna e o que a corrige. As duas ocorrências do Token ID 1807 têm distância 0,000000 entre os vetores de token — mostrado na questão 5 — mas, depois de somar o embedding posicional (posições 4 e 66), a distância entre os dois vetores finais sobe para **23,141924**. Nada mudou no conteúdo: o ID é o mesmo, o vetor de token é o mesmo. O que mudou foi exclusivamente a posição, e é essa mudança isolada que aparece na distância entre os vetores finais.

O custo dessa correção é baixo e não escala com o vocabulário. Na E3, a tabela posicional tem `context_length` linhas — 128 no caso medido — contra as `vocab_size` linhas da tabela de tokens: para `emb_dim=256`, isso é 32 768 parâmetros posicionais contra 12 865 792 de tokens no vocabulário do BPE, e o mesmo 32 768 contra 294 656 no vocabulário próprio. A tabela posicional custa a mesma coisa nos dois casos, porque seu tamanho depende de quantas posições a janela tem, não de quantos tokens o modelo reconhece.

---

## Questão 7 — Qual é a relação entre tamanho do contexto e quantidade de amostras de treinamento?

Inversa quando o stride acompanha o contexto, e quase constante quando o stride fica fixo em 1 — a relação depende dos dois parâmetros juntos, não do contexto isolado. O E1 varre as duas situações sobre o mesmo corpus de 5 145 Token IDs.

Com `stride == context_length` (sem sobreposição), o número de amostras cai de 643 (`context_length=8`) para 20 (`context_length=256`) — contextos maiores, ao repartir o mesmo corpus em janelas maiores, produzem proporcionalmente menos amostras por época. Já com `stride=1` (sobreposição máxima), o número de amostras quase não se move com o contexto: 5 137 amostras em `context_length=8` contra 4 889 em `context_length=256`, sempre perto do total de tokens do corpus, porque a janela avança um token de cada vez independentemente do quão larga ela é.

O preço dessa sobreposição aparece na coluna de tokens vistos por época. Em `context_length=256, stride=1`, a tabela reporta 1 251 584 tokens vistos por época contra apenas 5 145 tokens reais no corpus — uma redundância de **~243×**: o mesmo texto é reapresentado ao modelo centenas de vezes por época, em janelas que diferem por um único token. No outro extremo (`stride == context_length`), essa coluna nunca passa de ~5 145 (o próprio tamanho do corpus, arredondado para baixo pelo tamanho da janela): cada token pertence a exatamente uma amostra. A fração de sobreposição registrada na tabela — 0,00 quando `stride == context_length`, subindo até 1,00 quando `stride=1` e `context_length=256` — é a métrica direta dessa troca entre quantidade de amostras e redundância de conteúdo.

---

## Questão 8 — Qual é o impacto da dimensão do embedding sobre as estruturas utilizadas pelo modelo?

Linear no número de parâmetros da tabela de tokens, e o fator que multiplica todo tensor de ativação do modelo daqui em diante. O E3 varia `emb_dim` de 16 a 768 (a dimensão do GPT-2 small, incluída de propósito para comparação) sobre os dois vocabulários.

O crescimento é exatamente linear porque os parâmetros de uma tabela de embedding são `vocab_size × emb_dim`: multiplicar `emb_dim` por 48× (de 16 para 768) multiplica os parâmetros da tabela de tokens por 48×, nos dois vocabulários — de 804 112 para 38 597 376 no BPE, de 18 416 para 883 968 no vocabulário próprio. A tabela posicional cresce na mesma proporção (2 048 para 98 304), porque também é `context_length × emb_dim`. Só que `emb_dim` não é o fator dominante: comparando os dois vocabulários no mesmo `emb_dim=768`, a tabela de tokens do BPE tem 38 597 376 parâmetros contra 883 968 do vocabulário próprio — uma razão de **43,7×**, que é exatamente a razão entre os tamanhos dos vocabulários (50 257 / 1 151 ≈ 43,66). `vocab_size` pesa mais que `emb_dim` no custo da tabela de tokens, embora os dois entrem multiplicando.

O impacto não para na tabela de embeddings. `emb_dim` é a última dimensão do tensor que atravessa o resto do modelo: no E5, o lote de shape `(8, 128)` vira `(8, 128, 256)` já na saída do embedding de entrada, e essa largura de 256 é a que a atenção, o feed-forward e a normalização da Sprint 3 em diante vão herdar sem alteração. Para a configuração de referência (`emb_dim=256`), o bloco de embeddings sozinho já soma 12 898 560 parâmetros treináveis — antes de qualquer camada de atenção existir.

---

## Questão 9 — Qual é a função do DataLoader no pipeline?

Agrupar amostras individuais `(entrada, alvo)` em lotes de tensores com shape fixo, `(batch_size, context_length)`, e decidir o que fazer quando as amostras da época não dividem o `batch_size` exatamente. O E2 fixa `context_length=128, stride=128` — 40 amostras na época, independente do `batch_size` — e varia só o tamanho do lote.

Com `batch_size` em 1, 2, 4 e 8, o `DataLoader` produz 40, 20, 10 e 5 lotes por época respectivamente, sem descartar nenhuma amostra: 40 é múltiplo de todos esses valores. A partir de `batch_size=16`, isso muda: 40 ÷ 16 = 2 lotes completos, sobrando 8 amostras que `drop_last=True` descarta em vez de formar um lote menor; o mesmo acontece em `batch_size=32`, com 1 lote completo e 8 amostras descartadas. O `DataLoader` não completa o lote parcial com dado nenhum — ele simplesmente o joga fora, e é essa política que `drop_last` nomeia.

A tabela também deixa explícito que o shape do lote não depende de quantas amostras sobram: `(batch_size, context_length)` vale para todo `batch_size` testado, e o E2 confirma isso contra um `DataLoader` real, não só contra a fórmula — o primeiro lote medido com `batch_size=1` sai `(1, 128)`, exatamente como a tabela prevê. É esse tensor de shape fixa, não a lista de pares Python, que alimenta a camada de embedding.

---

## Questão 10 — Quais informações produzidas nesta Sprint serão utilizadas pelo mecanismo de atenção da Sprint seguinte?

O tensor de entrada final, e os parâmetros que passam a existir nele — nada além disso é passado adiante como código reutilizável, tudo o mais (Token IDs soltos, pares Python, tabelas intermediárias) fica para trás como etapa interna do pipeline.

O E5 rastreia a cadeia completa para a configuração de referência (`context_length=128, stride=128, batch_size=8, emb_dim=256`): do texto bruto (20 479 caracteres) aos 5 145 Token IDs do BPE, às 40 amostras que a janela deslizante produz, ao lote `(8, 128)` que o `DataLoader` entrega, até o tensor `(8, 128, 256)` que sai da soma entre o embedding de token e o embedding posicional. É esse último tensor — não o `(8, 128)` de Token IDs, que a Sprint 3 nunca vê diretamente — que entra na primeira camada de atenção.

Duas propriedades desse tensor importam mais que o valor numérico em si. A dimensão `context_length` (128 aqui) fixa o tamanho da janela sobre a qual a atenção vai calcular pontuações entre cada par de posições — é o parâmetro que, pela questão 1 da Frente A, controla o custo quadrático da atenção. A dimensão `emb_dim` (256 aqui) é a largura que atravessa sem mudança o resto da arquitetura, incluindo as matrizes de projeção Q/K/V que a Sprint 3 introduz. E os 12 898 560 parâmetros do bloco de embeddings, medidos no E5, são a base sobre a qual a contagem de parâmetros do modelo completo (Sprint 4) vai se acumular.

---

## Como reproduzir

```bash
python tests/test_dataset.py                          # 15 testes
python tests/test_embeddings.py                        # 11 testes
python experimentos/sprint02/frente_b_experimentos.py  # regenera os resultados
```

Há aleatoriedade nas tabelas de embeddings e a seed é fixada em 42 antes de cada bloco que sorteia — duas execuções produzem os mesmos parâmetros, shapes e contagens de amostras. A única coluna que pode divergir entre execuções é o tempo de forward medido no E3, por ruído de agendamento do sistema operacional; os demais números do arquivo são reproduzíveis byte a byte. O corpus é baixado automaticamente para `data/` na primeira execução, com verificação de tamanho.
