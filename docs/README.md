# Documentação técnica

Um diretório por sprint. Cada documento **interpreta** resultados — não repete código nem tabela.

A divisão entre as três pastas de escrita do projeto:

| Pasta            | Conteúdo                                              | Gerado por        |
| ---------------- | ----------------------------------------------------- | ----------------- |
| `docs/`          | Análise técnica: o que os números significam          | Escrito à mão     |
| `experimentos/`  | Configuração dos experimentos e resultados brutos     | Script            |
| `glossarios/`    | Glossário cumulativo e individual, um por integrante  | Escrito à mão     |

Regra prática: nada em `experimentos/resultados/` é editado à mão — se um número precisa mudar, muda-se o script e ele é executado de novo. `docs/` é o oposto: só texto autoral, citando os números de lá.

## Índice

### Sprint 2 — Capítulo 2: dados de texto

| Documento                                                | Cobre                                                                   |
| -------------------------------------------------------- | ----------------------------------------------------------------------- |
| [`sprint02/frente-a-analise.md`](sprint02/frente-a-analise.md) | Tokenização, vocabulário, Token IDs, comparativo com BPE. Questões 1 a 4 |
| [`sprint02/frente-b-analise.md`](sprint02/frente-b-analise.md) | Janela deslizante, DataLoader, embeddings, positional embeddings. Questões 5 a 10 |

Resultados correspondentes: [`experimentos/sprint02/resultados/frente-a.md`](../experimentos/sprint02/resultados/frente-a.md) e [`experimentos/sprint02/resultados/frente-b.md`](../experimentos/sprint02/resultados/frente-b.md).

### Sprint 3 — Capítulo 3: atenção

| Documento                                        | Cobre                                                                                   |
| ------------------------------------------------ | --------------------------------------------------------------------------------------- |
| [`sprint03/analise.md`](sprint03/analise.md)     | Self-attention, escala por √d_k, custo em d e T, atenção causal, dropout, multi-head, entrega para a Sprint 4 |

Resultados correspondentes: [`experimentos/sprint03/resultados/atencao.md`](../experimentos/sprint03/resultados/atencao.md).
