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

Resultados correspondentes: [`experimentos/sprint02/resultados/frente-a.md`](../experimentos/sprint02/resultados/frente-a.md).
