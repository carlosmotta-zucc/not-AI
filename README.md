# not-AI — Construindo um LLM do zero

Projeto transversal (Projeto LLM) da disciplina **Inteligência Artificial e Sistemas Inteligentes** — Engenharia de Computação, UNOESC Joaçaba, 2026/2. Prof. Kleyton Hoffmann.

O objetivo é construir, de forma incremental ao longo do semestre, um modelo de linguagem completo **do zero**, seguindo o livro _Build a Large Language Model (From Scratch)_, de Sebastian Raschka.

Não é objetivo competir com modelos comerciais. O modelo é pequeno o suficiente para rodar no hardware disponível. O que se busca aqui é compreensão arquitetural, implementação autoral, experimentação e análise.

## Pipeline

```
Texto
  -> Tokenização
  -> Token IDs
  -> Embeddings
  -> Positional Embeddings
  -> Multi-Head Attention
  -> Transformer Blocks
  -> GPT
  -> Treinamento
  -> Geração de Texto
```

Cada etapa do pipeline corresponde a uma sprint. O que é implementado em uma sprint é reaproveitado nas seguintes — daí o código reutilizável viver em `src/`, e não dentro dos notebooks.

## Integrantes

| Integrante   | Pasta do glossário   |
| ------------ | -------------------- |
| Carlos Motta | `glossarios/carlos/` |
| João Popp    | `glossarios/joao/`   |

## Estado atual

| Sprint   | Capítulo   | Conteúdo                                                    | Status                        |
| -------- | ---------- | ----------------------------------------------------------- | ----------------------------- |
| Sprint 0 | Preparação | Ambiente, Git, PyTorch, estrutura do repositório            | Concluído                     |
| Sprint 1 | Cap. 1     | Introdução a LLMs, glossário, Quiz 1, mapa conceitual GPT   | Glossário do Carlos/João concluído |
| Sprint 2 | Cap. 2     | Tokenização, embeddings, positional embeddings, DataLoader  | Não iniciada                  |
| Sprint 3 | Cap. 3     | Self-Attention, Causal Attention, Multi-Head Attention      | Não iniciada                  |
| Sprint 4 | Cap. 4     | Arquitetura GPT, Transformer Block, LayerNorm, FFN, geração | Não iniciada                  |
| Sprint 5 | Cap. 5     | Treinamento, função de perda, otimizadores, avaliação       | Não iniciada                  |
| Sprint 6 | Cap. 6 e 7 | Fine-tuning, integração, apresentação                       | Não iniciada                  |

Ciclo de cada sprint: **Leitura → Glossário → Quiz → Implementação → Experimentação → Análise**.

## Estrutura do repositório

```
.
├── README.md               # este arquivo
├── CLAUDE.md               # convenções do projeto e instruções para assistentes de IA
├── requirements.txt        # dependências (mantido sempre atualizado)
├── src/                    # implementações reutilizáveis entre sprints
│   ├── tokenizer/
│   ├── embeddings/
│   ├── attention/
│   ├── transformer/
│   └── model/
├── notebooks/              # exploração e experimentos, um diretório por sprint
│   ├── sprint01/
│   └── ...
├── glossarios/             # glossário cumulativo, um diretório por integrante
│   ├── carlos/
│   └── joao/
├── experimentos/           # configs, logs e resultados
├── relatorios/             # documentação técnica de cada sprint
└── data/                   # corpora de treino (arquivos grandes não são versionados)
```

Diretórios ainda não criados aparecem acima porque definem o destino do código que virá; hoje o repositório contém apenas `glossarios/`.

## Como rodar

Requer **Python 3.11+**.

```bash
# clonar
git clone <url-do-repositorio>
cd not-AI

# ambiente virtual
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# dependências
pip install -r requirements.txt

# notebooks
jupyter notebook
```

Para verificar se o PyTorch enxerga a GPU (opcional — o projeto roda em CPU):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Convenções

Detalhadas em [CLAUDE.md](CLAUDE.md). Em resumo:

- **Implementação autoral.** Nada de cópia integral do livro ou de repositórios públicos: nomes de variáveis, estrutura e comentários são próprios.
- **Sem atalhos prontos.** `nn.MultiheadAttention`, `nn.Transformer` e equivalentes não substituem os componentes que a sprint pede para implementar — entram só como _baseline_ de comparação em experimentos.
- **Código em inglês, comentários e docstrings em português.** Toda função pública documenta o _shape_ de entrada e saída.
- **Comentar o porquê, não o quê.**
- **Seeds fixas** em qualquer experimento com aleatoriedade, para reprodutibilidade.
- **Hiperparâmetros centralizados** em dicionário de configuração ou `config.py`, nunca espalhados como números mágicos.
- **Um commit por etapa significativa.**

## Glossário

O glossário é cumulativo, individual e vale nota. Ele não é uma tradução de termos: registra o **significado** e a **função** de cada conceito dentro do modelo de linguagem.

Cada entrada traz definição, função no modelo, relação com outros conceitos e um exemplo — conceitual, matemático ou computacional.

- [Capítulo 1 — Carlos](glossarios/carlos/glossario-cap1-llm.md)

## Experimentos

Todo componente implementado tem um experimento associado, variando parâmetros, configuração ou dados, e observando o impacto sobre comportamento, desempenho ou custo computacional.

Cada experimento registra:

1. Hipótese ou pergunta
2. Configuração testada (valores exatos)
3. Resultado (números, gráficos)
4. Interpretação técnica, relacionada ao conceito estudado

Resultado sem interpretação não conta como análise.

## Avaliação

O Projeto LLM corresponde à nota A1/1, peso 4 na média semestral.

| Critério                                   | Peso |
| ------------------------------------------ | ---- |
| Quizzes individuais                        | 15%  |
| Implementações intermediárias e glossários | 25%  |
| Modelo LLM integrado                       | 25%  |
| Experimentação e análise dos resultados    | 15%  |
| Documentação técnica                       | 10%  |
| Apresentação e arguição individual         | 10%  |

A avaliação inclui **arguição individual**: cada integrante precisa saber explicar qualquer trecho do que foi entregue.

## Referência

RASCHKA, Sebastian. _Build a Large Language Model (From Scratch)_. Manning, 2024.
