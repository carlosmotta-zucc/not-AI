# Sprint 4 — Arquitetura GPT (capítulo 4)

**Dos mecanismos de atenção a um modelo que prevê o próximo token.** Os componentes da Sprint 3 são integrados com LayerNorm, FeedForward e conexões residuais no Transformer Block; os blocos são empilhados num GPT simplificado que produz logits para o vocabulário; e o modelo ganha inferência e geração autoregressiva de texto.

```
Texto ─▶ BPE (Sprint 2) ─▶ Token IDs ─▶ embeddings (Sprint 2)
      ─▶ n × [LayerNorm ─▶ Multi-Head Attention (Sprint 3) ─▶ +x ─▶ LayerNorm ─▶ FeedForward ─▶ +x]
      ─▶ LayerNorm final ─▶ cabeça de saída ─▶ logits ─▶ próximo token ─▶ (de volta ao contexto)
```

---

## Mapa das entregas

| Entregável | Código | Testes | Experimentos | Documento |
| ---------- | ------ | ------ | ------------ | --------- |
| 1 — Glossário | — | — | — | [`glossarios/`](../../glossarios/) (por integrante) |
| 2 — Transformer Block | [`src/transformer/`](../../src/transformer/) | [`test_transformer_components.py`](../../tests/test_transformer_components.py), [`test_transformer_block.py`](../../tests/test_transformer_block.py) | A1, A9 | [01-transformer-block.md](01-transformer-block.md) |
| 3 — Modelo GPT | [`src/model/`](../../src/model/) | [`test_gpt_model.py`](../../tests/test_gpt_model.py), [`test_gpt_pipeline.py`](../../tests/test_gpt_pipeline.py) | A1–A3 | [02-modelo-gpt.md](02-modelo-gpt.md) |
| 4 — Inferência e geração | [`src/generation/`](../../src/generation/) | [`test_generation.py`](../../tests/test_generation.py) | G1, G2, G5, G6 | [03-inferencia-e-geracao.md](03-inferencia-e-geracao.md) |
| 5 — Experimentação e análise | [`experimentos/sprint04/`](../../experimentos/sprint04/) | — | A1–A9, G1–G7 | [04-experimentos-e-analise.md](04-experimentos-e-analise.md) |
| Exemplos do livro | — | [`test_book_checkpoints.py`](../../tests/test_book_checkpoints.py) | L1–L8 | [`resultados/livro.md`](../../experimentos/sprint04/resultados/livro.md) |

O notebook [`notebooks/sprint04/01-gpt-passo-a-passo.ipynb`](../../notebooks/sprint04/01-gpt-passo-a-passo.ipynb) percorre o capítulo 4 na ordem do livro, com o código do projeto e os checkpoints de cada seção.

---

## Organização do código

```
src/
├── transformer/              Entregável 2
│   ├── layer_norm.py         LayerNorm (implementação própria)
│   ├── gelu.py               GELU, aproximação tanh e exata
│   ├── feed_forward.py       Linear → GELU → Linear, expansão 4x
│   ├── block.py              TransformerBlock (Pre-LayerNorm, dois atalhos)
│   └── shortcut_demo.py      rede de demonstração da seção 4.4 (didática)
├── model/                    Entregável 3
│   ├── config.py             GPTConfig, BASE_CONFIG, GPT_CONFIG_124M, GPT2_CONFIGS
│   └── gpt.py                GPTModel, contagem de parâmetros por componente
└── generation/               Entregável 4
    ├── decoding.py           logits → probabilidades → token (gulosa, temperatura, top-k)
    └── autoregressive.py     generate(): o laço autoregressivo; texto ↔ Token IDs

experimentos/sprint04/
├── relatorio.py              ferramentas comuns (relatório, tempo, estilo das figuras)
├── livro_exemplos.py         L1–L8 → resultados/livro.md
├── arquitetura_experimentos.py  A1–A9 → resultados/arquitetura.md
├── geracao_experimentos.py   G1–G7 → resultados/geracao.md
└── resultados/figuras/
```

Os pacotes seguem a ordem do pipeline e só dependem dos anteriores: `generation` usa `model`, `model` usa `transformer` e `embeddings`, `transformer` usa `attention`.

---

## Configuração base

| Hiperparâmetro | `BASE_CONFIG` | `GPT_CONFIG_124M` (livro) |
| -------------- | ------------- | ------------------------- |
| `vocab_size` | 50 257 | 50 257 |
| `context_length` | 128 | 1 024 |
| `emb_dim` | 256 | 768 |
| `n_heads` | 4 (`head_dim` 64) | 12 (`head_dim` 64) |
| `n_layers` | 4 | 12 |
| FeedForward | 1 024 | 3 072 |
| `drop_rate` | 0,1 | 0,1 |
| Parâmetros | 28 920 832 (110 MB) | 163 009 536 (622 MB) |

A `BASE_CONFIG` mantém a configuração das Sprints 2 e 3 e acrescenta 4 blocos; é o modelo que a Sprint 5 vai treinar. A `GPT_CONFIG_124M` reproduz o livro.

---

## Principais resultados

- **O projeto reproduz o livro número a número**: 66 de 66 comparações conferem, incluindo os logits do `GPTModel` de 124M com seed 123 e a geração `Hello, I am Featureiman Byeswickattribute argue`.
- **No modelo base, o custo está nas pontas**: tabela de tokens e cabeça de saída têm 89% dos parâmetros, e a cabeça faz 79% dos FLOPs de um forward.
- **Atalho e LayerNorm tornam a profundidade possível**: sem os dois, o gradiente do primeiro de 12 blocos é 13 ordens de grandeza menor que o do último; com o bloco Pre-LN, os 12 recebem o mesmo gradiente.
- **O termo quadrático da atenção ainda não domina** no intervalo de contexto do projeto: em T = 1 024, a atenção é 11,6% dos FLOPs, e os logits ocupam 3x a memória dos scores.
- **Geração exige `eval()`**: com dropout ligado, até a decodificação gulosa muda 28 de 30 tokens entre duas execuções.

---

## Como reproduzir

```bash
# testes da Sprint 4 (sem pytest)
python tests/test_transformer_components.py   # 12
python tests/test_transformer_block.py        # 9
python tests/test_gpt_model.py                # 14
python tests/test_generation.py               # 18
python tests/test_gpt_pipeline.py             # 4  (baixa o corpus na 1ª vez)
python tests/test_book_checkpoints.py         # 13 (monta o GPT de 124M, ~650 MB)

# experimentos — regeneram experimentos/sprint04/resultados/ e as figuras
python experimentos/sprint04/livro_exemplos.py
python experimentos/sprint04/arquitetura_experimentos.py
python experimentos/sprint04/geracao_experimentos.py
```

Seeds fixas: duas execuções produzem as mesmas tabelas, exceto pelas colunas de tempo.
