# Sprint 5 — Pré-treinamento (capítulo 5)

**Do modelo que chuta ao modelo que prevê.** A Sprint 4 entregou um GPT que transforma Token IDs em logits; a Sprint 5 o treina. Uma função de perda mede o quanto o modelo aposta no token certo, e um laço ajusta os pesos para diminuí-la. Os experimentos mostram o que muda com épocas, taxa de aprendizado, lote e regularização, e em que momento o modelo deixa de aprender a língua e passa a decorar o texto. Por fim, o modelo treinado gera texto com cada estratégia de decodificação, e o treino vai para um arquivo do qual pode ser reconstruído e continuado.

```
Texto ─▶ DataLoaders de treino e validação (Sprint 2) ─▶ lotes (B, T)
      ─▶ GPTModel (Sprint 4) ─▶ logits (B, T, V) ─▶ cross-entropy ─▶ backward ─▶ AdamW ─▶ pesos novos
                                                         └─▶ perdas de treino e validação, perplexidade, amostra de texto
pesos treinados ─▶ save_checkpoint ─▶ .pth ─▶ load_checkpoint ─▶ generate (gulosa, temperatura, top-k) ─▶ texto
```

---

## Mapa das entregas

| Entregável | Código | Testes | Experimentos | Documento |
| ---------- | ------ | ------ | ------------ | --------- |
| 1 — Glossário | — | — | — | [`glossarios/`](../../glossarios/) (por integrante) |
| 2 — Avaliação do modelo | [`loss.py`](../../src/training/loss.py), [`data.py`](../../src/training/data.py) | [`test_training.py`](../../tests/test_training.py) | AV1–AV5, L2–L3 | [01-avaliacao-do-modelo.md](01-avaliacao-do-modelo.md) |
| 3 — Implementação do treinamento | [`trainer.py`](../../src/training/trainer.py), [`config.py`](../../src/training/config.py) | [`test_training.py`](../../tests/test_training.py) | T1, L4 | [02-treinamento.md](02-treinamento.md) |
| 4 — Experimentos de treinamento | [`experimentos/sprint05/`](../../experimentos/sprint05/) | — | T2–T8 | [03-experimentos-de-treinamento.md](03-experimentos-de-treinamento.md) |
| 5 — Estratégias de geração de texto | [`src/generation/`](../../src/generation/) (Sprint 4) | [`test_generation.py`](../../tests/test_generation.py) | D1–D5 | [04-estrategias-de-geracao.md](04-estrategias-de-geracao.md) |
| 6 — Salvamento e carregamento dos pesos | [`checkpoint.py`](../../src/training/checkpoint.py) | [`test_checkpoint.py`](../../tests/test_checkpoint.py) | C1–C4 | [05-salvamento-e-carregamento.md](05-salvamento-e-carregamento.md) |
| 7 — Documentação técnica | — | — | — | este README e os documentos 01 a 05 |
| Exemplos do livro (seções 5.1 e 5.2) | — | [`test_book_checkpoints_ch5.py`](../../tests/test_book_checkpoints_ch5.py) | L1–L4 | [`resultados/livro.md`](../../experimentos/sprint05/resultados/livro.md) |

---

## Organização do código

```
src/training/                 Entregáveis 2, 3 e 6
├── config.py                 TrainingConfig; BOOK_MODEL_CONFIG, BOOK_TRAINING_CONFIG, BASE_TRAINING_CONFIG
├── data.py                   split_text, create_train_val_loaders
├── loss.py                   target_log_probabilities, batch_loss, loader_loss, evaluate, perplexity
├── trainer.py                train_model (o laço), TrainingHistory, create_optimizer, train_gpt
└── checkpoint.py             save_checkpoint, load_checkpoint, Checkpoint

experimentos/sprint05/
├── comum.py                  relatório e estilo da Sprint 4, curvas de perda, medida de cópia
├── livro_exemplos.py         L1–L4 → resultados/livro.md         (~4 min: treina o GPT de 124M do livro)
├── avaliacao_experimentos.py AV1–AV5 → resultados/avaliacao.md  (~2 min)
├── treino_experimentos.py    T1–T8 → resultados/treino.md       (~22 min)
├── geracao_experimentos.py   D1–D5 → resultados/geracao.md      (~14 min)
├── checkpoint_experimentos.py C1–C4 → resultados/checkpoint.md  (~6 min; grava checkpoints/sprint05-base-10-epocas.pth)
└── resultados/figuras/
```

O pacote `training` usa `model`, `dataset` e `generation` (a amostra gulosa de cada época), e nenhum deles depende dele. A decodificação (gulosa, temperatura, top-k) continua em `src/generation/`, desde a Sprint 4: o Entregável 5 a usa sem mudar uma linha, e o que ele acrescenta são os experimentos sobre o modelo treinado.

---

## Configuração do treino

| | Projeto | Livro |
| - | ------- | ----- |
| Modelo | `BASE_CONFIG`: contexto 128, `emb_dim` 256, 4 cabeças, 4 blocos, dropout 0,1 — 28 920 832 parâmetros | `BOOK_MODEL_CONFIG`: GPT-2 small com contexto 256 — 162 419 712 parâmetros |
| Treino | `BASE_TRAINING_CONFIG` | `BOOK_TRAINING_CONFIG` |
| Otimizador | AdamW, lr 4·10⁻⁴, weight decay 0,1 | idem |
| Lotes, épocas | 2 sequências, 10 épocas | idem |
| Avaliação | a cada 5 passos, conjuntos inteiros | a cada 5 passos, 5 lotes |
| Seed | 42 | 123 |
| Dados | The Verdict: 90% treino / 10% validação, janelas sem sobreposição | idem |

Todo treino dos relatórios roda em **CPU**, a máquina de referência do projeto. É nela que o treino do livro é reproduzido número a número; em GPU, as perdas mudam a partir da 3ª ou 4ª casa.

---

## Principais resultados

- **O projeto reproduz o treino do livro número a número**: 45 de 45 comparações, das perdas iniciais nos 16 algarismos impressos às amostras de texto das épocas 9 e 10, com as 10 épocas refeitas em CPU (223 s).
- **O ponto de partida é previsível**: perda inicial de 10,98 ± 0,01 em 5 seeds, o valor esperado (ln V + σ²/2) para logits aleatórios com desvio 1/√3. É uma perplexidade de ~58 mil, acima do próprio vocabulário.
- **O modelo base aprende**: em 10 épocas, a perplexidade de validação cai de 54 888 para 464, e o acerto top-1 de validação chega a 12%. A melhor validação (6,106, na época 11) é praticamente a mesma do GPT de 124M do livro (6,134): o limite está nos 4 612 tokens de treino, e não no modelo.
- **O sobreajuste começa na época 3 e vira memorização**: na época 30, a perda de treino é 0,043, a de validação sobe para 7,067, e a amostra gerada é uma cópia de 50 tokens do conto.
- **A memorização tem endereço**: o modelo recita 32 de 32 tokens dos trechos do treino nas posições da janela em que os viu, e menos de 1 quando o mesmo trecho aparece deslocado meia janela.
- **Hiperparâmetros**: lr 10⁻³ chega ao mesmo mínimo com 40% menos passos que a do livro. Dropout e weight decay só adiam a memorização. Janelas sobrepostas dão a melhor validação de todos os experimentos (5,974). Trocar o trecho de validação muda a perda em 1,1 nat, 40 vezes o ruído entre seeds.
- **Custo**: um passo de treino leva 155 ms em CPU no modelo base e 1 808 ms no do livro, e a GPU é 4,7 a 7,7 vezes mais rápida. Avaliar os conjuntos inteiros a cada 5 passos consome 45% do tempo do treino.
- **A estratégia redistribui, o treino fornece**: sem treino, nenhuma estratégia gera um único par de tokens que exista no conto. Entre os três estados dos pesos (0, 10 e 30 épocas), o treino explica 62% da variação da coerência local, e a estratégia, 25%.
- **A cauda é o problema da amostragem**: o conto usa 1 360 dos 50 257 tokens, e os outros ficam com 14% da probabilidade depois de 10 épocas. A amostragem pura gera 7% de tokens que o modelo nunca viu; o top-k, com qualquer k até 200, nenhum. A temperatura troca repetição por ruído (de 0 a 2, a repetição da gulosa cai de 37,5% para 0, e a coerência local, de 90% para 0,2%); com top-k, ela vira ajuste fino.
- **A repetição da gulosa é de modelos meio treinados**: aparece com 10 épocas e some com 30, quando o modelo passa a recitar o conto. O sorteio reduz a cópia do modelo decorado de 50 para ~10 tokens, sem melhorar a perda de validação dele.
- **O arquivo devolve o treino bit a bit**: pesos, perdas e texto idênticos, também num processo novo. Retomado com otimizador e gerador, o treino é o mesmo do T1, peso a peso; sem o estado do AdamW, ou guardando só a seed, vira outro treino, tão bom quanto, mas não o mesmo. O estado do otimizador ocupa o dobro dos pesos (220 contra 110 MB).

---

## Como reproduzir

```bash
# testes da Sprint 5 (sem pytest)
python tests/test_training.py                # 21, poucos segundos
python tests/test_checkpoint.py              # 11, poucos segundos
python tests/test_book_checkpoints_ch5.py    # 6, ~30 s (treina o GPT de 124M por 6 passos, ~3 GB de RAM)

# experimentos — regeneram experimentos/sprint05/resultados/ e as figuras
python experimentos/sprint05/livro_exemplos.py
python experimentos/sprint05/avaliacao_experimentos.py
python experimentos/sprint05/treino_experimentos.py
python experimentos/sprint05/geracao_experimentos.py
python experimentos/sprint05/checkpoint_experimentos.py         # grava checkpoints/sprint05-base-10-epocas.pth
python experimentos/sprint05/checkpoint_experimentos.py --gerar # só carrega o arquivo e gera texto
# --rapido em treino_, geracao_ e checkpoint_experimentos: só confere que o script roda (resultados numa pasta temporária)
```

Seeds fixas: duas execuções produzem as mesmas tabelas, exceto pelas colunas de tempo. Os tempos de `geracao.md` e `checkpoint.md` são de outra máquina (Intel Core 5 220U), em que o treino de referência leva ~2 minutos em vez de 1.

---

## Como treinar o modelo em outro código

```python
from src.corpus import load_verdict
from src.model import BASE_CONFIG
from src.tokenizer import BPETokenizer
from src.training import BASE_TRAINING_CONFIG, train_gpt

run = train_gpt(BASE_CONFIG, BASE_TRAINING_CONFIG, load_verdict(), BPETokenizer())
run.model        # o GPTModel treinado (o mesmo do T1)
run.optimizer    # o AdamW, com os momentos m e v de cada peso
run.history      # perdas, amostras e tempos; run.history.to_dict() vira JSON
```

Para continuar um treino, chame `train_model(..., history=run.history)` com o **mesmo** otimizador, ou um com o estado restaurado; passos, tokens e épocas seguem a contagem. Para a continuação ser idêntica à de um treino sem pausa, o estado do gerador global (`torch.get_rng_state()`) também precisa ser guardado: é dele que saem a ordem dos lotes e as máscaras de dropout.

## Como salvar, carregar e gerar texto

```python
import torch
from src.generation import generate_text
from src.training import load_checkpoint, save_checkpoint

save_checkpoint("checkpoints/modelo.pth", run.model, BASE_TRAINING_CONFIG, run.history,
                optimizer=run.optimizer, rng_state=torch.get_rng_state())

ck = load_checkpoint("checkpoints/modelo.pth")        # restore_rng=True para retomar idêntico
texto = generate_text(ck.model, BPETokenizer(), "Every effort moves you", 50,
                      temperature=0.7, top_k=50, generator=torch.Generator().manual_seed(42))
```

O arquivo leva pesos, estado do AdamW, `asdict` das duas configurações, o histórico e o estado do gerador; `load_checkpoint` remonta modelo e otimizador sem que quem chama precise saber a arquitetura. Detalhes e a retomada do treino no [Entregável 6](05-salvamento-e-carregamento.md).
