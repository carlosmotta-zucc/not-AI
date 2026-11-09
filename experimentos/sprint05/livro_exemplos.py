"""Exemplos do livro reproduzidos com o código do projeto (seções 5.1 e 5.2).

Cada exemplo do capítulo é refeito com src/ -- não com o código do livro -- e o
resultado vai ao lado do valor impresso, com a página. É o checkpoint de cada
etapa: se o projeto chega no mesmo número com a mesma seed, a etapa está
implementada como o livro descreve, até a ordem em que o gerador global é
consumido.

L1  5.1.1  Geração antes do treino
L2  5.1.2  A perda passo a passo, a cross-entropy e a perplexidade
L3  5.1.3  O conto, os loaders e as perdas iniciais de treino e validação
L4  5.2    O treino de 10 épocas: log, amostras, curvas e memorização

Os valores do livro vêm de tests/book_reference.py, a mesma fonte de
tests/test_book_checkpoints_ch5.py. O teste afirma até o passo 5 do treino;
este script publica o treino inteiro, que leva ~4 minutos em CPU.

Uso:
    python experimentos/sprint05/livro_exemplos.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comum import (  # noqa: E402
    DEVICE,
    INK_SECONDARY,
    RESULTS_DIR,
    SERIES_COLORS,
    Report,
    compact,
    legend,
    longest_shared_run,
    loss_curves,
    new_axes,
    save_figure,
    thousands,
    utf8_console,
)
from src.corpus import load_verdict  # noqa: E402
from src.generation import generate, text_to_token_ids, token_ids_to_text  # noqa: E402
from src.model import GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.training import (  # noqa: E402
    BOOK_MODEL_CONFIG,
    BOOK_TRAINING_CONFIG,
    batch_loss,
    create_train_val_loaders,
    evaluate,
    perplexity,
    split_text,
    target_log_probabilities,
    train_gpt,
)
from tests import book_reference as livro  # noqa: E402

RESULTS_PATH = RESULTS_DIR / "livro.md"

# Passadas pelo loader de treino antes de medir a perda inicial (L3): mostra
# que a 7ª casa depende de quais sequências caem juntas no mesmo lote.
PAIRING_PASSES = range(5)


class Checks:
    """Registro de cada comparação, para o resumo no topo do relatório."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool]] = []

    def add(self, section: str, example: str, ok: bool) -> str:
        self.rows.append((section, example, ok))
        return "sim" if ok else "**NÃO**"


def scientific_close(obtained: float, expected: float) -> bool:
    """Valores impressos como 7.4541e-05: meia unidade na 4ª casa da mantissa."""
    scale = 10.0 ** int(f"{expected:e}".split("e")[1])
    return abs(obtained - expected) <= 5e-5 * scale * (1 + 1e-6)


def same_text(obtained: str, expected: str) -> bool:
    """Compara ignorando a quantidade de espaços: o livro imprime as amostras
    quebradas em linhas e sem os espaços do fim."""
    return obtained.split() == expected.split()


def example_1_untrained_generation(report: Report, checks: Checks, bpe: BPETokenizer) -> GPTModel:
    """L1: o GPT de 124M recém-criado continua "Every effort moves you"."""
    report.write("## L1 - Seção 5.1.1: geração antes do treino (p. 130-132)")
    report.write()
    report.write(
        "**Exemplo do livro:** o GPT-2 small com contexto 256, criado com seed "
        "123 e posto em `eval()`, gera 10 tokens gulosos a partir de "
        f"`{livro.START_CONTEXT}`."
    )
    report.write()
    report.write(
        "**Como foi refeito:** `GPTModel(BOOK_MODEL_CONFIG)` e o `generate()` "
        "da Sprint 4 com temperatura 0 (a decodificação gulosa do "
        "`generate_text_simple`)."
    )
    report.write()

    torch.manual_seed(livro.BOOK_SEED)
    modelo = GPTModel(BOOK_MODEL_CONFIG).eval()

    ids = text_to_token_ids(livro.START_CONTEXT, bpe)
    saida = generate(modelo, ids, max_new_tokens=livro.UNTRAINED_NEW_TOKENS)
    texto = token_ids_to_text(saida, bpe)
    novos = saida[0, ids.shape[1]:].tolist()

    report.table(
        ["grandeza", "livro", "obtido", "confere"],
        [
            [
                "Token IDs do contexto",
                str(livro.START_CONTEXT_IDS),
                str(ids[0].tolist()),
                checks.add("5.1.1", "Token IDs do contexto", ids[0].tolist() == livro.START_CONTEXT_IDS),
            ],
            [
                "texto gerado",
                f"`{livro.UNTRAINED_OUTPUT_TEXT}`",
                f"`{texto}`",
                checks.add(
                    "5.1.1",
                    "texto gerado sem treino",
                    texto.replace(livro.UNPRINTABLE_IN_BOOK, "?") == livro.UNTRAINED_OUTPUT_TEXT,
                ),
            ],
        ],
    )
    report.write(
        f"Token IDs gerados: {novos}. A única diferença visível é `wasn?` no "
        f"livro contra `wasn{livro.UNPRINTABLE_IN_BOOK}` aqui: o modelo escolhe "
        "um token com a letra árabe mim (U+0645), que a impressão do livro "
        "mostra como `?`. Trocada essa letra por `?`, os textos coincidem "
        "letra a letra."
    )
    report.write()
    return modelo


def example_2_loss_step_by_step(report: Report, checks: Checks, bpe: BPETokenizer, modelo: GPTModel) -> None:
    """L2: os seis passos da figura 5.7 e o atalho do PyTorch."""
    report.write("## L2 - Seção 5.1.2: a perda passo a passo (p. 133-140)")
    report.write()
    report.write(
        "**Exemplo do livro:** dois textos de 3 tokens e seus alvos (a entrada "
        "deslocada de uma posição). O modelo sem treino dá probabilidades; a "
        "perda é a média de -log p(alvo) nas 6 posições; "
        "`F.cross_entropy` chega ao mesmo número; a perplexidade é o `exp` dela."
    )
    report.write()
    report.write(
        "**Como foi refeito:** o mesmo modelo da L1. O passo a passo usa "
        "`target_log_probabilities` (log_softmax + a coluna do alvo); o atalho, "
        "`batch_loss`, o `calc_loss_batch` do projeto."
    )
    report.write()

    entradas = torch.tensor(livro.LOSS_INPUTS)
    alvos = torch.tensor(livro.LOSS_TARGETS)
    with torch.no_grad():
        logits = modelo(entradas)
        perda_projeto = batch_loss(modelo, entradas, alvos)
    probabilidades = torch.softmax(logits, dim=-1)
    escolhidos = probabilidades.argmax(dim=-1)
    log_do_alvo = target_log_probabilities(logits, alvos)

    report.write("**Entrada, alvo e o que o modelo sem treino escolheria (argmax):**")
    report.write()
    report.table(
        ["texto", "entrada", "alvo", "argmax (livro)", "argmax (obtido)", "confere"],
        [
            [
                str(indice),
                f"`{bpe.decode(livro.LOSS_INPUTS[indice])}`",
                f"`{bpe.decode(livro.LOSS_TARGETS[indice])}`",
                str(livro.LOSS_ARGMAX_IDS[indice]),
                str(escolhidos[indice].tolist()),
                checks.add(
                    "5.1.2",
                    f"argmax do texto {indice}",
                    escolhidos[indice].tolist() == livro.LOSS_ARGMAX_IDS[indice],
                ),
            ]
            for indice in range(2)
        ],
    )
    saida_1 = token_ids_to_text(escolhidos[0], bpe)
    report.table(
        ["grandeza", "livro", "obtido", "confere"],
        [
            [
                "shape das probabilidades",
                str(livro.PROBAS_SHAPE),
                str(tuple(probabilidades.shape)),
                checks.add("5.1.2", "shape das probabilidades", tuple(probabilidades.shape) == livro.PROBAS_SHAPE),
            ],
            [
                "Targets batch 1",
                f"`{livro.TARGETS_TEXT_1}`",
                f"`{token_ids_to_text(alvos[0], bpe)}`",
                checks.add("5.1.2", "texto dos alvos", token_ids_to_text(alvos[0], bpe) == livro.TARGETS_TEXT_1),
            ],
            [
                "Outputs batch 1",
                f"`{livro.OUTPUTS_TEXT_1}`",
                f"`{saida_1}`",
                checks.add("5.1.2", "texto do argmax", saida_1 == livro.OUTPUTS_TEXT_1),
            ],
        ],
    )

    report.write("**Figura 5.7, passos 1 e 2 (primeira posição do primeiro texto):**")
    report.write()
    primeiros_logits = logits[0, 0, :3].tolist()
    primeiras_probs = probabilidades[0, 0, :3].tolist()
    report.table(
        ["passo", "livro", "obtido", "confere"],
        [
            [
                "1. logits",
                str(livro.FIRST_LOGITS),
                "[" + ", ".join(f"{v:.4f}" for v in primeiros_logits) + "]",
                checks.add(
                    "5.1.2",
                    "logits da figura 5.7",
                    all(abs(o - e) < livro.PRINT_TOLERANCE for o, e in zip(primeiros_logits, livro.FIRST_LOGITS)),
                ),
            ],
            [
                "2. probabilidades",
                "[" + ", ".join(f"{v:.4e}" for v in livro.FIRST_PROBAS) + "]",
                "[" + ", ".join(f"{v:.4e}" for v in primeiras_probs) + "]",
                checks.add(
                    "5.1.2",
                    "probabilidades da figura 5.7",
                    all(scientific_close(o, e) for o, e in zip(primeiras_probs, livro.FIRST_PROBAS)),
                ),
            ],
        ],
    )

    report.write("**Passos 3 e 4: a probabilidade de cada alvo e o seu log:**")
    report.write()
    linhas = []
    for indice in range(6):
        texto, posicao = divmod(indice, 3)
        p_obtida = log_do_alvo[texto, posicao].exp().item()
        p_livro = livro.TARGET_PROBAS[texto][posicao]
        log_obtido = log_do_alvo[texto, posicao].item()
        log_livro = livro.LOG_PROBAS[indice]
        ok = scientific_close(p_obtida, p_livro) and abs(log_obtido - log_livro) < livro.PRINT_TOLERANCE
        linhas.append(
            [
                str(texto),
                str(posicao),
                f"`{bpe.decode([livro.LOSS_TARGETS[texto][posicao]])}`",
                f"{p_livro:.4e}",
                f"{p_obtida:.4e}",
                f"{log_livro:.4f}",
                f"{log_obtido:.4f}",
                checks.add("5.1.2", f"alvo {texto}.{posicao}", ok),
            ]
        )
    report.table(
        ["texto", "posição", "alvo", "p(alvo) livro", "p(alvo) obtido", "log p livro", "log p obtido", "confere"],
        linhas,
    )

    media = log_do_alvo.mean().item()
    perda_pytorch = F.cross_entropy(logits.flatten(0, 1), alvos.flatten())
    perplexidade_float32 = torch.exp(perda_pytorch).item()
    perplexidade_projeto = perplexity(perda_pytorch)
    tol = livro.PRINT_TOLERANCE

    report.write("**Passos 5 e 6, e os atalhos:**")
    report.write()
    report.table(
        ["grandeza", "livro", "obtido", "confere"],
        [
            ["5. média dos log p", f"{livro.AVG_LOG_PROBA:.4f}", f"{media:.4f}",
             checks.add("5.1.2", "média dos log p", abs(media - livro.AVG_LOG_PROBA) < tol)],
            ["6. cross-entropy à mão (-média)", f"{livro.CROSS_ENTROPY:.4f}", f"{-media:.4f}",
             checks.add("5.1.2", "cross-entropy à mão", abs(-media - livro.CROSS_ENTROPY) < tol)],
            ["logits achatados", str(livro.FLAT_LOGITS_SHAPE), str(tuple(logits.flatten(0, 1).shape)),
             checks.add("5.1.2", "logits achatados", tuple(logits.flatten(0, 1).shape) == livro.FLAT_LOGITS_SHAPE)],
            ["alvos achatados", str(livro.FLAT_TARGETS_SHAPE), str(tuple(alvos.flatten().shape)),
             checks.add("5.1.2", "alvos achatados", tuple(alvos.flatten().shape) == livro.FLAT_TARGETS_SHAPE)],
            ["`F.cross_entropy`", f"{livro.CROSS_ENTROPY:.4f}", f"{perda_pytorch.item():.4f}",
             checks.add("5.1.2", "F.cross_entropy", abs(perda_pytorch.item() - livro.CROSS_ENTROPY) < tol)],
            ["`batch_loss` (projeto)", f"{livro.CROSS_ENTROPY:.4f}", f"{perda_projeto.item():.4f}",
             checks.add("5.1.2", "batch_loss", abs(perda_projeto.item() - livro.CROSS_ENTROPY) < tol)],
            ["perplexidade (`torch.exp`, float32)", f"{livro.PERPLEXITY:.4f}", f"{perplexidade_float32:.4f}",
             checks.add("5.1.2", "perplexidade", abs(perplexidade_float32 - livro.PERPLEXITY) < tol)],
            ["perplexidade (`perplexity`, float64)", f"{livro.PERPLEXITY:.4f}", f"{perplexidade_projeto:.4f}",
             "(mesma conta em float64)"],
        ],
    )
    report.write(
        f"O modelo sem treino hesita como se escolhesse entre {thousands(round(perplexidade_projeto))} "
        f"tokens ({perplexidade_projeto / len(bpe):.0%} do vocabulário de "
        f"{thousands(len(bpe))}). A perplexidade em float64 difere da do "
        f"livro em {abs(perplexidade_projeto - livro.PERPLEXITY):.4f}: o livro arredonda o "
        "`exp` em float32."
    )
    report.write()


def example_3_initial_losses(report: Report, checks: Checks, bpe: BPETokenizer, modelo: GPTModel) -> None:
    """L3: o conto, os loaders e a perda do modelo sem treino nos dois conjuntos."""
    report.write("## L3 - Seção 5.1.3: perdas de treino e validação (p. 140-145)")
    report.write()
    report.write(
        "**Exemplo do livro:** The Verdict dividido em 90% treino e 10% "
        "validação (nos caracteres), janelas de 256 tokens sem sobreposição, "
        "lotes de 2; a perda do modelo sem treino sobre os dois loaders."
    )
    report.write()
    report.write(
        "**Como foi refeito:** `create_train_val_loaders` (que monta os dois "
        "`DataLoader` da Sprint 2 com os padrões do livro) e `evaluate` "
        "(o `evaluate_model` do projeto, com os conjuntos inteiros)."
    )
    report.write()

    texto = load_verdict()
    treino_texto, validacao_texto = split_text(texto, livro.TRAIN_RATIO)
    treino, validacao = create_train_val_loaders(
        texto, bpe, context_length=livro.CH5_CONTEXT_LENGTH, batch_size=livro.LOADER_BATCH_SIZE
    )
    shapes = [tuple(x.shape) for x, _ in treino] + [tuple(x.shape) for x, _ in validacao]

    report.table(
        ["grandeza", "livro", "obtido", "confere"],
        [
            ["caracteres", thousands(livro.VERDICT_CHARS), thousands(len(texto)),
             checks.add("5.1.3", "caracteres", len(texto) == livro.VERDICT_CHARS)],
            ["tokens", thousands(livro.VERDICT_TOKENS), thousands(len(bpe.encode(texto))),
             checks.add("5.1.3", "tokens", len(bpe.encode(texto)) == livro.VERDICT_TOKENS)],
            ["lotes de treino", str(livro.TRAIN_BATCHES), str(len(treino)),
             checks.add("5.1.3", "lotes de treino", len(treino) == livro.TRAIN_BATCHES)],
            ["lotes de validação", str(livro.VAL_BATCHES), str(len(validacao)),
             checks.add("5.1.3", "lotes de validação", len(validacao) == livro.VAL_BATCHES)],
            ["shape de cada lote", str(livro.LOADER_BATCH_SHAPE), ", ".join(sorted({str(s) for s in shapes})),
             checks.add("5.1.3", "shape dos lotes", set(shapes) == {livro.LOADER_BATCH_SHAPE})],
        ],
    )
    report.write(
        f"O corte cai no caractere {thousands(len(treino_texto))}: {thousands(len(treino_texto))} "
        f"caracteres de treino ({thousands(len(treino.dataset.token_ids))} tokens, "
        f"{len(treino.dataset)} janelas) e {thousands(len(validacao_texto))} de validação "
        f"({thousands(len(validacao.dataset.token_ids))} tokens, {len(validacao.dataset)} janelas). "
        f"As duas partes somam {thousands(len(treino.dataset.token_ids) + len(validacao.dataset.token_ids))} "
        "tokens, um a mais que o conto inteiro: o corte partiu uma palavra em dois tokens."
    )
    report.write()

    # O mesmo ponto do gerador em que o livro itera o loader: logo depois da seed.
    torch.manual_seed(livro.BOOK_SEED)
    perda_treino, perda_validacao = evaluate(modelo, treino, validacao)
    report.table(
        ["perda inicial", "livro", "obtido", "diferença", "confere"],
        [
            ["treino", repr(livro.INITIAL_TRAIN_LOSS), repr(perda_treino),
             f"{perda_treino - livro.INITIAL_TRAIN_LOSS:.1e}",
             checks.add("5.1.3", "perda inicial de treino",
                        abs(perda_treino - livro.INITIAL_TRAIN_LOSS) < livro.INITIAL_LOSS_TOLERANCE)],
            ["validação", repr(livro.INITIAL_VAL_LOSS), repr(perda_validacao),
             f"{perda_validacao - livro.INITIAL_VAL_LOSS:.1e}",
             checks.add("5.1.3", "perda inicial de validação",
                        abs(perda_validacao - livro.INITIAL_VAL_LOSS) < livro.INITIAL_LOSS_TOLERANCE)],
        ],
    )

    report.write(
        "**A 7ª casa depende do pareamento.** A perda de treino é a média das "
        "9 perdas de lote, cada uma calculada em float32. Os pesos são os "
        "mesmos em todas as linhas abaixo; muda só quantas vezes o loader foi "
        "iterado depois da seed, isto é, quais sequências caíram juntas em "
        "cada lote:"
    )
    report.write()
    linhas = []
    for passadas in PAIRING_PASSES:
        torch.manual_seed(livro.BOOK_SEED)
        for _ in range(passadas):
            for _lote in treino:
                pass
        valor = evaluate(modelo, treino, validacao)[0]
        linhas.append([str(passadas), repr(valor), f"{valor - livro.INITIAL_TRAIN_LOSS:+.1e}"])
    report.table(["passadas antes da medida", "perda de treino", "diferença para o livro"], linhas)
    report.write(
        "Matematicamente a média não depende da ordem; em float32, cada média "
        "de lote arredonda em ~1e-6 e o total oscila na 7ª casa. É por isso que "
        "a comparação usa tolerância de 1e-6, e não igualdade."
    )
    report.write()


def example_4_training(report: Report, checks: Checks, bpe: BPETokenizer) -> None:
    """L4: o train_model_simple de 10 épocas, refeito com train_gpt."""
    report.write("## L4 - Seção 5.2: o treino de 10 épocas (p. 147-150)")
    report.write()
    report.write(
        "**Exemplo do livro:** `torch.manual_seed(123)`, um `GPTModel` novo, "
        "AdamW e o `train_model_simple` por 10 épocas, avaliando a cada 5 passos "
        "sobre 5 lotes e gerando uma amostra gulosa de 50 tokens no fim de cada "
        "época."
    )
    report.write()
    report.write(
        "**Como foi refeito:** `train_gpt(BOOK_MODEL_CONFIG, BOOK_TRAINING_CONFIG, ...)` "
        "-- os loaders da L3, a seed, o modelo, o `create_optimizer` e o "
        "`train_model` do projeto, nessa ordem. Em CPU."
    )
    report.write()

    cfg = BOOK_TRAINING_CONFIG
    hiperparametros = [
        ("taxa de aprendizado", livro.LEARNING_RATE, cfg.learning_rate),
        ("weight decay", livro.WEIGHT_DECAY, cfg.weight_decay),
        ("épocas", livro.NUM_EPOCHS, cfg.num_epochs),
        ("avaliação a cada (passos)", livro.EVAL_FREQ, cfg.eval_freq),
        ("lotes por avaliação", livro.EVAL_ITER, cfg.eval_iter),
    ]
    report.table(
        ["hiperparâmetro", "livro", "projeto", "confere"],
        [[nome, str(a), str(b), checks.add("5.2", nome, a == b)] for nome, a, b in hiperparametros],
    )

    texto = load_verdict()
    inicio = time.perf_counter()
    run = train_gpt(BOOK_MODEL_CONFIG, cfg, texto, bpe, device=DEVICE)
    total = time.perf_counter() - inicio
    historico = run.history
    passos_por_epoca = len(run.train_loader)

    report.write(
        f"**Log do treino** ({len(historico.steps)} avaliações; o livro imprime 6 "
        "e omite o resto com `[...]`):"
    )
    report.write()
    linhas = []
    for indice, passo in enumerate(historico.steps):
        treino_obtido = historico.train_losses[indice]
        validacao_obtida = historico.val_losses[indice]
        if passo in livro.TRAINING_LOG:
            epoca, treino_livro, validacao_livro = livro.TRAINING_LOG[passo]
            ok = (
                historico.epochs[indice] == epoca
                and abs(treino_obtido - treino_livro) < livro.LOSS_PRINT_TOLERANCE
                and abs(validacao_obtida - validacao_livro) < livro.LOSS_PRINT_TOLERANCE
            )
            livro_txt = f"{treino_livro:.3f} / {validacao_livro:.3f}"
            confere = checks.add("5.2", f"log do passo {passo}", ok)
        else:
            livro_txt, confere = "[...]", "-"
        linhas.append(
            [
                str(historico.epochs[indice]),
                f"{passo:06d}",
                thousands(historico.tokens_seen[indice]),
                f"{treino_obtido:.3f}",
                f"{validacao_obtida:.3f}",
                livro_txt,
                confere,
            ]
        )
    report.table(
        ["época", "passo", "tokens vistos", "treino", "validação", "livro (treino / validação)", "confere"],
        linhas,
    )

    report.write("**Amostra gulosa do fim de cada época** (50 tokens, quebras de linha como espaço):")
    report.write()
    linhas = []
    for epoca, amostra in enumerate(historico.samples, start=1):
        if epoca in livro.EPOCH_SAMPLES:
            confere = checks.add("5.2", f"amostra da época {epoca}", same_text(amostra, livro.EPOCH_SAMPLES[epoca]))
        else:
            confere = "-"
        linhas.append([str(epoca), compact(amostra), confere])
    report.table(["época", "amostra (obtida)", "igual à do livro"], linhas)

    report.write(
        f"Tempo: {total:.0f} s no total, "
        + ", ".join(f"{s:.1f}" for s in historico.epoch_seconds)
        + " s por época, das quais "
        + f"{sum(historico.monitor_seconds) / sum(historico.epoch_seconds):.0%} em avaliação e "
        "amostra. O livro cita \"cerca de 5 minutos num MacBook Air\"."
    )
    report.write()

    # Figura 5.12: as duas curvas, com os tokens vistos no eixo de cima.
    figura, eixo = new_axes(7.0, 3.8)
    loss_curves(eixo, historico, passos_por_epoca, SERIES_COLORS[0])
    eixo.set_ylim(bottom=0)
    eixo.set_title("Perdas de treino e validação (figura 5.12 do livro)", fontsize=10)
    legend(eixo, loc="upper right")
    topo = eixo.twiny()
    topo.plot(historico.tokens_seen, historico.train_losses, alpha=0)
    topo.set_xlabel("tokens vistos", color=INK_SECONDARY)
    topo.tick_params(colors=INK_SECONDARY, labelsize=9)
    for lado in ("top", "right"):
        topo.spines[lado].set_visible(False)
    caminho = save_figure(figura, "livro-perdas.png")
    report.figure(caminho, "Perdas de treino e validação do treino do livro")

    melhor = historico.best_evaluation()
    fim_epoca = [
        (epoca, historico.train_losses[i], historico.val_losses[i])
        for i, epoca in enumerate(historico.epochs)
        if i + 1 == len(historico.epochs) or historico.epochs[i + 1] != epoca
    ]
    report.write(
        "**Onde as curvas se separam.** Última avaliação de cada época "
        "(a tabela que a figura mostra):"
    )
    report.write()
    report.table(
        ["época", "treino", "validação", "diferença"],
        [[str(e), f"{t:.3f}", f"{v:.3f}", f"{v - t:.3f}"] for e, t, v in fim_epoca],
    )
    report.write(
        f"A menor perda de validação é {historico.val_losses[melhor]:.3f}, no passo "
        f"{historico.steps[melhor]} (época {historico.epochs[melhor]}); daí em diante a "
        "validação sobe enquanto o treino continua caindo. O livro descreve a "
        f"separação a partir da época {livro.DIVERGENCE_EPOCH}: é quando a diferença "
        "entre as duas passa a crescer a cada época."
    )
    report.write()

    # Memorização (p. 150): o trecho gerado está no conto, e nos 90% de treino.
    texto_treino, texto_validacao = split_text(texto, livro.TRAIN_RATIO)
    posicao = texto_treino.find(livro.MEMORIZED_SNIPPET)
    na_validacao = livro.MEMORIZED_SNIPPET in texto_validacao
    confere = checks.add("5.2", "trecho memorizado está no treino", posicao >= 0 and not na_validacao)
    report.write(
        f"**Memorização (p. 150).** O livro procura `{livro.MEMORIZED_SNIPPET}` no "
        f"conto: o trecho está no caractere {thousands(posicao)} do texto de treino, "
        f"{'e também' if na_validacao else 'e não'} na validação (confere: {confere})."
    )
    report.write()

    ids_treino = bpe.encode(texto_treino)
    ids_validacao = bpe.encode(texto_validacao)
    contexto = len(bpe.encode(livro.START_CONTEXT))
    linhas = []
    for epoca, amostra in enumerate(historico.samples, start=1):
        continuacao = bpe.encode(amostra)[contexto:]
        linhas.append(
            [
                str(epoca),
                str(longest_shared_run(continuacao, ids_treino)),
                str(longest_shared_run(continuacao, ids_validacao)),
            ]
        )
    report.write(
        "Maior trecho contíguo da continuação gerada (50 tokens) que aparece, "
        "token a token, no texto de treino e no de validação:"
    )
    report.write()
    report.table(["época", "copiado do treino (tokens)", "em comum com a validação (tokens)"], linhas)


def write_summary(report: Report, checks: Checks, position: int) -> None:
    secoes: dict[str, list[bool]] = {}
    for secao, _, ok in checks.rows:
        secoes.setdefault(secao, []).append(ok)

    total = len(checks.rows)
    conferem = sum(ok for _, _, ok in checks.rows)
    linhas = [
        "## Resumo",
        "",
        f"**{conferem} de {total} comparações conferem.**",
        "",
        "| seção do livro | comparações | conferem |",
        "| -------------- | ----------- | -------- |",
    ]
    linhas += [f"| {s} | {len(oks)} | {sum(oks)} |" for s, oks in secoes.items()]
    falhas = [f"- {secao}: {exemplo}" for secao, exemplo, ok in checks.rows if not ok]
    if falhas:
        linhas += ["", "Não conferem:", *falhas]
    linhas.append("")
    report.insert(position, linhas)


def main() -> None:
    utf8_console()
    report = Report(RESULTS_PATH)
    checks = Checks()
    bpe = BPETokenizer()

    report.write("# Sprint 5 - Exemplos do livro reproduzidos (seções 5.1 e 5.2)")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint05/livro_exemplos.py`. Não editar à mão: rode o script novamente."
    )
    report.write()
    report.write(
        "Cada exemplo das seções 5.1 e 5.2 foi refeito com o código de `src/`, "
        "com a seed do livro (123) e em CPU, e comparado com o valor impresso. "
        "Valores de 4 casas conferem com tolerância de meia unidade na última "
        "casa; perdas do log, na 3ª casa; Token IDs, contagens e textos, com "
        "igualdade exata (nas amostras de texto, ignorando a quantidade de "
        "espaços). Fonte dos valores do livro: `tests/book_reference.py`."
    )
    report.write()
    resumo = report.mark()

    modelo = example_1_untrained_generation(report, checks, bpe)
    example_2_loss_step_by_step(report, checks, bpe, modelo)
    example_3_initial_losses(report, checks, bpe, modelo)
    del modelo  # o treino da L4 cria o seu; libera ~650 MB antes dele
    example_4_training(report, checks, bpe)

    write_summary(report, checks, resumo)
    report.save()


if __name__ == "__main__":
    main()
