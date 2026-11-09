"""Ferramentas comuns aos scripts de experimento da Sprint 5.

O relatório em markdown, a formatação de números e o estilo das figuras são os
da Sprint 4 (experimentos/sprint04/relatorio.py), importados de lá: uma paleta
e um tratamento de eixos para o projeto inteiro. O que este módulo acrescenta:

- RESULTS_DIR, FIGURES_DIR e save_figure apontando para os resultados desta
  sprint.
- DEVICE: onde os experimentos treinam (a CPU; ver o comentário).
- loss_curves: as perdas de treino e validação ao longo do treino, no formato
  da figura 5.12 do livro.
- longest_shared_run: o maior trecho contíguo de um texto gerado que aparece,
  token a token, em outro texto -- a medida de memorização.

As regras dos relatórios continuam as da Sprint 4: nenhum número existe só no
gráfico, e toda figura tem, no relatório, a tabela com os valores que ela
mostra.
"""

import sys
from pathlib import Path

import torch

SPRINT04_DIR = Path(__file__).resolve().parent.parent / "sprint04"
sys.path.insert(0, str(SPRINT04_DIR))

# relatorio vem antes do pyplot: é ele que escolhe o backend sem tela (Agg).
from relatorio import (  # noqa: E402,F401 - reexportados para os scripts desta sprint
    BASELINE,
    GRIDLINE,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES_COLORS,
    SURFACE,
    FIGURE_DPI,
    Report,
    legend,
    megabytes,
    new_axes,
    percent,
    thousands,
)

import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"

# Seed dos experimentos do projeto, a mesma da Sprint 4. Os exemplos do livro
# usam a do livro (123), de tests/book_reference.py.
SEED = 42

# Todo treino dos relatórios roda na CPU. É a máquina de referência do projeto
# (requirements.txt instala o PyTorch de CPU), e é nela que o projeto reproduz
# os números do livro: em GPU, as rotinas numéricas e o gerador do dropout são
# outros, e as perdas mudam a partir da 3ª ou 4ª casa. A GPU, quando existe,
# aparece só no experimento de custo, como comparação.
DEVICE = torch.device("cpu")


def utf8_console() -> None:
    """Terminal em UTF-8. O console do Windows usa cp1252, e os textos gerados
    por um modelo sem treino trazem qualquer caractere do vocabulário -- o da
    seção 5.1.1 tem uma letra árabe. Sem isto, o print do relatório quebraria."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def save_figure(figura: plt.Figure, name: str) -> Path:
    """Salva em resultados/figuras/<name> desta sprint e fecha a figura."""
    path = FIGURES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figura.tight_layout()
    figura.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(figura)
    return path


def epochs_axis(steps: list[int], steps_per_epoch: int) -> list[float]:
    """Posição de cada avaliação em épocas: o passo 0 termina em 1/steps_per_epoch.

    O livro usa torch.linspace(0, num_epochs, len(perdas)), uma aproximação que
    põe a primeira avaliação na época 0; aqui é a posição exata.
    """
    return [(step + 1) / steps_per_epoch for step in steps]


def loss_curves(
    eixo: plt.Axes,
    history,
    steps_per_epoch: int,
    color: str,
    label: str | None = None,
    train: bool = True,
    val: bool = True,
) -> None:
    """Desenha as perdas avaliadas de um histórico: treino em linha cheia,
    validação tracejada (o mesmo par de estilos da figura 5.12)."""
    x = epochs_axis(history.steps, steps_per_epoch)
    if train:
        eixo.plot(
            x,
            history.train_losses,
            color=color,
            linewidth=1.6,
            label=f"{label} - treino" if label else "treino",
        )
    if val:
        eixo.plot(
            x,
            history.val_losses,
            color=color,
            linewidth=1.6,
            linestyle="--",
            label=f"{label} - validação" if label else "validação",
        )
    eixo.set_xlabel("épocas")
    eixo.set_ylabel("perda (cross-entropy, nats)")


def longest_shared_run(generated: list[int], reference: list[int]) -> int:
    """Tamanho, em tokens, do maior trecho contíguo de `generated` que aparece
    contíguo em `reference`.

    Args:
        generated: Token IDs gerados (sem o contexto de partida).
        reference: Token IDs do texto em que se procura a cópia.

    Returns:
        0 se nenhum token gerado aparece em `reference`; len(generated) se a
        continuação inteira é uma cópia literal.

    Programação dinâmica da maior substring comum: comp[j] é o tamanho do
    trecho em comum que termina no token atual de `generated` e no j-ésimo de
    `reference`. O(len(generated) · len(reference)), poucos milhares de
    operações por token -- sobra para 50 tokens contra o conto inteiro.
    """
    best = 0
    previous = [0] * (len(reference) + 1)
    for token in generated:
        current = [0] * (len(reference) + 1)
        for j, other in enumerate(reference, start=1):
            if token == other:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


def compact(text: str, limit: int | None = None) -> str:
    """Texto numa linha, para tabela: quebras viram espaço, `|` escapado e,
    com `limit`, cortado no último espaço antes do limite."""
    text = text.replace("\n", " ").replace("|", "\\|").strip()
    if limit is not None and len(text) > limit:
        corte = text.rfind(" ", 0, limit)
        text = text[: corte if corte > 0 else limit] + " ..."
    return text
