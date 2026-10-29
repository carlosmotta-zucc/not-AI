"""Ferramentas comuns aos três scripts de experimento da Sprint 4.

Na Sprint 3, `Report`, `thousands` e a medição de tempo viviam dentro do script
de experimentos. Com três scripts nesta sprint, passam a viver aqui, uma vez só:

- Report: acumula o relatório em markdown enquanto imprime no terminal.
- Formatação: milhares com espaço (padrão pt-BR) e tamanhos em MB.
- median_ms: tempo de execução robusto a picos do escalonador.
- Estilo das figuras: uma paleta e um tratamento de eixos para todas.

As figuras seguem a mesma regra dos relatórios: nenhum número existe só no
gráfico. Toda figura tem, no relatório, a tabela com os valores que ela mostra.
"""

import statistics
import time
from pathlib import Path
from typing import Callable

import matplotlib

# Antes do pyplot: a máquina roda sem display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"

SEED = 42

BYTES_PER_FLOAT32 = 4
BYTES_PER_MEGABYTE = 1024 * 1024

# Tempo: aquecimento descartado (alocação e caches da primeira chamada) e
# mediana, e não média, das medidas restantes. Os mesmos números da Sprint 3.
WARMUP = 3
REPEATS = 11

# Paleta categórica validada (azul, laranja, água, amarelo, magenta, nesta
# ordem), com a tinta e a superfície do mesmo sistema. A ordem é fixa: a série
# i sempre tem a cor i, em todas as figuras.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
FIGURE_DPI = 150


def thousands(value: int) -> str:
    """Inteiro com espaço como separador de milhar (padrão pt-BR)."""
    return f"{value:,}".replace(",", " ")


def megabytes(elements: int, bytes_per_element: int = BYTES_PER_FLOAT32) -> str:
    """Contagem de elementos float32 em MB legíveis."""
    return f"{elements * bytes_per_element / BYTES_PER_MEGABYTE:.2f} MB"


def percent(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f}%"


def median_ms(function: Callable[[], object]) -> float:
    """Mediana do tempo de `function()`, em milissegundos, sob no_grad.

    Mediana e não média: a CPU é compartilhada com o resto do sistema, e um
    pico do escalonador numa repetição levaria a média junto.
    """
    with torch.no_grad():
        for _ in range(WARMUP):
            function()

        tempos = []
        for _ in range(REPEATS):
            inicio = time.perf_counter()
            function()
            tempos.append((time.perf_counter() - inicio) * 1000)

    return statistics.median(tempos)


class Report:
    """Acumula o relatório em markdown enquanto imprime no terminal."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lines: list[str] = []

    def write(self, line: str = "") -> None:
        self._lines.append(line)
        print(line)

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Tabela markdown com colunas alinhadas no terminal."""
        widths = [
            max(len(headers[column]), *(len(row[column]) for row in rows))
            for column in range(len(headers))
        ]

        self.write("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
        self.write("| " + " | ".join("-" * w for w in widths) + " |")
        for row in rows:
            self.write("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
        self.write()

    def figure(self, path: Path, caption: str) -> None:
        """Referencia uma figura já salva, com caminho relativo ao relatório."""
        self.write(f"![{caption}]({path.relative_to(self.path.parent).as_posix()})")
        self.write()

    def mark(self) -> int:
        """Posição atual, para inserir um bloco depois (ex.: um resumo no topo)."""
        return len(self._lines)

    def insert(self, position: int, lines: list[str]) -> None:
        self._lines[position:position] = lines

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")


def new_axes(width: float = 7.0, height: float = 3.6) -> tuple[plt.Figure, plt.Axes]:
    """Figura com o tratamento comum: grade recessiva, sem moldura superior/direita."""
    figura, eixo = plt.subplots(figsize=(width, height))
    figura.patch.set_facecolor(SURFACE)
    eixo.set_facecolor(SURFACE)

    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color(BASELINE)

    eixo.tick_params(colors=INK_SECONDARY, labelsize=9)
    eixo.xaxis.label.set_color(INK_SECONDARY)
    eixo.yaxis.label.set_color(INK_SECONDARY)
    eixo.title.set_color(INK)

    # Grade em linha cheia e fina, atrás dos dados.
    eixo.grid(True, color=GRIDLINE, linewidth=0.6)
    eixo.set_axisbelow(True)
    return figura, eixo


def save_figure(figura: plt.Figure, name: str) -> Path:
    """Salva em resultados/figuras/<name> e fecha a figura."""
    path = FIGURES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figura.tight_layout()
    figura.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(figura)
    return path


def legend(eixo: plt.Axes, **kwargs) -> None:
    """Legenda sem moldura, com texto na tinta secundária."""
    caixa = eixo.legend(frameon=False, fontsize=9, **kwargs)
    for texto in caixa.get_texts():
        texto.set_color(INK_SECONDARY)
