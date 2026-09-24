"""Heatmap de pesos de atencao, com os tokens do BPE nos eixos.

A matriz de pesos e (T, T) e some numa tabela de numeros: e olhando a figura que
"para onde este token olha" vira uma pergunta respondivel. Vive em
experimentos/ e nao em src/ porque so os experimentos plotam; se a Sprint 4
precisar, sobe para src/ com o uso ja conhecido.

Convencao das figuras: LINHA = quem olha (query), COLUNA = quem e olhado (key).
Trocar os dois e o erro classico de leitura, entao os eixos vao nomeados.
"""

from pathlib import Path
from typing import Sequence

import matplotlib

# Antes do pyplot: a maquina roda sem display, e o backend interativo padrao
# falharia na importacao.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

FIGURE_DPI = 150

# O BPE guarda o espaco dentro do token: ' bank' e 'bank' sao entradas
# diferentes do vocabulario. Sem marcador visivel os dois ficariam identicos no
# eixo, e a figura mentiria sobre o que foi tokenizado.
SPACE_MARKER = "␣"


def label_for(token: str) -> str:
    """Rotulo de eixo para um token do BPE, com o espaco a esquerda visivel."""
    return token.replace(" ", SPACE_MARKER).replace("\n", "\\n")


def attention_heatmap(
    weights: torch.Tensor,
    tokens: Sequence[str],
    title: str,
    path: Path,
    annotate: bool = True,
) -> Path:
    """Salva o heatmap de uma matriz de pesos de atencao.

    Args:
        weights: (T, T). Linha i sao os pesos da query i sobre todas as chaves,
            somando 1. Aceita tensor com gradiente -- e destacado aqui.
        tokens: T rotulos, na ordem da sequencia; de BPETokenizer.tokenize().
        title: titulo da figura.
        path: arquivo .png de destino; os diretorios sao criados.
        annotate: escreve o valor dentro de cada celula. So cabe ate ~12 tokens.

    Returns:
        O proprio `path`, para o chamador registrar no relatorio.

    Raises:
        ValueError: se `weights` nao for quadrada ou nao casar com `tokens`.
    """
    if weights.dim() != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"esperado (T, T), veio {tuple(weights.shape)}")
    if weights.shape[0] != len(tokens):
        raise ValueError(
            f"{len(tokens)} tokens para uma matriz {tuple(weights.shape)}"
        )

    matriz = weights.detach().cpu().numpy()
    rotulos = [label_for(token) for token in tokens]

    figura, eixo = plt.subplots(figsize=(1.0 + 0.8 * len(tokens), 0.8 * len(tokens)))

    # vmin/vmax fixos: duas figuras do mesmo experimento so sao comparaveis a
    # olho se a escala de cor nao for reajustada a cada uma.
    imagem = eixo.imshow(matriz, cmap="viridis", vmin=0.0, vmax=1.0)
    figura.colorbar(imagem, ax=eixo, label="peso de atencao")

    eixo.set_xticks(range(len(tokens)), rotulos, rotation=45, ha="right")
    eixo.set_yticks(range(len(tokens)), rotulos)
    eixo.set_xlabel("chave (quem e olhado)")
    eixo.set_ylabel("query (quem olha)")
    # Titulo em fonte menor e quebrado em linhas pelo chamador: a largura da
    # figura vem do numero de tokens, entao titulo longo em figura estreita sai
    # cortado.
    eixo.set_title(title, fontsize=10)

    if annotate:
        for linha in range(len(tokens)):
            for coluna in range(len(tokens)):
                valor = matriz[linha, coluna]
                # Texto claro sobre celula escura e vice-versa, senao os
                # numeros somem justamente onde o peso e alto.
                eixo.text(
                    coluna,
                    linha,
                    f"{valor:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if valor < 0.5 else "black",
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    figura.tight_layout()
    figura.savefig(path, dpi=FIGURE_DPI)
    plt.close(figura)

    return path
