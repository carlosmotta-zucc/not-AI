"""Exemplos do livro reproduzidos com o código do projeto (capítulo 4 e seção 5.3).

Cada exemplo do capítulo 4 é refeito com src/ -- não com o código do livro -- e
o resultado é posto ao lado do valor impresso, com a página. É o "checkpoint" de
cada etapa: se o projeto chega no mesmo número com a mesma seed, a etapa está
implementada como o livro descreve, até a ordem em que os pesos são sorteados.

L1  4.1  O lote de entrada e o esqueleto (DummyGPTModel)
L2  4.2  LayerNorm passo a passo
L3  4.3  GELU contra ReLU, e o shape dentro da FeedForward
L4  4.4  Gradiente por camada, com e sem atalho
L5  4.5  TransformerBlock e exercício 4.1
L6  4.6  GPTModel de 124M: logits, parâmetros, memória; exercícios 4.2 e 4.3
L7  4.7  Geração gulosa de "Hello, I am", iteração por iteração
L8  5.3  Gulosa, amostragem, temperatura e top-k no vocabulário de 9 palavras;
         exercícios 5.1 e 5.3

Os valores do livro vêm de tests/book_reference.py, a mesma fonte do teste
tests/test_book_checkpoints.py. O teste afirma; este script publica.

Uso:
    python experimentos/sprint04/livro_exemplos.py
"""

import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from relatorio import (  # noqa: E402
    INK_SECONDARY,
    RESULTS_DIR,
    SERIES_COLORS,
    SURFACE,
    Report,
    legend,
    new_axes,
    save_figure,
    thousands,
)
from src.generation import (  # noqa: E402
    apply_top_k,
    generate,
    next_token_probabilities,
    select_next_token,
    text_to_token_ids,
    token_ids_to_text,
)
from src.model import GPT2_CONFIGS, GPT_CONFIG_124M, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.transformer import (  # noqa: E402
    GELU,
    FeedForward,
    LayerNorm,
    ShortcutDemoNetwork,
    TransformerBlock,
    mean_weight_gradients,
)
from tests import book_reference as livro  # noqa: E402

RESULTS_PATH = RESULTS_DIR / "livro.md"


class Checks:
    """Registro de cada comparação, para o resumo no topo do relatório."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool]] = []

    def add(self, section: str, example: str, ok: bool) -> str:
        self.rows.append((section, example, ok))
        return "sim" if ok else "**NÃO**"


def vector(values) -> str:
    """Lista de floats com 4 casas, como o livro imprime."""
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


def close(obtained, expected, atol: float = livro.PRINT_TOLERANCE) -> bool:
    """Mesmo critério do teste: meia unidade na 4ª casa, mais a folga relativa
    do allclose. Sem a folga, um valor como 1,67545 -- que o livro imprime 1,6755
    e o Python formata 1,6754 -- cairia fora por um erro de arredondamento do
    float32, e não por uma diferença de conta."""
    return torch.allclose(
        torch.as_tensor(obtained, dtype=torch.float32),
        torch.as_tensor(expected, dtype=torch.float32),
        atol=atol,
    )


def logits_table(report: Report, checks: Checks, section: str, logits: torch.Tensor, esperado: list) -> None:
    """Compara as 8 linhas de logits (2 textos x 4 posições) com o livro."""
    linhas = []
    for exemplo, posicoes in enumerate(esperado):
        for posicao, (inicio, fim) in enumerate(posicoes):
            linha = logits[exemplo, posicao]
            obtido = torch.cat([linha[:3], linha[-3:]])
            diferenca = (obtido - torch.tensor(inicio + fim)).abs().max().item()
            ok = close(obtido, inicio + fim)
            linhas.append(
                [
                    str(exemplo),
                    str(posicao),
                    f"{vector(inicio)} ... {vector(fim)}",
                    f"{vector(linha[:3])} ... {vector(linha[-3:])}",
                    f"{diferenca:.1e}",
                    checks.add(section, f"logits, texto {exemplo}, posição {posicao}", ok),
                ]
            )

    report.table(
        ["texto", "posição", "livro (3 primeiros ... 3 últimos)", "obtido", "dif. máx.", "confere"],
        linhas,
    )


def example_1_skeleton(report: Report, checks: Checks, bpe: BPETokenizer) -> torch.Tensor:
    """L1: o lote de entrada e o esqueleto do modelo."""
    report.write("## L1 - Seção 4.1: o lote de entrada e o esqueleto (p. 97-99)")
    report.write()
    report.write(
        "**Exemplo do livro:** dois textos tokenizados pelo BPE do GPT-2 e "
        "empilhados num lote; um `DummyGPTModel` com seed 123 produz os logits."
    )
    report.write()
    report.write(
        "**Como foi refeito:** o `DummyGPTModel` é o `GPTModel` do projeto com "
        "`n_layers=0` e a `final_norm` trocada por identidade. Os blocos falsos "
        "do livro não têm pesos, então não sorteiam nada: a ordem dos sorteios "
        "(tabela de tokens, tabela de posições, cabeça de saída) é a mesma. A "
        "passada roda em `train()`, como no livro, com o dropout de 0,1 ligado."
    )
    report.write()

    lote = torch.tensor([bpe.encode(texto) for texto in livro.BATCH_TEXTS])
    report.table(
        ["texto", "livro", "obtido", "confere"],
        [
            [
                f"`{texto}`",
                str(esperado),
                str(obtido),
                checks.add("4.1", f"Token IDs de '{texto}'", obtido == esperado),
            ]
            for texto, esperado, obtido in zip(livro.BATCH_TEXTS, livro.BATCH_IDS, lote.tolist())
        ],
    )

    torch.manual_seed(livro.BOOK_SEED)
    esqueleto = GPTModel(replace(GPT_CONFIG_124M, n_layers=0))
    esqueleto.final_norm = nn.Identity()
    with torch.no_grad():
        logits = esqueleto(lote)

    report.write(f"Shape dos logits: `{tuple(logits.shape)}` (livro: `(2, 4, 50257)`).")
    report.write()
    logits_table(report, checks, "4.1", logits, livro.DUMMY_LOGITS)
    return lote


def example_2_layer_norm(report: Report, checks: Checks) -> None:
    """L2: a normalização feita à mão e pela classe."""
    report.write("## L2 - Seção 4.2: LayerNorm passo a passo (p. 100-104)")
    report.write()
    report.write(
        "**Exemplo do livro:** `torch.randn(2, 5)` passa por "
        "`nn.Sequential(nn.Linear(5, 6), nn.ReLU())`; a saída é normalizada à "
        "mão e depois pela classe `LayerNorm`."
    )
    report.write()

    torch.manual_seed(livro.BOOK_SEED)
    entrada = torch.randn(2, 5)
    camada = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    with torch.no_grad():
        saida = camada(entrada)
        media = saida.mean(dim=-1, keepdim=True)
        variancia = saida.var(dim=-1, keepdim=True)
        normalizada = (saida - media) / torch.sqrt(variancia)
        pela_classe = LayerNorm(emb_dim=5)(entrada)

    comparacoes = [
        ("saída da camada, linha 0", livro.LAYER_OUTPUT[0], saida[0]),
        ("saída da camada, linha 1", livro.LAYER_OUTPUT[1], saida[1]),
        ("média por linha", livro.LAYER_OUTPUT_MEAN, media.squeeze()),
        ("variância por linha", livro.LAYER_OUTPUT_VAR, variancia.squeeze()),
        ("normalizada, linha 0", livro.LAYER_OUTPUT_NORMALIZED[0], normalizada[0]),
        ("normalizada, linha 1", livro.LAYER_OUTPUT_NORMALIZED[1], normalizada[1]),
    ]
    report.table(
        ["grandeza", "livro", "obtido", "confere"],
        [
            [nome, vector(esperado), vector(obtido), checks.add("4.2", nome, close(obtido, esperado))]
            for nome, esperado, obtido in comparacoes
        ],
    )

    media_classe = pela_classe.mean(dim=-1)
    variancia_classe = pela_classe.var(dim=-1, unbiased=False)
    ok = close(media_classe, [0.0, 0.0], 1e-6) and close(variancia_classe, [1.0, 1.0], 1e-4)
    report.write(
        f"**Classe `LayerNorm(emb_dim=5)` sobre a entrada:** média "
        f"{vector(media_classe)}, variância {vector(variancia_classe)} -- o livro "
        f"mostra 0 e 1. Confere: {checks.add('4.2', 'LayerNorm: média 0 e variância 1', ok)}."
    )
    report.write()

    torch.manual_seed(livro.BOOK_SEED)
    x = torch.randn(4, 768)
    with torch.no_grad():
        diferenca = (LayerNorm(768)(x) - nn.LayerNorm(768)(x)).abs().max().item()
    report.write(
        f"**Contra `nn.LayerNorm` (baseline do PyTorch), em (4, 768):** diferença "
        f"máxima de {diferenca:.1e}. A variância enviesada (`unbiased=False`) é o "
        "que torna as duas iguais; com n - 1 no lugar de n, cada saída sairia "
        f"multiplicada por √(767/768) = {math.sqrt(767 / 768):.5f}."
    )
    report.write()
    report.write(
        "Detalhe que o livro deixa implícito: a normalização feita à mão usa a "
        "variância padrão do PyTorch (n - 1), e a classe usa a enviesada (n). Os "
        "números da tabela só batem porque cada conta foi refeita com a mesma "
        "variância que o livro usou naquele trecho."
    )
    report.write()


def example_3_gelu(report: Report, checks: Checks) -> None:
    """L3: a curva da GELU e o shape dentro da FeedForward."""
    report.write("## L3 - Seção 4.3: GELU contra ReLU e a FeedForward (p. 105-108)")
    report.write()
    report.write(
        "**Exemplo do livro:** as duas funções em `torch.linspace(-3, 3, 100)` "
        "(figura 4.8) e uma `FeedForward` do GPT-2 small sobre `torch.rand(2, 3, 768)`."
    )
    report.write()

    gelu, relu = GELU(), nn.ReLU()
    x = torch.linspace(-3, 3, 100)
    with torch.no_grad():
        y_gelu, y_relu = gelu(x), relu(x)

    fino = torch.linspace(-3, 3, 100_001)
    with torch.no_grad():
        curva = gelu(fino)
        exata = GELU(approximate="none")(fino)
    posicao_minimo = int(curva.argmin())
    x_minimo, y_minimo = fino[posicao_minimo].item(), curva[posicao_minimo].item()

    figura, eixo = new_axes(6.4, 3.4)
    eixo.plot(x, y_gelu, color=SERIES_COLORS[0], linewidth=2, label="GELU (aproximação tanh)")
    eixo.plot(x, y_relu, color=SERIES_COLORS[1], linewidth=2, label="ReLU")
    eixo.scatter([x_minimo], [y_minimo], color=SERIES_COLORS[0], s=36, zorder=3, edgecolor=SURFACE, linewidth=2)
    eixo.annotate(
        f"mínimo da GELU\nx = {x_minimo:.3f}",
        xy=(x_minimo, y_minimo),
        xytext=(-2.9, 0.9),
        fontsize=9,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "linewidth": 0.8},
    )
    eixo.set_xlabel("x")
    eixo.set_ylabel("f(x)")
    eixo.set_title("GELU e ReLU em [-3, 3] (figura 4.8 do livro)", fontsize=10)
    legend(eixo, loc="upper left")
    caminho = save_figure(figura, "livro-gelu-relu.png")
    report.figure(caminho, "GELU e ReLU")

    pontos = [-3.0, -2.0, -1.0, x_minimo, -0.5, 0.0, 0.5, 1.0, 3.0]
    with torch.no_grad():
        valores = gelu(torch.tensor(pontos))
    report.table(
        ["x"] + [f"{p:.2f}" for p in pontos],
        [
            ["GELU(x)"] + [f"{v:.4f}" for v in valores],
            ["ReLU(x)"] + [f"{max(p, 0.0):.4f}" for p in pontos],
        ],
    )

    ok = abs(x_minimo - livro.GELU_MINIMUM_X) < 0.01
    report.write(
        f"**Mínimo da GELU:** x = {x_minimo:.4f}, GELU(x) = {y_minimo:.4f}. O "
        f"livro diz \"aproximadamente x = -0,75\". Confere: "
        f"{checks.add('4.3', 'mínimo da GELU perto de -0,75', ok)}. À esquerda "
        "dele a curva desce, à direita sobe: é o único ponto negativo com "
        "derivada zero, e é o que o livro chama de exceção ao gradiente não nulo."
    )
    report.write()
    report.write(
        f"**Aproximação tanh contra a GELU exata (erf)** em [-3, 3]: diferença "
        f"máxima de {(curva - exata).abs().max().item():.1e}. Desprezível para o "
        "modelo; a tanh fica por compatibilidade com o GPT-2."
    )
    report.write()

    torch.manual_seed(livro.BOOK_SEED)
    ffn = FeedForward(GPT_CONFIG_124M.emb_dim)
    x = torch.rand(2, 3, 768)
    formas = [("entrada", x)]
    with torch.no_grad():
        h = x
        for nome, camada in zip(("Linear 1 (expande)", "GELU", "Linear 2 (contrai)"), ffn.layers):
            h = camada(h)
            formas.append((nome, h))

    report.table(
        ["etapa da FeedForward", "shape obtido", "shape na figura 4.9"],
        [
            [nome, str(tuple(t.shape)), esperado]
            for (nome, t), esperado in zip(
                formas, ["(2, 3, 768)", "(2, 3, 3072)", "(2, 3, 3072)", "(2, 3, 768)"]
            )
        ],
    )
    ok = formas[-1][1].shape == (2, 3, 768) and formas[1][1].shape == (2, 3, 3072)
    checks.add("4.3", "shapes dentro da FeedForward", ok)


def example_4_shortcut(report: Report, checks: Checks) -> None:
    """L4: o gradiente que chega a cada camada, com e sem atalho."""
    report.write("## L4 - Seção 4.4: gradiente com e sem atalho (p. 110-113)")
    report.write()
    report.write(
        "**Exemplo do livro:** rede de cinco camadas Linear + GELU "
        "(`layer_sizes = [3, 3, 3, 3, 3, 1]`), entrada `[[1., 0., -1.]]`, erro "
        "quadrático contra 0, seed 123 antes de cada rede. Imprime o gradiente "
        "médio absoluto dos pesos de cada camada."
    )
    report.write()
    report.write(
        "**Como foi refeito:** `ShortcutDemoNetwork` e `mean_weight_gradients`, "
        "de `src/transformer/shortcut_demo.py`, com a GELU do projeto."
    )
    report.write()

    entrada = torch.tensor(livro.SHORTCUT_INPUT)
    obtidos = {}
    for usa_atalho in (False, True):
        torch.manual_seed(livro.BOOK_SEED)
        rede = ShortcutDemoNetwork(livro.SHORTCUT_LAYER_SIZES, use_shortcut=usa_atalho)
        obtidos[usa_atalho] = mean_weight_gradients(rede, entrada)

    linhas = []
    for camada in range(len(livro.GRADIENTS_WITH_SHORTCUT)):
        sem_livro, sem = livro.GRADIENTS_WITHOUT_SHORTCUT[camada], obtidos[False][camada]
        com_livro, com = livro.GRADIENTS_WITH_SHORTCUT[camada], obtidos[True][camada]
        ok = abs(sem - sem_livro) <= 1e-5 * sem_livro and abs(com - com_livro) <= 1e-5 * com_livro
        linhas.append(
            [
                f"layers.{camada}",
                f"{sem_livro:.6g}",
                f"{sem:.6g}",
                f"{com_livro:.6g}",
                f"{com:.6g}",
                checks.add("4.4", f"gradiente da camada {camada}", ok),
            ]
        )

    report.table(
        ["camada", "sem atalho: livro", "obtido", "com atalho: livro", "obtido", "confere"],
        linhas,
    )

    sem, com = obtidos[False], obtidos[True]
    report.write(
        f"Sem atalho, o gradiente da primeira camada é {sem[-1] / sem[0]:.0f}x "
        f"menor que o da última; com atalho, {com[-1] / com[0]:.1f}x. A primeira "
        f"camada recebe {com[0] / sem[0]:.0f}x mais gradiente com o atalho -- "
        "é o vanishing gradient que a seção 4.4 descreve, e o motivo de o "
        "TransformerBlock somar a entrada à saída das duas subcamadas."
    )
    report.write()


def example_5_block(report: Report, checks: Checks) -> None:
    """L5: o bloco do GPT-2 small e o exercício 4.1."""
    report.write("## L5 - Seção 4.5 e exercício 4.1: o TransformerBlock (p. 115-116, 302)")
    report.write()

    cfg = GPT_CONFIG_124M
    torch.manual_seed(livro.BOOK_SEED)
    x = torch.rand(*livro.BLOCK_INPUT_SHAPE)
    bloco = TransformerBlock(
        emb_dim=cfg.emb_dim,
        context_length=cfg.context_length,
        n_heads=cfg.n_heads,
        drop_rate=cfg.drop_rate,
    )
    with torch.no_grad():
        saida = bloco(x)

    atencao = sum(p.numel() for p in bloco.att.parameters())
    feed_forward = sum(p.numel() for p in bloco.ff.parameters())
    normas = sum(p.numel() for n in (bloco.norm1, bloco.norm2) for p in n.parameters())

    report.table(
        ["grandeza", "livro", "obtido", "confere"],
        [
            [
                "shape de entrada / saída",
                f"{livro.BLOCK_INPUT_SHAPE} / {livro.BLOCK_INPUT_SHAPE}",
                f"{tuple(x.shape)} / {tuple(saida.shape)}",
                checks.add("4.5", "shape do bloco", tuple(saida.shape) == livro.BLOCK_INPUT_SHAPE),
            ],
            [
                "parâmetros da FeedForward",
                thousands(livro.FEED_FORWARD_PARAMS),
                thousands(feed_forward),
                checks.add("4.5", "exercício 4.1: FeedForward", feed_forward == livro.FEED_FORWARD_PARAMS),
            ],
            [
                "parâmetros da atenção",
                thousands(livro.ATTENTION_PARAMS),
                thousands(atencao),
                checks.add("4.5", "exercício 4.1: atenção", atencao == livro.ATTENTION_PARAMS),
            ],
        ],
    )

    report.write(
        f"Exercício 4.1: a FeedForward tem {feed_forward / atencao:.2f}x os "
        f"parâmetros da atenção. As contas: atenção = 4 · 768² + 768 (Q, K, V e "
        f"`out_proj`, só esta com bias) = {thousands(atencao)}; FeedForward = "
        f"2 · 768 · 3072 + 3072 + 768 = {thousands(feed_forward)}. Com as duas "
        f"LayerNorms ({thousands(normas)}), o bloco tem "
        f"{thousands(atencao + feed_forward + normas)} parâmetros."
    )
    report.write()


def example_6_gpt(report: Report, checks: Checks, lote: torch.Tensor) -> GPTModel:
    """L6: o GPTModel de 124M e os exercícios 4.2 e 4.3."""
    report.write("## L6 - Seção 4.6: o GPTModel de 124M (p. 119-122)")
    report.write()
    report.write(
        "**Exemplo do livro:** `torch.manual_seed(123)`, `GPTModel(GPT_CONFIG_124M)` "
        "e uma passada sobre o lote do L1, em modo de treino."
    )
    report.write()

    torch.manual_seed(livro.BOOK_SEED)
    modelo = GPTModel(GPT_CONFIG_124M)
    with torch.no_grad():
        logits = modelo(lote)

    report.write(f"Shape dos logits: `{tuple(logits.shape)}`.")
    report.write()
    logits_table(report, checks, "4.6", logits, livro.GPT_LOGITS)

    total = modelo.parameter_count()
    cabeca = sum(p.numel() for p in modelo.out_head.parameters())
    megas = f"{modelo.size_in_megabytes():.2f}"
    forma_tokens = tuple(modelo.embedding.token_embedding.weight.shape)
    forma_cabeca = tuple(modelo.out_head.weight.shape)

    report.table(
        ["grandeza", "livro", "obtido", "confere"],
        [
            [
                "parâmetros",
                thousands(livro.GPT_124M_PARAMS),
                thousands(total),
                checks.add("4.6", "total de parâmetros", total == livro.GPT_124M_PARAMS),
            ],
            [
                "shape da tabela de tokens",
                str(livro.EMBEDDING_SHAPE),
                str(forma_tokens),
                checks.add("4.6", "shape da tabela de tokens", forma_tokens == livro.EMBEDDING_SHAPE),
            ],
            [
                "shape da cabeça de saída",
                str(livro.EMBEDDING_SHAPE),
                str(forma_cabeca),
                checks.add("4.6", "shape da cabeça de saída", forma_cabeca == livro.EMBEDDING_SHAPE),
            ],
            [
                "parâmetros com weight tying",
                thousands(livro.GPT_124M_PARAMS_TIED),
                thousands(total - cabeca),
                checks.add("4.6", "parâmetros com weight tying", total - cabeca == livro.GPT_124M_PARAMS_TIED),
            ],
            [
                "tamanho em float32",
                f"{livro.GPT_124M_MEGABYTES} MB",
                f"{megas} MB",
                checks.add("4.6", "tamanho em MB", megas == livro.GPT_124M_MEGABYTES),
            ],
        ],
    )

    partes = modelo.parameter_breakdown()
    report.write(
        "De onde vêm os 163 milhões (conta que o livro não faz, e que explica a "
        "diferença para 124M):"
    )
    report.write()
    report.table(
        ["componente", "parâmetros", "fração"],
        [[nome, thousands(valor), f"{100 * valor / total:.1f}%"] for nome, valor in partes.items()],
    )

    report.write("### Exercício 4.2 - os outros tamanhos do GPT-2")
    report.write()
    report.write(
        "Só a configuração muda (`emb_dim`, `n_layers`, `n_heads`). Os modelos "
        "são montados no device `meta`, que cria os shapes sem alocar memória -- "
        "contar o XL de verdade pediria 6 GB. O livro só imprime o XL."
    )
    report.write()

    linhas = []
    for nome, cfg in GPT2_CONFIGS.items():
        with torch.device("meta"):
            gpt = GPTModel(cfg)
        total_cfg = gpt.parameter_count()
        sem_cabeca = total_cfg - gpt.parameter_breakdown()["out_head"]
        megas_cfg = f"{gpt.size_in_megabytes():.2f}"

        if nome == "gpt2-xl":
            ok = (
                total_cfg == livro.GPT2_XL_PARAMS
                and sem_cabeca == livro.GPT2_XL_PARAMS_TIED
                and megas_cfg == livro.GPT2_XL_MEGABYTES
            )
            confere = checks.add("4.6", "exercício 4.2: GPT-2 XL", ok)
            livro_txt = (
                f"{thousands(livro.GPT2_XL_PARAMS)} / {thousands(livro.GPT2_XL_PARAMS_TIED)} "
                f"/ {livro.GPT2_XL_MEGABYTES} MB"
            )
        else:
            confere, livro_txt = "-", "não impresso"

        linhas.append(
            [
                nome,
                f"{cfg.emb_dim} / {cfg.n_layers} / {cfg.n_heads}",
                thousands(total_cfg),
                thousands(sem_cabeca),
                f"{megas_cfg} MB",
                livro_txt,
                confere,
            ]
        )

    report.table(
        [
            "modelo",
            "emb_dim / n_layers / n_heads",
            "parâmetros",
            "com weight tying",
            "float32",
            "livro",
            "confere",
        ],
        linhas,
    )

    report.write("### Exercício 4.3 - taxas de dropout separadas")
    report.write()
    report.write(
        "A `GPTConfig` do projeto tem `drop_rate_emb`, `drop_rate_attn` e "
        "`drop_rate_shortcut`, que herdam `drop_rate` quando ficam em `None`. "
        "Taxas propositalmente diferentes, para mostrar que cada uma chega ao seu módulo:"
    )
    report.write()
    cfg = replace(
        GPT_CONFIG_124M, n_layers=2, drop_rate_emb=0.1, drop_rate_attn=0.05, drop_rate_shortcut=0.2
    )
    with torch.device("meta"):
        gpt = GPTModel(cfg)
    lugares = [
        ("embeddings (`drop_emb`)", 0.1, gpt.drop_emb.p),
        ("pesos de atenção (`att.dropout`)", 0.05, gpt.trf_blocks[0].att.dropout.p),
        ("antes do atalho (`drop_shortcut`)", 0.2, gpt.trf_blocks[0].drop_shortcut.p),
    ]
    report.table(
        ["onde", "taxa pedida", "taxa no módulo", "confere"],
        [
            [onde, f"{pedida}", f"{obtida}", checks.add("4.6", f"exercício 4.3: {onde}", pedida == obtida)]
            for onde, pedida, obtida in lugares
        ],
    )

    return modelo


def example_7_generation(report: Report, checks: Checks, modelo: GPTModel, bpe: BPETokenizer) -> None:
    """L7: a geração gulosa da seção 4.7, uma iteração por linha."""
    report.write('## L7 - Seção 4.7: geração gulosa de "Hello, I am" (p. 125-127)')
    report.write()
    report.write(
        "**Exemplo do livro:** o modelo do L6 em `eval()`, 6 tokens novos com "
        "`generate_text_simple`."
    )
    report.write()
    report.write(
        "**Como foi refeito:** `generate` de `src/generation/` com a temperatura "
        "padrão (0, gulosa). Para mostrar o laço, cada linha abaixo é uma chamada "
        "de `generate(..., max_new_tokens=1)` sobre o contexto da linha anterior; "
        "a última linha confere que uma chamada única com 6 tokens dá o mesmo."
    )
    report.write()

    ids = text_to_token_ids(livro.GENERATION_PROMPT, bpe)
    checks.add("4.7", "Token IDs do prompt", ids.tolist() == [livro.GENERATION_PROMPT_IDS])

    linhas = []
    contexto = ids
    for iteracao in range(1, livro.GENERATION_NEW_TOKENS + 1):
        seguinte = generate(modelo, contexto, max_new_tokens=1)
        novo = seguinte[0, -1].item()
        linhas.append(
            [
                str(iteracao),
                str(contexto[0].tolist()),
                str(novo),
                f"`{bpe.decode([novo])!r}`",
            ]
        )
        contexto = seguinte

    report.table(["iteração", "contexto (Token IDs)", "próximo token", "texto do token"], linhas)

    de_uma_vez = generate(modelo, ids, max_new_tokens=livro.GENERATION_NEW_TOKENS)
    texto = token_ids_to_text(de_uma_vez, bpe)

    report.table(
        ["grandeza", "livro", "obtido", "confere"],
        [
            [
                "Token IDs gerados",
                str(livro.GENERATION_OUTPUT_IDS),
                str(de_uma_vez[0].tolist()),
                checks.add("4.7", "Token IDs gerados", de_uma_vez.tolist() == [livro.GENERATION_OUTPUT_IDS]),
            ],
            [
                "texto",
                f"`{livro.GENERATION_OUTPUT_TEXT}`",
                f"`{texto}`",
                checks.add("4.7", "texto gerado", texto == livro.GENERATION_OUTPUT_TEXT),
            ],
            [
                "laço passo a passo = chamada única",
                "-",
                str(torch.equal(contexto, de_uma_vez)),
                checks.add("4.7", "laço passo a passo = chamada única", torch.equal(contexto, de_uma_vez)),
            ],
        ],
    )

    report.write(
        "O texto é incoerente pelo motivo que o livro dá: os pesos ainda são os "
        "sorteados pela seed 123. Reproduzir o lixo exato importa mesmo assim -- "
        "mostra que embeddings, 12 blocos, norma final, cabeça e laço de geração "
        "fazem a mesma conta que a do livro, peso por peso."
    )
    report.write()


def example_8_decoding(report: Report, checks: Checks) -> None:
    """L8: as estratégias de decodificação no vocabulário de 9 palavras."""
    report.write("## L8 - Seção 5.3: decodificação no vocabulário de 9 palavras (p. 152-158)")
    report.write()
    report.write(
        "**Exemplo do livro:** logits fixos para o próximo token depois de "
        '"every effort moves you". Não depende de modelo: isola o que '
        "temperatura, amostragem e top-k fazem com uma distribuição."
    )
    report.write()

    vocab = livro.TOY_VOCAB
    logits = torch.tensor(livro.TOY_LOGITS)

    gulosa = vocab[select_next_token(logits.unsqueeze(0)).item()]
    report.write(
        f"**Gulosa (temperatura 0):** `{gulosa}`; livro: `{livro.TOY_GREEDY_WORD}`. "
        f"Confere: {checks.add('5.3', 'escolha gulosa', gulosa == livro.TOY_GREEDY_WORD)}."
    )
    report.write()

    def amostra(temperatura: float, top_k: int | None = None) -> list[int]:
        # Um sorteio por chamada, gerador semeado com a seed do livro: a mesma
        # sequência de números que o laço de print_sampled_tokens consome.
        gerador = torch.Generator().manual_seed(livro.BOOK_SEED)
        return [
            select_next_token(logits.unsqueeze(0), temperatura, top_k, gerador).item()
            for _ in range(livro.TOY_SAMPLES)
        ]

    contagem = torch.bincount(torch.tensor(amostra(1.0)), minlength=len(vocab)).tolist()

    report.write(f"**Amostragem com temperatura 1, {livro.TOY_SAMPLES} sorteios, seed {livro.BOOK_SEED}:**")
    report.write()
    linhas = []
    for indice, palavra in enumerate(vocab):
        esperado = livro.TOY_SAMPLE_COUNTS.get(palavra)
        if esperado is None:
            confere = "-"
            livro_txt = "não listado"
        else:
            confere = checks.add("5.3", f"sorteios de '{palavra}'", contagem[indice] == esperado)
            livro_txt = str(esperado)
        linhas.append([f"`{palavra}`", livro_txt, str(contagem[indice]), confere])
    report.table(["palavra", "livro", "obtido", "confere"], linhas)

    probabilidades = {t: next_token_probabilities(logits, temperature=t) for t in livro.TOY_TEMPERATURES}

    figura, eixo = new_axes(7.0, 3.4)
    largura = 0.27
    posicoes = torch.arange(len(vocab))
    for serie, temperatura in enumerate(livro.TOY_TEMPERATURES):
        eixo.bar(
            posicoes + (serie - 1) * largura,
            probabilidades[temperatura],
            width=largura,
            color=SERIES_COLORS[serie],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=f"temperatura {temperatura:g}",
        )
    eixo.set_xticks(posicoes.tolist(), vocab, rotation=45, ha="right")
    eixo.set_ylabel("probabilidade")
    eixo.set_ylim(0, 1.05)
    eixo.set_title("Efeito da temperatura (figura 5.14 do livro)", fontsize=10)
    legend(eixo, loc="upper left")
    caminho = save_figure(figura, "livro-temperatura.png")
    report.figure(caminho, "Probabilidades por temperatura")

    report.table(
        ["palavra"] + [f"T = {t:g}" for t in livro.TOY_TEMPERATURES],
        [
            [f"`{palavra}`"] + [f"{probabilidades[t][indice].item():.4f}" for t in livro.TOY_TEMPERATURES]
            for indice, palavra in enumerate(vocab)
        ],
    )

    pizza = vocab.index("pizza")
    sorteios_pizza = {t: amostra(t).count(pizza) for t in livro.TOY_TEMPERATURES}
    prob_pizza = probabilidades[5.0][pizza].item()
    report.write("**Exercício 5.1 - quantas vezes `pizza` é sorteado:**")
    report.write()
    report.table(
        ["temperatura", "livro", "obtido", "probabilidade exata", "confere"],
        [
            [
                "1",
                "0",
                str(sorteios_pizza[1.0]),
                f"{probabilidades[1.0][pizza].item():.4f}",
                checks.add("5.3", "exercício 5.1: pizza com T = 1", sorteios_pizza[1.0] == 0),
            ],
            [
                "0.1",
                "0",
                str(sorteios_pizza[0.1]),
                f"{probabilidades[0.1][pizza].item():.1e}",
                checks.add("5.3", "exercício 5.1: pizza com T = 0,1", sorteios_pizza[0.1] == 0),
            ],
            [
                "5",
                f"{livro.TOY_PIZZA_COUNT_T5} (prob. {livro.TOY_PIZZA_PROBABILITY_T5:.1%})",
                str(sorteios_pizza[5.0]),
                f"{prob_pizza:.4f}",
                checks.add(
                    "5.3",
                    "exercício 5.1: pizza com T = 5",
                    sorteios_pizza[5.0] == livro.TOY_PIZZA_COUNT_T5
                    and abs(prob_pizza - livro.TOY_PIZZA_PROBABILITY_T5) < 5e-4,
                ),
            ],
        ],
    )
    report.write(
        "O apêndice C escreve \"0x if the temperature is 0 or 0.1\"; a figura "
        "5.14 usa as temperaturas 1, 0,1 e 5, então o \"0\" do texto só pode ser "
        "a temperatura 1. A resposta à segunda pergunta do exercício (\"um jeito "
        "mais rápido e preciso\") é a coluna da probabilidade exata: ler o "
        "softmax em vez de contar sorteios. 32/1000 = 3,2% é uma estimativa "
        f"ruidosa de {prob_pizza:.1%}."
    )
    report.write()

    valores, posicoes_top = torch.topk(logits, livro.TOY_TOP_K)
    filtrados = apply_top_k(logits, livro.TOY_TOP_K)
    top_k_probs = next_token_probabilities(logits, top_k=livro.TOY_TOP_K)
    report.write(f"**Top-k com k = {livro.TOY_TOP_K}:**")
    report.write()
    report.table(
        ["grandeza", "livro", "obtido", "confere"],
        [
            [
                "maiores logits",
                vector(livro.TOY_TOP3_LOGITS),
                vector(valores),
                checks.add("5.3", "top-3: logits", close(valores, livro.TOY_TOP3_LOGITS)),
            ],
            [
                "posições",
                str(livro.TOY_TOP3_POSITIONS),
                str(posicoes_top.tolist()),
                checks.add("5.3", "top-3: posições", posicoes_top.tolist() == livro.TOY_TOP3_POSITIONS),
            ],
            [
                "logits filtrados",
                "-inf fora das 3 posições",
                "[" + ", ".join("-inf" if v == float("-inf") else f"{v:.2f}" for v in filtrados.tolist()) + "]",
                checks.add("5.3", "top-3: máscara -inf", int(torch.isfinite(filtrados).sum()) == 3),
            ],
            [
                "probabilidades",
                vector(livro.TOY_TOP3_PROBABILITIES),
                vector(top_k_probs),
                checks.add("5.3", "top-3: probabilidades", close(top_k_probs, livro.TOY_TOP3_PROBABILITIES)),
            ],
        ],
    )

    deterministicas = {
        "temperatura 0 (top_k qualquer)": amostra(0.0, top_k=5),
        "temperatura 1 com top_k = 1": amostra(1.0, top_k=1),
    }
    report.write("**Exercício 5.3 - configurações determinísticas:**")
    report.write()
    report.table(
        ["configuração", "palavras distintas em 1000 sorteios", "palavra", "confere"],
        [
            [
                nome,
                str(len(set(sorteios))),
                f"`{vocab[sorteios[0]]}`",
                checks.add("5.3", f"exercício 5.3: {nome}", set(sorteios) == {vocab.index(livro.TOY_GREEDY_WORD)}),
            ]
            for nome, sorteios in deterministicas.items()
        ],
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    bpe = BPETokenizer()
    checks = Checks()

    report = Report(RESULTS_PATH)
    report.write("# Sprint 4 - Exemplos do livro reproduzidos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint04/livro_exemplos.py`. Não editar à mão: "
        "rode o script novamente."
    )
    report.write()
    report.write(
        "Cada exemplo do capítulo 4 (e da seção 5.3) foi refeito com o código de "
        "`src/`, com a seed do livro, e comparado com o valor impresso. Valores "
        "de 4 casas conferem com tolerância de meia unidade na última casa; "
        "contagens e Token IDs, com igualdade exata. Fonte dos valores do livro: "
        "`tests/book_reference.py`."
    )
    report.write()
    resumo = report.mark()

    lote = example_1_skeleton(report, checks, bpe)
    example_2_layer_norm(report, checks)
    example_3_gelu(report, checks)
    example_4_shortcut(report, checks)
    example_5_block(report, checks)
    modelo = example_6_gpt(report, checks, lote)
    example_7_generation(report, checks, modelo, bpe)
    example_8_decoding(report, checks)

    # Resumo por seção, inserido no topo depois de tudo medido.
    secoes: dict[str, list[bool]] = {}
    for secao, _, ok in checks.rows:
        secoes.setdefault(secao, []).append(ok)
    total_ok = sum(ok for _, _, ok in checks.rows)
    linhas = [
        "## Resumo",
        "",
        f"**{total_ok} de {len(checks.rows)} comparações conferem.**",
        "",
        "| seção do livro | comparações | conferem |",
        "| -------------- | ----------- | -------- |",
    ]
    linhas += [f"| {secao} | {len(oks)} | {sum(oks)} |" for secao, oks in secoes.items()]
    falhas = [f"- {secao}: {exemplo}" for secao, exemplo, ok in checks.rows if not ok]
    if falhas:
        linhas += ["", "Divergências:", ""] + falhas
    linhas.append("")
    report.insert(resumo, linhas)
    print("\n".join(linhas))

    report.save()
    print(f"\nResultados salvos em {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0 if total_ok == len(checks.rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
