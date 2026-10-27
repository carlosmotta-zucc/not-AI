"""Experimentos de arquitetura da Sprint 4: o que cada peça custa e o que ela muda.

Os testes provam que o GPTModel está correto e o livro_exemplos.py prova que ele
é o do livro. Estes experimentos medem o comportamento e o custo quando a
arquitetura muda. A interpretação vai para docs/sprint04/; aqui ficam os números.

A1  Anatomia do modelo base: shapes medidos em cada etapa e parâmetros por componente
A2  O modelo sem treino: logits, probabilidades e perda inicial
A3  Número de blocos (n_layers)
A4  Número de cabeças (n_heads), com emb_dim fixo
A5  Dimensão do embedding (emb_dim)
A6  Dimensão da FeedForward (ff_hidden_dim)
A7  Tamanho do contexto (context_length)
A8  Dropout: a taxa global e as três taxas separadas do exercício 4.3
A9  Ablação do bloco: sem atalho, sem LayerNorm e Post-LayerNorm

Configuração de partida: BASE_CONFIG (src/model/config.py). Cada experimento
muda um hiperparâmetro e mantém os outros. Perdas e estatísticas sobre o
primeiro lote de The Verdict (8 x 128, sem embaralhar), em eval() e sob
no_grad(), exceto onde o experimento é justamente sobre train() (A8) ou sobre
o gradiente (A9). Tempo: forward de uma sequência (1 x T), mediana de 11
medidas após 3 de aquecimento. Seed fixa antes de cada modelo.

Uso:
    python experimentos/sprint04/arquitetura_experimentos.py
"""

import copy
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.attention.multi_head as multi_head_module  # noqa: E402
from relatorio import (  # noqa: E402
    BYTES_PER_FLOAT32,
    INK_SECONDARY,
    REPEATS,
    RESULTS_DIR,
    SEED,
    SERIES_COLORS,
    SURFACE,
    WARMUP,
    Report,
    legend,
    median_ms,
    megabytes,
    new_axes,
    percent,
    save_figure,
    thousands,
)
from src.corpus import load_verdict  # noqa: E402
from src.dataset import create_dataloader  # noqa: E402
from src.model import BASE_CONFIG, GPT2_CONFIGS, GPT_CONFIG_124M, GPTConfig, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.transformer import TransformerBlock  # noqa: E402

RESULTS_PATH = RESULTS_DIR / "arquitetura.md"

BATCH_SIZE = 8

# Varreduras. O valor da configuração base está em todas, para cada tabela ter
# uma linha de referência.
LAYERS_SWEEP = (1, 2, 4, 8, 12)
HEADS_SWEEP = (1, 2, 4, 8, 16)
EMB_DIM_SWEEP = (64, 128, 256, 512, 768)
FF_FACTOR_SWEEP = (1, 2, 4, 8)
CONTEXT_SWEEP = (32, 64, 128, 256, 512, 1024)
DROPOUT_SWEEP = (0.0, 0.1, 0.2, 0.5)

# A8: passadas em train() para a média da perda com dropout.
DROPOUT_PASSES = 5

# A9: profundo o bastante para o efeito do atalho e da norma aparecer, com a
# largura da configuração base.
ABLATION_LAYERS = 12


# --- medidas -----------------------------------------------------------------


def build(cfg: GPTConfig, seed: int = SEED) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel(cfg).eval()


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Perda média do próximo token: cada posição é uma classificação em V classes."""
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten()).item()


def forward_ms(model: GPTModel, tokens: int) -> float:
    ids = torch.zeros(1, tokens, dtype=torch.long)
    return median_ms(lambda: model(ids))


def estimated_gflops(cfg: GPTConfig, tokens: int) -> dict[str, float]:
    """FLOPs de um forward de 1 sequência, separados por origem (em GFLOP).

    Conta só as multiplicações de matriz (2 FLOPs por multiplica-e-soma):

    - linear: as projeções de cada bloco, Q, K, V e saída (4·D²) e as duas da
      FeedForward (2·D·H), por token -- cresce linear com T.
    - atenção: Q·Kᵀ e pesos·V, T² · D cada, por bloco -- cresce com T².
      A matriz é calculada inteira e só depois mascarada, então conta inteira.
    - cabeça: a projeção D -> V de cada token -- linear com T, mas com V = 50 257.

    LayerNorm, GELU, softmax e somas são elemento a elemento e ficam de fora.
    Os embeddings também: são leitura de tabela, não conta.
    """
    d, h, v, n = cfg.emb_dim, cfg.feed_forward_dim, cfg.vocab_size, cfg.n_layers
    linear = n * 2 * tokens * (4 * d * d + 2 * d * h)
    atencao = n * 2 * 2 * tokens * tokens * d
    cabeca = 2 * tokens * d * v
    return {
        "linear": linear / 1e9,
        "atencao": atencao / 1e9,
        "cabeca": cabeca / 1e9,
        "total": (linear + atencao + cabeca) / 1e9,
    }


def residual_stream(model: GPTModel, ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """O fluxo residual depois dos embeddings e de cada bloco, e os logits.

    Refaz o forward do GPTModel passo a passo -- mesma sequência de módulos --
    para poder olhar entre os blocos.
    """
    x = model.drop_emb(model.embedding(ids))
    estados = [x]
    for bloco in model.trf_blocks:
        x = bloco(x)
        estados.append(x)
    return estados, model.out_head(model.final_norm(x))


def blocks_share(model: GPTModel) -> float:
    partes = model.parameter_breakdown()
    blocos = partes["attention"] + partes["feed_forward"] + partes["block_norms"]
    return blocos / model.parameter_count()


def on_meta(cfg: GPTConfig) -> GPTModel:
    with torch.device("meta"):
        return GPTModel(cfg)


# --- experimentos ------------------------------------------------------------


def experiment_1_anatomy(report: Report, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """A1: por onde passa o tensor, e onde estão os parâmetros."""
    report.write("## A1 - Anatomia do modelo base")
    report.write()
    report.write(
        "**Pergunta:** que shape o tensor tem em cada etapa do forward, e em "
        "quais componentes estão os parâmetros?"
    )
    report.write()
    cfg = BASE_CONFIG
    report.write(
        f"**Configuração:** BASE_CONFIG -- vocab_size {cfg.vocab_size}, "
        f"context_length {cfg.context_length}, emb_dim {cfg.emb_dim}, n_heads "
        f"{cfg.n_heads} (head_dim {cfg.head_dim}), n_layers {cfg.n_layers}, "
        f"FeedForward {cfg.feed_forward_dim}, drop_rate {cfg.drop_rate}. Entrada: "
        f"o lote do The Verdict, {tuple(batch[0].shape)}."
    )
    report.write()

    model = build(cfg)
    rotulos = {
        "embedding.token_embedding": "1. tabela de tokens (Sprint 2)",
        "embedding.positional_embedding": "2. tabela de posições (Sprint 2)",
        "embedding": "3. soma tokens + posições",
        "drop_emb": "4. dropout dos embeddings",
        "trf_blocks.0.norm1": "5. bloco 0: LayerNorm 1",
        "trf_blocks.0.att.W_query": "6. bloco 0: projeção Q (K e V iguais)",
        "trf_blocks.0.att": "8. bloco 0: atenção multi-cabeça (contexto, pesos)",
        "trf_blocks.0.norm2": "9. bloco 0: LayerNorm 2",
        "trf_blocks.0.ff.layers.0": "10. bloco 0: FeedForward, Linear 1",
        "trf_blocks.0.ff.layers.1": "11. bloco 0: FeedForward, GELU",
        "trf_blocks.0.ff.layers.2": "12. bloco 0: FeedForward, Linear 2",
        "trf_blocks.0": "13. saída do bloco 0 (dois atalhos somados)",
        "trf_blocks": f"14. saída da pilha de {cfg.n_layers} blocos",
        "final_norm": "15. LayerNorm final",
        "out_head": "16. cabeça de saída: logits",
    }

    def descreve(valor) -> str:
        if isinstance(valor, torch.Tensor):
            return str(tuple(valor.shape))
        if isinstance(valor, (tuple, list)):
            return " e ".join(descreve(item) for item in valor)
        if isinstance(valor, int):
            return f"T = {valor}"
        return type(valor).__name__

    linhas: list[list[str]] = []
    ganchos = []
    for nome, modulo in model.named_modules():
        if nome in rotulos:
            def gancho(_, entradas, saida, nome=nome):
                linhas.append([rotulos[nome], descreve(entradas[0]), descreve(saida)])

            ganchos.append(modulo.register_forward_hook(gancho))

    # O split em cabeças acontece dentro do forward da atenção, numa função
    # interna que não é módulo. Para medi-lo, a função núcleo é embrulhada só
    # durante esta passada e restaurada em seguida.
    original = multi_head_module.scaled_dot_product_attention
    vistos = []

    def espia(queries, keys, values, **kwargs):
        if not vistos:
            linhas.append(
                [
                    "7. bloco 0: Q, K e V divididos em cabeças",
                    descreve(queries),
                    f"{descreve(queries)} por matriz",
                ]
            )
        vistos.append(True)
        return original(queries, keys, values, **kwargs)

    multi_head_module.scaled_dot_product_attention = espia
    try:
        with torch.no_grad():
            model(batch[0])
    finally:
        multi_head_module.scaled_dot_product_attention = original
        for gancho in ganchos:
            gancho.remove()

    # Os ganchos disparam quando o módulo TERMINA: a soma (3) termina depois
    # das duas tabelas, o bloco (13) depois das suas subpeças. A ordenação
    # pelo número do rótulo devolve a ordem em que os dados passam.
    linhas.sort(key=lambda linha: int(linha[0].split(".")[0]))
    report.write("### Shapes medidos (ganchos de forward)")
    report.write()
    report.table(["etapa", "entrada", "saída"], linhas)

    report.write(
        f"Legenda: B = {BATCH_SIZE} sequências, T = {cfg.context_length} tokens, "
        f"D = emb_dim = {cfg.emb_dim}, H = {cfg.n_heads} cabeças de "
        f"{cfg.head_dim}, FeedForward com {cfg.feed_forward_dim} = 4·D, V = "
        f"{cfg.vocab_size}. Os blocos 1 a {cfg.n_layers - 1} repetem as linhas "
        "5 a 13 com os mesmos shapes."
    )
    report.write()

    report.write("### Parâmetros por componente")
    report.write()
    pequeno = on_meta(GPT_CONFIG_124M)
    partes_base = model.parameter_breakdown()
    partes_gpt2 = pequeno.parameter_breakdown()
    total_base, total_gpt2 = model.parameter_count(), pequeno.parameter_count()

    report.table(
        ["componente", "base", "fração", "GPT-2 small", "fração"],
        [
            [
                nome,
                thousands(partes_base[nome]),
                percent(partes_base[nome], total_base),
                thousands(partes_gpt2[nome]),
                percent(partes_gpt2[nome], total_gpt2),
            ]
            for nome in partes_base
        ]
        + [["**total**", thousands(total_base), "100%", thousands(total_gpt2), "100%"]],
    )

    def tabelas(partes: dict[str, int]) -> int:
        return partes["token_embedding"] + partes["positional_embedding"] + partes["out_head"]

    report.write(
        f"Tabelas de embedding e cabeça de saída somam "
        f"{percent(tabelas(partes_base), total_base)} do modelo base e "
        f"{percent(tabelas(partes_gpt2), total_gpt2)} do GPT-2 small. Em float32 "
        f"o modelo base ocupa {model.size_in_megabytes():.2f} MB."
    )
    report.write()

    bloco = model.trf_blocks[0]
    por_parte = {
        "atenção": sum(p.numel() for p in bloco.att.parameters()),
        "FeedForward": sum(p.numel() for p in bloco.ff.parameters()),
        "2 LayerNorms": sum(p.numel() for n in (bloco.norm1, bloco.norm2) for p in n.parameters()),
    }
    total_bloco = sum(por_parte.values())
    report.write(f"Um bloco do modelo base ({thousands(total_bloco)} parâmetros):")
    report.write()
    report.table(
        ["subpeça", "parâmetros", "fração do bloco"],
        [[nome, thousands(valor), percent(valor, total_bloco)] for nome, valor in por_parte.items()],
    )


def experiment_2_untrained(report: Report, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """A2: o que os logits de um modelo sem treino dizem."""
    report.write("## A2 - O modelo sem treino: logits, probabilidades e perda")
    report.write()
    report.write(
        "**Pergunta:** antes de qualquer treino, o que o modelo prevê? Os "
        "logits já são uma distribuição útil, ou ainda não dizem nada?"
    )
    report.write()
    report.write(
        "**Configuração:** BASE_CONFIG (seed 42) e GPT-2 small (seed 123, a do "
        "livro), os dois em eval(), sobre o lote do The Verdict. A perda é a "
        "cross-entropy entre os logits de cada posição e o token seguinte (o "
        "alvo do DataLoader). Referência: uma distribuição uniforme sobre V "
        "tokens tem entropia e perda iguais a ln V."
    )
    report.write()

    entrada, alvo = batch
    colunas = {}
    for nome, cfg, seed in (("base", BASE_CONFIG, SEED), ("GPT-2 small", GPT_CONFIG_124M, 123)):
        model = build(cfg, seed)
        with torch.no_grad():
            logits = model(entrada)
        probabilidades = torch.softmax(logits, dim=-1)
        entropia = -(probabilidades * probabilidades.clamp_min(1e-30).log()).sum(dim=-1)
        prob_do_alvo = probabilidades.gather(-1, alvo.unsqueeze(-1)).squeeze(-1)
        acertos = (logits.argmax(dim=-1) == alvo).float().mean().item()

        colunas[nome] = [
            f"{logits.mean().item():.4f}",
            f"{logits.std().item():.4f}",
            f"{logits.min().item():.2f} a {logits.max().item():.2f}",
            f"{probabilidades.max(dim=-1).values.mean().item():.2e}",
            f"{prob_do_alvo.mean().item():.2e}",
            f"{entropia.mean().item():.3f}",
            f"{cross_entropy(logits, alvo):.3f}",
            f"{100 * acertos:.2f}%",
        ]
        del model, logits, probabilidades

    v = BASE_CONFIG.vocab_size
    referencia = [
        "-",
        "-",
        "-",
        f"{1 / v:.2e}",
        f"{1 / v:.2e}",
        f"{math.log(v):.3f}",
        f"{math.log(v):.3f}",
        f"{100 / v:.4f}%",
    ]
    grandezas = [
        "média dos logits",
        "desvio dos logits",
        "faixa dos logits",
        "maior probabilidade (média por posição)",
        "probabilidade dada ao token certo (média)",
        "entropia da previsão (nats)",
        "cross-entropy (nats)",
        "acerto do argmax",
    ]
    report.table(
        ["grandeza", "base", "GPT-2 small", "uniforme sobre V"],
        [
            [grandeza, colunas["base"][i], colunas["GPT-2 small"][i], referencia[i]]
            for i, grandeza in enumerate(grandezas)
        ],
    )


def sweep_row_basics(model: GPTModel, cfg: GPTConfig, batch, tokens: int) -> dict[str, str]:
    """Colunas comuns às varreduras: parâmetros, memória, FLOPs, tempo e perda."""
    with torch.no_grad():
        logits = model(batch[0])
    return {
        "parametros": thousands(model.parameter_count()),
        "mb": f"{model.size_in_megabytes():.1f} MB",
        "gflop": f"{estimated_gflops(cfg, tokens)['total']:.2f}",
        "ms": f"{forward_ms(model, tokens):.1f}",
        "perda": f"{cross_entropy(logits, batch[1]):.3f}",
    }


def experiment_3_layers(report: Report, batch) -> None:
    """A3: mais blocos -- o que cresce, e o que a LayerNorm final segura."""
    report.write("## A3 - Número de blocos (n_layers)")
    report.write()
    report.write(
        "**Pergunta:** empilhar mais blocos custa quanto, e o que acontece com a "
        "escala do fluxo residual e dos logits?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG com n_layers em "
        f"{', '.join(map(str, LAYERS_SWEEP))}. GFLOP e tempo para 1 sequência de "
        f"{BASE_CONFIG.context_length} tokens. O desvio do fluxo residual é "
        "medido na saída do último bloco, antes da LayerNorm final; o dos "
        "logits, depois da cabeça."
    )
    report.write()

    linhas = []
    for camadas in LAYERS_SWEEP:
        cfg = replace(BASE_CONFIG, n_layers=camadas)
        model = build(cfg)
        basicos = sweep_row_basics(model, cfg, batch, cfg.context_length)
        with torch.no_grad():
            estados, logits = residual_stream(model, batch[0])
        linhas.append(
            [
                str(camadas),
                basicos["parametros"],
                percent(blocks_share(model), 1),
                basicos["mb"],
                basicos["gflop"],
                basicos["ms"],
                f"{estados[-1].std().item():.3f}",
                f"{logits.std().item():.3f}",
                basicos["perda"],
            ]
        )

    report.table(
        [
            "n_layers",
            "parâmetros",
            "nos blocos",
            "float32",
            "GFLOP",
            "tempo (ms)",
            "desvio do fluxo residual",
            "desvio dos logits",
            "perda inicial",
        ],
        linhas,
    )


def experiment_4_heads(report: Report, batch) -> None:
    """A4: mais cabeças com o mesmo emb_dim."""
    report.write("## A4 - Número de cabeças (n_heads), com emb_dim fixo")
    report.write()
    report.write(
        "**Pergunta:** o resultado da Sprint 3 (mais cabeças não mudam os "
        "parâmetros, mas multiplicam a matriz de scores) vale para o modelo "
        "inteiro?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG com n_heads em "
        f"{', '.join(map(str, HEADS_SWEEP))} e emb_dim {BASE_CONFIG.emb_dim}. As "
        "células de scores são as de todos os blocos juntos, L·H·T², para 1 "
        "sequência de T tokens."
    )
    report.write()

    t = BASE_CONFIG.context_length
    linhas = []
    for cabecas in HEADS_SWEEP:
        cfg = replace(BASE_CONFIG, n_heads=cabecas)
        model = build(cfg)
        basicos = sweep_row_basics(model, cfg, batch, t)
        celulas = cfg.n_layers * cabecas * t * t
        linhas.append(
            [
                str(cabecas),
                str(cfg.head_dim),
                basicos["parametros"],
                thousands(celulas),
                megabytes(celulas),
                basicos["ms"],
                basicos["perda"],
            ]
        )

    report.table(
        ["n_heads", "head_dim", "parâmetros", "células de scores (L·H·T²)", "scores em float32", "tempo (ms)", "perda inicial"],
        linhas,
    )


def experiment_5_emb_dim(report: Report, batch) -> None:
    """A5: a largura do modelo, e para onde vão os parâmetros."""
    report.write("## A5 - Dimensão do embedding (emb_dim)")
    report.write()
    report.write(
        "**Pergunta:** quanto custa alargar o modelo, e como muda a divisão dos "
        "parâmetros entre as tabelas (tokens, posições, cabeça) e os blocos?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG com emb_dim em "
        f"{', '.join(map(str, EMB_DIM_SWEEP))} e n_heads {BASE_CONFIG.n_heads} "
        "fixo (head_dim = emb_dim / 4). A FeedForward acompanha: 4·emb_dim. Na "
        "última linha, o GPT-2 small como referência (contado no device meta)."
    )
    report.write()

    linhas = []
    fatias = []
    for dim in EMB_DIM_SWEEP:
        cfg = replace(BASE_CONFIG, emb_dim=dim)
        model = build(cfg)
        basicos = sweep_row_basics(model, cfg, batch, cfg.context_length)
        partes = model.parameter_breakdown()
        total = model.parameter_count()
        tabelas = partes["token_embedding"] + partes["positional_embedding"] + partes["out_head"]
        linhas.append(
            [
                str(dim),
                str(cfg.head_dim),
                basicos["parametros"],
                percent(tabelas, total),
                percent(partes["attention"], total),
                percent(partes["feed_forward"], total),
                basicos["mb"],
                basicos["gflop"],
                basicos["ms"],
                basicos["perda"],
            ]
        )
        fatias.append((f"emb_dim {dim}", tabelas / total, partes["attention"] / total, partes["feed_forward"] / total))

    pequeno = on_meta(GPT_CONFIG_124M)
    partes = pequeno.parameter_breakdown()
    total = pequeno.parameter_count()
    tabelas = partes["token_embedding"] + partes["positional_embedding"] + partes["out_head"]
    linhas.append(
        [
            "768 (GPT-2 small, 12 blocos)",
            "64",
            thousands(total),
            percent(tabelas, total),
            percent(partes["attention"], total),
            percent(partes["feed_forward"], total),
            f"{pequeno.size_in_megabytes():.1f} MB",
            f"{estimated_gflops(GPT_CONFIG_124M, BASE_CONFIG.context_length)['total']:.2f}",
            "-",
            "-",
        ]
    )
    fatias.append(("GPT-2 small", tabelas / total, partes["attention"] / total, partes["feed_forward"] / total))

    report.table(
        [
            "emb_dim",
            "head_dim",
            "parâmetros",
            "tabelas + cabeça",
            "atenção",
            "FeedForward",
            "float32",
            "GFLOP",
            "tempo (ms)",
            "perda inicial",
        ],
        linhas,
    )

    figura, eixo = new_axes(7.0, 3.4)
    nomes = [nome for nome, *_ in fatias]
    posicoes = list(range(len(fatias)))[::-1]
    esquerda = [0.0] * len(fatias)
    segmentos = ("tabelas + cabeça", "atenção", "FeedForward")
    for serie, rotulo in enumerate(segmentos):
        larguras = [100 * fatia[serie + 1] for fatia in fatias]
        eixo.barh(
            posicoes,
            larguras,
            left=esquerda,
            height=0.6,
            color=SERIES_COLORS[serie],
            edgecolor=SURFACE,
            linewidth=2,
            label=rotulo,
        )
        esquerda = [e + w for e, w in zip(esquerda, larguras)]
    for posicao, (_, tabelas_frac, *_resto) in zip(posicoes, fatias):
        eixo.text(1.5, posicao, f"{100 * tabelas_frac:.0f}%", va="center", fontsize=8, color="white")
    eixo.set_yticks(posicoes, nomes)
    eixo.set_xlim(0, 100)
    eixo.set_xlabel("fração dos parâmetros (%)")
    eixo.set_title("Para onde vão os parâmetros conforme emb_dim cresce", fontsize=10)
    eixo.grid(axis="y", visible=False)
    legend(eixo, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    caminho = save_figure(figura, "arquitetura-parametros-por-componente.png")
    report.figure(caminho, "Fração dos parâmetros por componente")

    report.write(
        "Atenção e FeedForward somam todos os blocos; \"tabelas + cabeça\" são "
        "as tabelas de tokens e de posições mais a cabeça de saída. As "
        "LayerNorms (menos de 0,1% em todas as linhas) ficam fora da figura; "
        "os valores exatos estão na tabela."
    )
    report.write()


def experiment_6_feed_forward(report: Report, batch) -> None:
    """A6: a largura da FeedForward."""
    report.write("## A6 - Dimensão da FeedForward (ff_hidden_dim)")
    report.write()
    report.write(
        "**Pergunta:** o fator 4 da FeedForward pesa quanto no bloco e no custo?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG com ff_hidden_dim = fator · emb_dim, "
        f"fator em {', '.join(map(str, FF_FACTOR_SWEEP))} (4 é o do GPT-2)."
    )
    report.write()

    linhas = []
    for fator in FF_FACTOR_SWEEP:
        cfg = replace(BASE_CONFIG, ff_hidden_dim=fator * BASE_CONFIG.emb_dim)
        model = build(cfg)
        basicos = sweep_row_basics(model, cfg, batch, cfg.context_length)
        bloco = model.trf_blocks[0]
        ff = sum(p.numel() for p in bloco.ff.parameters())
        total_bloco = sum(p.numel() for p in bloco.parameters())
        flops = estimated_gflops(cfg, cfg.context_length)
        linhas.append(
            [
                str(fator),
                str(cfg.feed_forward_dim),
                thousands(ff),
                percent(ff, total_bloco),
                basicos["parametros"],
                basicos["gflop"],
                f"{flops['linear']:.2f}",
                basicos["ms"],
                basicos["perda"],
            ]
        )

    report.table(
        [
            "fator",
            "ff_hidden_dim",
            "parâmetros da FFN (1 bloco)",
            "fração do bloco",
            "parâmetros do modelo",
            "GFLOP total",
            "GFLOP dos blocos",
            "tempo (ms)",
            "perda inicial",
        ],
        linhas,
    )


def experiment_7_context(report: Report, token_ids: list[int]) -> None:
    """A7: o comprimento da sequência, e onde o custo quadrático aparece."""
    report.write("## A7 - Tamanho do contexto (context_length)")
    report.write()
    report.write(
        "**Pergunta:** a Sprint 3 mostrou que a atenção é quadrática em T. No "
        "modelo inteiro, esse termo já domina o custo?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG com context_length em "
        f"{', '.join(map(str, CONTEXT_SWEEP))}, forward de 1 sequência com T = "
        "context_length (os primeiros tokens do The Verdict). Os FLOPs são "
        "separados pela origem: projeções dos blocos (linear em T), scores e "
        "média ponderada da atenção (T²) e cabeça de saída (linear em T, vezes V)."
    )
    report.write()

    linhas = []
    for contexto in CONTEXT_SWEEP:
        cfg = replace(BASE_CONFIG, context_length=contexto)
        model = build(cfg)
        flops = estimated_gflops(cfg, contexto)
        celulas_scores = cfg.n_layers * cfg.n_heads * contexto * contexto
        celulas_logits = contexto * cfg.vocab_size

        ids = torch.tensor([token_ids[:contexto]])
        alvo = torch.tensor([token_ids[1 : contexto + 1]])
        with torch.no_grad():
            perda = cross_entropy(model(ids), alvo)

        linhas.append(
            [
                str(contexto),
                thousands(model.parameter_count()),
                thousands(model.parameter_breakdown()["positional_embedding"]),
                megabytes(celulas_scores),
                megabytes(celulas_logits),
                f"{flops['linear']:.2f}",
                f"{flops['atencao']:.2f}",
                f"{flops['cabeca']:.2f}",
                percent(flops["atencao"], flops["total"]),
                f"{forward_ms(model, contexto):.1f}",
                f"{perda:.3f}",
            ]
        )

    report.table(
        [
            "T",
            "parâmetros",
            "tabela de posições",
            "scores (L·H·T²)",
            "logits (T·V)",
            "GFLOP blocos",
            "GFLOP atenção",
            "GFLOP cabeça",
            "atenção no total",
            "tempo (ms)",
            "perda inicial",
        ],
        linhas,
    )

    cfg = BASE_CONFIG
    cruzamento = cfg.vocab_size / (2 * cfg.n_layers)
    report.write(
        f"Ponto em que os FLOPs da atenção (4·L·T²·D) igualariam os da cabeça "
        f"(2·T·D·V): T = V / (2·L) = {cruzamento:.0f} tokens, para L = "
        f"{cfg.n_layers}."
    )
    report.write()


def experiment_8_dropout(report: Report, batch) -> None:
    """A8: o que o dropout faz com a saída, em train() e em eval()."""
    report.write("## A8 - Dropout")
    report.write()
    report.write(
        "**Pergunta:** quanto o dropout mexe na saída do modelo durante o treino, "
        "e ele some mesmo em eval()? Qual dos três lugares do exercício 4.3 mais "
        "perturba a saída?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG com a taxa variando. Em train(), duas "
        "passadas sobre o mesmo lote: a variação relativa é o desvio da "
        "diferença entre elas dividido pelo desvio dos logits. A perda em "
        f"train() é a média de {DROPOUT_PASSES} passadas. Os pesos são os mesmos "
        "em todas as linhas (seed fixa): só a taxa muda."
    )
    report.write()

    entrada, alvo = batch

    def mede(cfg: GPTConfig) -> list[str]:
        model = build(cfg)
        with torch.no_grad():
            logits_eval = model(entrada)
            iguais_em_eval = torch.equal(logits_eval, model(entrada))

            model.train()
            torch.manual_seed(SEED)
            primeira, segunda = model(entrada), model(entrada)
            perdas_treino = [cross_entropy(model(entrada), alvo) for _ in range(DROPOUT_PASSES)]

        variacao = (primeira - segunda).std() / logits_eval.std()
        return [
            f"{variacao.item():.3f}",
            f"{sum(perdas_treino) / len(perdas_treino):.4f}",
            f"{cross_entropy(logits_eval, alvo):.4f}",
            "sim" if iguais_em_eval else "não",
        ]

    cabecalho = [
        "variação entre 2 passadas em train()",
        "perda em train()",
        "perda em eval()",
        "eval() determinístico",
    ]
    report.table(
        ["drop_rate"] + cabecalho,
        [[f"{p:.1f}"] + mede(replace(BASE_CONFIG, drop_rate=p)) for p in DROPOUT_SWEEP],
    )

    report.write("### As três taxas separadas (exercício 4.3), cada uma sozinha em 0,1")
    report.write()
    sozinhas = {
        "só embeddings": {"drop_rate_emb": 0.1},
        "só pesos de atenção": {"drop_rate_attn": 0.1},
        "só antes dos atalhos": {"drop_rate_shortcut": 0.1},
        "as três (padrão do livro)": {"drop_rate_emb": 0.1, "drop_rate_attn": 0.1, "drop_rate_shortcut": 0.1},
    }
    report.table(
        ["onde"] + cabecalho,
        [
            [nome] + mede(replace(BASE_CONFIG, drop_rate=0.0, **taxas))
            for nome, taxas in sozinhas.items()
        ],
    )


class NoShortcutBlock(TransformerBlock):
    """Ablação: as duas subcamadas sem o atalho. A saída de cada uma substitui x."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.att(self.norm1(x))
        x = self.drop_shortcut(attended)
        return self.drop_shortcut(self.ff(self.norm2(x)))


class NoLayerNormBlock(TransformerBlock):
    """Ablação: atalhos mantidos, LayerNorms removidas."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.att(x)
        x = x + self.drop_shortcut(attended)
        return x + self.drop_shortcut(self.ff(x))


class PlainBlock(TransformerBlock):
    """Ablação: sem atalho e sem LayerNorm -- a rede profunda "pura" da seção 4.4."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.att(x)
        x = self.drop_shortcut(attended)
        return self.drop_shortcut(self.ff(x))


class PostLayerNormBlock(TransformerBlock):
    """Ablação: a ordem do Transformer original, normalizando DEPOIS da soma."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.att(x)
        x = self.norm1(x + self.drop_shortcut(attended))
        return self.norm2(x + self.drop_shortcut(self.ff(x)))


def variant(model: GPTModel, block_class: type[TransformerBlock], drop_final_norm: bool = False) -> GPTModel:
    """Cópia do modelo com outro forward nos blocos -- mesmos pesos, mesma seed.

    As classes de ablação só redefinem forward, então trocar a classe de cada
    bloco na cópia muda a conta sem tocar em nenhum parâmetro: a comparação
    isola a ordem das operações.
    """
    copia = copy.deepcopy(model)
    for bloco in copia.trf_blocks:
        bloco.__class__ = block_class
    if drop_final_norm:
        copia.final_norm = nn.Identity()
    return copia


def experiment_9_ablation(report: Report, batch) -> None:
    """A9: o que acontece quando se tira o atalho ou a LayerNorm."""
    report.write("## A9 - Ablação do bloco: atalho e LayerNorm")
    report.write()
    report.write(
        "**Pergunta:** qual a contribuição do atalho e da LayerNorm para a "
        "estabilidade? O que muda quando cada um sai, ou quando a norma vai "
        "para depois da soma (Post-LayerNorm, como no Transformer original)?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG com {ABLATION_LAYERS} blocos e dropout 0. "
        "As cinco variantes têm os MESMOS pesos (cópia do mesmo modelo); só o "
        "forward do bloco muda. 'Sem LayerNorm' tira as duas normas de cada "
        "bloco e a final. 'Sem atalho e sem LayerNorm' é a rede profunda pura, "
        "o caso da rede de demonstração da seção 4.4. Forward e backward da "
        "cross-entropy sobre o lote do The Verdict. Gradiente por bloco: média "
        "de |grad| sobre todos os parâmetros do bloco (o critério do livro na "
        "seção 4.4)."
    )
    report.write()

    base = build(replace(BASE_CONFIG, n_layers=ABLATION_LAYERS, drop_rate=0.0))
    variantes = {
        "Pre-LN (GPT-2, o do projeto)": base,
        "sem atalho": variant(base, NoShortcutBlock),
        "sem LayerNorm": variant(base, NoLayerNormBlock, drop_final_norm=True),
        "sem atalho e sem LayerNorm": variant(base, PlainBlock, drop_final_norm=True),
        "Post-LN (Transformer original)": variant(base, PostLayerNormBlock),
    }

    entrada, alvo = batch
    escalas, gradientes, linhas = {}, {}, []
    for nome, model in variantes.items():
        model.zero_grad()
        estados, logits = residual_stream(model, entrada)
        perda = F.cross_entropy(logits.flatten(0, 1), alvo.flatten())
        perda.backward()

        escalas[nome] = [estado.std().item() for estado in estados]
        # Na variante sem LayerNorm, scale e shift das normas não entram no
        # forward e ficam sem gradiente (None): só os parâmetros usados contam.
        gradientes[nome] = [
            torch.cat(
                [p.grad.abs().flatten() for p in bloco.parameters() if p.grad is not None]
            ).mean().item()
            for bloco in model.trf_blocks
        ]
        primeiro, ultimo = gradientes[nome][0], gradientes[nome][-1]
        linhas.append(
            [
                nome,
                f"{escalas[nome][0]:.3f}",
                f"{escalas[nome][-1]:.3f}",
                f"{logits.std().item():.3f}",
                f"{perda.item():.3f}",
                f"{primeiro:.2e}",
                f"{ultimo:.2e}",
                # 3 algarismos significativos, e não 2 casas: a razão vai de
                # ~1 a ~1e-13, e "0.00" esconderia a diferença que importa.
                f"{primeiro / ultimo:.3g}",
            ]
        )

    report.table(
        [
            "variante",
            "desvio após embeddings",
            f"desvio após bloco {ABLATION_LAYERS}",
            "desvio dos logits",
            "perda inicial",
            "|grad| bloco 1",
            f"|grad| bloco {ABLATION_LAYERS}",
            "bloco 1 / último",
        ],
        linhas,
    )

    report.write("Desvio do fluxo residual depois de cada bloco (0 = depois dos embeddings):")
    report.write()
    report.table(
        ["variante"] + [str(i) for i in range(ABLATION_LAYERS + 1)],
        [[nome] + [f"{v:.2f}" for v in valores] for nome, valores in escalas.items()],
    )
    report.write("Gradiente médio absoluto por bloco:")
    report.write()
    report.table(
        ["variante"] + [str(i + 1) for i in range(ABLATION_LAYERS)],
        [[nome] + [f"{v:.1e}" for v in valores] for nome, valores in gradientes.items()],
    )

    # Rótulo direto só onde as linhas não se sobrepõem; a legenda cobre todas.
    # No gradiente, as quatro variantes estáveis ficam coladas no topo e só a
    # que despenca recebe rótulo.
    todas = set(variantes)
    for titulo, dados, eixo_x, rotulo_y, arquivo, rotuladas in (
        (
            "Escala do fluxo residual ao longo da pilha",
            escalas,
            list(range(ABLATION_LAYERS + 1)),
            "desvio padrão (escala log)",
            "arquitetura-ablacao-escala.png",
            todas,
        ),
        (
            "Gradiente que chega a cada bloco",
            gradientes,
            list(range(1, ABLATION_LAYERS + 1)),
            "média de |grad| (escala log)",
            "arquitetura-ablacao-gradiente.png",
            {"sem atalho e sem LayerNorm"},
        ),
    ):
        figura, eixo = new_axes(7.0, 3.6)
        for serie, (nome, valores) in enumerate(dados.items()):
            eixo.plot(eixo_x, valores, color=SERIES_COLORS[serie], linewidth=2, marker="o", markersize=4, label=nome)
            if nome not in rotuladas:
                continue
            eixo.annotate(
                nome.split(" (")[0],
                xy=(eixo_x[-1], valores[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK_SECONDARY,
            )
        eixo.set_yscale("log")
        eixo.set_xticks(eixo_x)
        eixo.set_xlabel("bloco")
        eixo.set_ylabel(rotulo_y)
        eixo.set_title(titulo, fontsize=10)
        eixo.set_xlim(eixo_x[0] - 0.3, eixo_x[-1] + 2.2)
        legend(eixo, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        caminho = save_figure(figura, arquivo)
        report.figure(caminho, titulo)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    bpe = BPETokenizer()
    texto = load_verdict()
    loader = create_dataloader(
        texto,
        bpe,
        context_length=BASE_CONFIG.context_length,
        stride=BASE_CONFIG.context_length,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )
    batch = next(iter(loader))

    report = Report(RESULTS_PATH)
    report.write("# Sprint 4 - Arquitetura: resultados dos experimentos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint04/arquitetura_experimentos.py`. Não "
        "editar à mão: rode o script novamente."
    )
    report.write()
    report.write(
        f"Modelos em `eval()` e sob `torch.no_grad()`, exceto o A8 (dropout em "
        f"`train()`) e o A9 (backward). Seed {SEED} antes de cada modelo. Tempos "
        f"em CPU, mediana de {REPEATS} medidas após {WARMUP} de aquecimento: "
        "valem as razões entre linhas, não os valores absolutos, e diferenças "
        "de 1-2 ms são ruído. Memória em float32 "
        f"({BYTES_PER_FLOAT32} bytes por número). Figuras em `resultados/figuras/`."
    )
    report.write()

    experiment_1_anatomy(report, batch)
    experiment_2_untrained(report, batch)
    experiment_3_layers(report, batch)
    experiment_4_heads(report, batch)
    experiment_5_emb_dim(report, batch)
    experiment_6_feed_forward(report, batch)
    experiment_7_context(report, bpe.encode(texto))
    experiment_8_dropout(report, batch)
    experiment_9_ablation(report, batch)

    report.save()
    print(f"\nResultados salvos em {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
