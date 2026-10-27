"""Experimentos de inferência e geração da Sprint 4.

O livro_exemplos.py reproduz a geração do livro com o GPT de 124M e o
vocabulário de 9 palavras da seção 5.3. Aqui as mesmas ferramentas rodam sobre
o modelo do projeto (BASE_CONFIG), para medir o que cada etapa da geração faz.

G1  Um passo de geração por dentro: logits -> probabilidades -> token
G2  O laço autoregressivo, passo a passo, e a coerência com uma passada única
G3  Temperatura: a forma da distribuição e a diversidade dos sorteios
G4  Top-k: quanta probabilidade sobra e quantos tokens podem sair
G5  Estratégias lado a lado: texto, repetição, determinismo e o papel do eval()
G6  Custo por token: o contexto cresce, enche e passa a deslizar
G7  Custo por token conforme o tamanho do modelo

O modelo NÃO foi treinado (isso é a Sprint 5): os textos gerados são
incoerentes, e o que se mede é o mecanismo, não a qualidade. Prompt: "Every
effort moves you", o da seção 5.3 do livro. Sorteios com torch.Generator
semeado (seed 42), modelos com seed 42 antes da criação.

Uso:
    python experimentos/sprint04/geracao_experimentos.py
"""

import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from relatorio import (  # noqa: E402
    INK_SECONDARY,
    RESULTS_DIR,
    SEED,
    SERIES_COLORS,
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
from src.model import BASE_CONFIG, GPT_CONFIG_124M, GPTConfig, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

RESULTS_PATH = RESULTS_DIR / "geracao.md"

PROMPT = "Every effort moves you"

G1_TOP = 5
G2_STEPS = 10
G3_TEMPERATURES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SAMPLES = 1_000
G4_TOP_K = (1, 5, 20, 100, 1_000, None)
G4_GRID_TEMPERATURES = (0.5, 1.0, 1.5)
G4_GRID_TOP_K = (5, 50, None)
G5_NEW_TOKENS = 30
G5_STRATEGIES = {
    "gulosa (T = 0)": {"temperature": 0.0},
    "top-k 5, T = 1": {"temperature": 1.0, "top_k": 5},
    "top-k 50, T = 0,7": {"temperature": 0.7, "top_k": 50},
    "T = 1, sem top-k": {"temperature": 1.0},
    "T = 1,5, sem top-k": {"temperature": 1.5},
}
G6_NEW_TOKENS = 200
G6_REPEATS = 3
G6_BIN = 25
G7_NEW_TOKENS = 20


def build(cfg: GPTConfig, seed: int = SEED) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel(cfg).eval()


def entropy(probabilities: torch.Tensor) -> float:
    """Entropia em nats; 0·log 0 tratado como 0."""
    seguras = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    # max com 0.0 primeiro: a distribuição one-hot dá -0.0, que imprimiria
    # "-0.000"; max devolve o primeiro argumento quando os dois empatam.
    return max(0.0, float(-(probabilities * seguras.log()).sum(dim=-1)))


def token_label(bpe: BPETokenizer, token_id: int) -> str:
    """Texto do token entre crases, com o espaço inicial visível."""
    return f"`{bpe.decode([token_id])!r}`"


def generator() -> torch.Generator:
    return torch.Generator().manual_seed(SEED)


def experiment_1_one_step(report: Report, model: GPTModel, bpe: BPETokenizer) -> None:
    """G1: o que acontece entre a passada do modelo e o próximo token."""
    report.write("## G1 - Um passo de geração por dentro")
    report.write()
    report.write(
        "**Pergunta:** como o vetor de logits da última posição vira um token? "
        "E o que são as outras linhas que o modelo devolve?"
    )
    report.write()

    ids = text_to_token_ids(PROMPT, bpe)
    with torch.no_grad():
        logits = model(ids)
    ultimo = logits[0, -1]
    probabilidades = torch.softmax(ultimo, dim=-1)

    report.write(
        f"**Configuração:** BASE_CONFIG (seed {SEED}, sem treino), prompt "
        f"`{PROMPT}` = {ids[0].tolist()}."
    )
    report.write()
    report.table(
        ["etapa", "shape", "observação"],
        [
            ["Token IDs do prompt", str(tuple(ids.shape)), "lote de 1 sequência de 4 tokens"],
            ["logits do modelo", str(tuple(logits.shape)), "uma linha de V logits por posição"],
            ["logits da última posição", str(tuple(ultimo.shape)), "a única linha usada na geração"],
            [
                "softmax",
                str(tuple(probabilidades.shape)),
                f"soma = {probabilidades.sum().item():.6f}; todos > 0",
            ],
            ["argmax", "()", "o próximo token na escolha gulosa"],
        ],
    )

    valores, posicoes = torch.topk(ultimo, G1_TOP)
    report.write(f"Os {G1_TOP} maiores logits da última posição, de {ultimo.numel()}:")
    report.write()
    report.table(
        ["posição no ranking", "Token ID", "token", "logit", "probabilidade", "razão para 1/V"],
        [
            [
                str(rank + 1),
                str(int(posicao)),
                token_label(bpe, int(posicao)),
                f"{valor.item():.4f}",
                f"{probabilidades[posicao].item():.3e}",
                f"{probabilidades[posicao].item() * ultimo.numel():.2f}x",
            ]
            for rank, (valor, posicao) in enumerate(zip(valores, posicoes))
        ],
    )

    mesmo_argmax = int(ultimo.argmax()) == int(probabilidades.argmax())
    deslocado = torch.softmax(ultimo + 100.0, dim=-1)
    report.write(
        f"argmax dos logits = argmax das probabilidades: {mesmo_argmax}. Somar "
        f"100 a todos os logits muda as probabilidades em no máximo "
        f"{(deslocado - probabilidades).abs().max().item():.1e}: o softmax só "
        "enxerga diferenças entre logits."
    )
    report.write()

    report.write("**Uma passada, T previsões.** Cada linha dos logits prevê o token seguinte à sua posição:")
    report.write()
    linhas = []
    for posicao in range(ids.shape[1]):
        previsto = int(logits[0, posicao].argmax())
        if posicao + 1 < ids.shape[1]:
            real = int(ids[0, posicao + 1])
            prob_real = torch.softmax(logits[0, posicao], dim=-1)[real].item()
            real_txt, prob_txt = token_label(bpe, real), f"{prob_real:.2e}"
        else:
            real_txt, prob_txt = "(ainda não existe: é o que se gera)", "-"
        linhas.append(
            [
                str(posicao),
                token_label(bpe, int(ids[0, posicao])),
                token_label(bpe, previsto),
                real_txt,
                prob_txt,
            ]
        )
    report.table(
        ["posição", "token de entrada", "argmax da linha", "token seguinte de fato", "prob. dada a ele"],
        linhas,
    )


def experiment_2_loop(report: Report, model: GPTModel, bpe: BPETokenizer) -> None:
    """G2: o laço autoregressivo e a coerência com uma passada só."""
    report.write("## G2 - O laço autoregressivo")
    report.write()
    report.write(
        "**Pergunta:** o que muda a cada passo do laço, e a geração token a "
        "token concorda com uma passada única sobre o texto final?"
    )
    report.write()
    report.write(
        f"**Configuração:** decodificação gulosa, {G2_STEPS} passos. Cada linha "
        "é uma chamada de `generate(..., max_new_tokens=1)` sobre o contexto da "
        "linha anterior; as probabilidades são o softmax da última linha de "
        "logits daquele passo."
    )
    report.write()

    contexto = text_to_token_ids(PROMPT, bpe)
    logits_por_passo = []
    linhas = []
    for passo in range(1, G2_STEPS + 1):
        with torch.no_grad():
            ultimo = model(contexto)[0, -1]
        probabilidades = torch.softmax(ultimo, dim=-1)
        seguinte = generate(model, contexto, max_new_tokens=1)
        escolhido = int(seguinte[0, -1])

        logits_por_passo.append(ultimo)
        linhas.append(
            [
                str(passo),
                str(contexto.shape[1]),
                str(escolhido),
                token_label(bpe, escolhido),
                f"{probabilidades[escolhido].item():.2e}",
                f"{entropy(probabilidades):.3f}",
            ]
        )
        contexto = seguinte

    report.table(
        ["passo", "tokens no contexto", "Token ID escolhido", "token", "probabilidade", "entropia (nats)"],
        linhas,
    )
    report.write(f"Texto final: `{token_ids_to_text(contexto, bpe)}`")
    report.write()

    tamanho_prompt = len(bpe.encode(PROMPT))
    with torch.no_grad():
        completos = model(contexto[:, :-1])[0]
    previstos = completos[tamanho_prompt - 1 :]
    diferenca = (torch.stack(logits_por_passo) - previstos).abs().max().item()
    mesmos_tokens = torch.equal(previstos.argmax(dim=-1), contexto[0, tamanho_prompt:])
    processados = sum(tamanho_prompt + passo for passo in range(G2_STEPS))

    report.table(
        ["verificação", "resultado"],
        [
            [
                "logits de cada passo contra a mesma posição numa passada única",
                f"diferença máxima {diferenca:.1e}",
            ],
            ["argmax da passada única = tokens gerados pelo laço", str(mesmos_tokens)],
            [
                "tokens processados pelo modelo no laço",
                f"{processados} para gerar {G2_STEPS} (contexto refeito a cada passo)",
            ],
        ],
    )


def experiment_3_temperature(report: Report, model: GPTModel, bpe: BPETokenizer) -> None:
    """G3: o que a temperatura faz com a distribuição do modelo."""
    report.write("## G3 - Temperatura")
    report.write()
    report.write(
        "**Pergunta:** como a temperatura muda a distribuição do próximo token "
        "do modelo, e quantos tokens diferentes a amostragem passa a produzir?"
    )
    report.write()
    report.write(
        f"**Configuração:** logits da última posição do prompt (os do G1). Para "
        f"cada temperatura: probabilidade do token mais provável, entropia, "
        "número efetivo de candidatos (e^entropia: o tamanho de um vocabulário "
        f"uniforme com a mesma entropia) e {SAMPLES} sorteios com o gerador "
        f"semeado. Temperatura 0 é a escolha gulosa. Referência: ln V = "
        f"{math.log(BASE_CONFIG.vocab_size):.3f} nats."
    )
    report.write()

    ids = text_to_token_ids(PROMPT, bpe)
    with torch.no_grad():
        ultimo = model(ids)[0, -1]
    argmax = int(ultimo.argmax())

    linhas = [["0 (gulosa)", "1.0000", "0.000", "1", "1", "100.0%"]]
    efetivos, distintos = [], []
    for temperatura in G3_TEMPERATURES:
        probabilidades = next_token_probabilities(ultimo, temperature=temperatura)
        sorteios = torch.multinomial(probabilidades, SAMPLES, replacement=True, generator=generator())
        h = entropy(probabilidades)
        efetivos.append(math.exp(h))
        distintos.append(len(set(sorteios.tolist())))
        linhas.append(
            [
                f"{temperatura:g}",
                f"{probabilidades.max().item():.4f}",
                f"{h:.3f}",
                f"{math.exp(h):.0f}",
                str(distintos[-1]),
                f"{100 * (sorteios == argmax).float().mean().item():.1f}%",
            ]
        )

    report.table(
        [
            "temperatura",
            "maior probabilidade",
            "entropia (nats)",
            "candidatos efetivos",
            f"tokens distintos em {SAMPLES}",
            "sorteios = argmax",
        ],
        linhas,
    )

    figura, eixo = new_axes(6.6, 3.4)
    eixo.plot(G3_TEMPERATURES, efetivos, color=SERIES_COLORS[0], linewidth=2, marker="o", markersize=5, label="candidatos efetivos (e^entropia)")
    eixo.plot(G3_TEMPERATURES, distintos, color=SERIES_COLORS[1], linewidth=2, marker="o", markersize=5, label=f"tokens distintos em {SAMPLES} sorteios")
    eixo.axhline(BASE_CONFIG.vocab_size, color=INK_SECONDARY, linewidth=0.8)
    # Rótulo acima da linha de referência, onde não há dado para encobrir.
    eixo.annotate(
        f"vocabulário inteiro ({thousands(BASE_CONFIG.vocab_size)})",
        xy=(G3_TEMPERATURES[0], BASE_CONFIG.vocab_size),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
    )
    eixo.set_yscale("log")
    eixo.set_ylim(top=4 * BASE_CONFIG.vocab_size)
    eixo.set_xlabel("temperatura")
    eixo.set_ylabel("número de tokens (escala log)")
    eixo.set_title("Temperatura e diversidade no modelo sem treino", fontsize=10)
    # Entre as duas curvas, onde o gráfico está vazio.
    legend(eixo, loc="center right")
    caminho = save_figure(figura, "geracao-temperatura.png")
    report.figure(caminho, "Temperatura e diversidade")

    top = torch.topk(ultimo, G1_TOP).indices
    temperaturas = (0.25, 0.5, 1.0, 2.0)
    report.write(f"Os {G1_TOP} tokens mais prováveis, com a probabilidade de cada um em cada temperatura:")
    report.write()
    report.table(
        ["token"] + [f"T = {t:g}" for t in temperaturas],
        [
            [token_label(bpe, int(t))]
            + [f"{next_token_probabilities(ultimo, temperature=temp)[t].item():.2e}" for temp in temperaturas]
            for t in top
        ],
    )


def experiment_4_top_k(report: Report, model: GPTModel, bpe: BPETokenizer) -> None:
    """G4: o que o top-k corta e o que sobra."""
    report.write("## G4 - Top-k")
    report.write()
    report.write(
        "**Pergunta:** quanto da distribuição o top-k descarta, e quanto ele "
        "limita a diversidade dos sorteios, sozinho e combinado com a temperatura?"
    )
    report.write()
    report.write(
        f"**Configuração:** os mesmos logits do G1. Massa mantida = soma das "
        f"probabilidades originais (T = 1) dos k tokens mantidos. {SAMPLES} "
        "sorteios com o gerador semeado. `None` = sem top-k (todo o vocabulário)."
    )
    report.write()

    ids = text_to_token_ids(PROMPT, bpe)
    with torch.no_grad():
        ultimo = model(ids)[0, -1]
    original = torch.softmax(ultimo, dim=-1)

    linhas = []
    for k in G4_TOP_K:
        probabilidades = next_token_probabilities(ultimo, temperature=1.0, top_k=k)
        mantidos = torch.isfinite(apply_top_k(ultimo, k)) if k is not None else torch.ones_like(ultimo, dtype=torch.bool)
        sorteios = torch.multinomial(probabilidades, SAMPLES, replacement=True, generator=generator())
        linhas.append(
            [
                "None" if k is None else str(k),
                f"{100 * original[mantidos].sum().item():.2f}%",
                f"{entropy(probabilidades):.3f}",
                f"{probabilidades.max().item():.4f}",
                str(len(set(sorteios.tolist()))),
            ]
        )
    report.table(
        ["top_k", "massa original mantida", "entropia (nats)", "maior probabilidade", f"tokens distintos em {SAMPLES}"],
        linhas,
    )

    report.write(f"Tokens distintos em {SAMPLES} sorteios, combinando temperatura e top-k:")
    report.write()
    linhas = []
    for temperatura in G4_GRID_TEMPERATURES:
        linha = [f"T = {temperatura:g}"]
        for k in G4_GRID_TOP_K:
            probabilidades = next_token_probabilities(ultimo, temperature=temperatura, top_k=k)
            sorteios = torch.multinomial(probabilidades, SAMPLES, replacement=True, generator=generator())
            linha.append(str(len(set(sorteios.tolist()))))
        linhas.append(linha)
    report.table(
        ["temperatura"] + [f"top_k = {k}" for k in G4_GRID_TOP_K],
        linhas,
    )


def longest_run(tokens: list[int]) -> int:
    """Maior sequência do mesmo token repetido em seguida."""
    maior = atual = 1
    for anterior, seguinte in zip(tokens, tokens[1:]):
        atual = atual + 1 if seguinte == anterior else 1
        maior = max(maior, atual)
    return maior


def experiment_5_strategies(report: Report, model: GPTModel, bpe: BPETokenizer) -> None:
    """G5: as estratégias lado a lado, e por que a geração roda em eval()."""
    report.write("## G5 - Estratégias lado a lado")
    report.write()
    report.write(
        "**Pergunta:** o que cada estratégia produz a partir do mesmo prompt, "
        "quais são reprodutíveis, e o que acontece se a geração rodar com o "
        "dropout ligado?"
    )
    report.write()
    report.write(
        f"**Configuração:** {G5_NEW_TOKENS} tokens novos a partir de `{PROMPT}`. "
        "Repetição = maior sequência do mesmo token em seguida. Log-prob média = "
        "média de ln p(token) segundo o próprio modelo (T = 1, sem top-k), "
        "calculada numa passada sobre o texto final: quanto mais perto de 0, "
        "mais \"provável\" o texto é para o modelo. Reprodutibilidade: a mesma "
        "estratégia com o gerador na mesma seed, e com outra seed."
    )
    report.write()

    ids = text_to_token_ids(PROMPT, bpe)
    inicio = ids.shape[1]

    def gera(opcoes: dict, seed: int = SEED) -> torch.Tensor:
        return generate(
            model, ids, G5_NEW_TOKENS, generator=torch.Generator().manual_seed(seed), **opcoes
        )

    linhas, textos = [], []
    for nome, opcoes in G5_STRATEGIES.items():
        saida = gera(opcoes)
        novos = saida[0, inicio:].tolist()
        with torch.no_grad():
            log_probs = torch.log_softmax(model(saida[:, :-1])[0, inicio - 1 :], dim=-1)
        media = log_probs.gather(-1, saida[0, inicio:].unsqueeze(-1)).mean().item()

        linhas.append(
            [
                nome,
                f"{len(set(novos))}/{G5_NEW_TOKENS}",
                str(longest_run(novos)),
                f"{media:.3f}",
                "sim" if torch.equal(saida, gera(opcoes)) else "não",
                "sim" if torch.equal(saida, gera(opcoes, SEED + 1)) else "não",
            ]
        )
        textos.append((nome, token_ids_to_text(saida, bpe)))

    report.table(
        [
            "estratégia",
            "tokens distintos",
            "maior repetição",
            "log-prob média",
            "igual com a mesma seed",
            "igual com outra seed",
        ],
        linhas,
    )
    report.write(f"Referência: ln(1/V) = {-math.log(BASE_CONFIG.vocab_size):.3f}, a log-prob de um token sorteado uniformemente.")
    report.write()

    report.write("Textos gerados (o prompt é o começo de todos):")
    report.write()
    for nome, texto in textos:
        report.write(f"- **{nome}:** `{texto!r}`")
    report.write()

    report.write("### Dropout ligado durante a geração")
    report.write()
    report.write(
        f"Decodificação gulosa de {G5_NEW_TOKENS} tokens, duas vezes, com o "
        "modelo em `train()` (drop_rate 0.1) num laço manual -- o mesmo laço "
        "de `generate`, sem o `eval()` que ela impõe. Depois, as mesmas duas "
        "execuções por `generate`."
    )
    report.write()

    def laco_em_train() -> list[int]:
        model.train()
        contexto = ids
        with torch.no_grad():
            for _ in range(G5_NEW_TOKENS):
                ultimo = model(contexto[:, -model.context_length :])[:, -1, :]
                contexto = torch.cat([contexto, select_next_token(ultimo)], dim=1)
        model.eval()
        return contexto[0, inicio:].tolist()

    torch.manual_seed(SEED)
    primeira, segunda = laco_em_train(), laco_em_train()
    gulosa_eval = generate(model, ids, G5_NEW_TOKENS)[0, inicio:].tolist()
    diferentes = sum(a != b for a, b in zip(primeira, segunda))
    diferentes_eval = sum(
        a != b for a, b in zip(gulosa_eval, generate(model, ids, G5_NEW_TOKENS)[0, inicio:].tolist())
    )
    diverge_em = next((i + 1 for i, (a, b) in enumerate(zip(primeira, segunda)) if a != b), None)

    report.table(
        ["modo", f"posições diferentes entre as 2 execuções (de {G5_NEW_TOKENS})", "primeira divergência"],
        [
            ["laço manual em train()", str(diferentes), f"token {diverge_em}" if diverge_em else "-"],
            ["generate() (força eval())", str(diferentes_eval), "-"],
        ],
    )


def experiment_6_cost(report: Report, model: GPTModel, bpe: BPETokenizer) -> None:
    """G6: o custo de cada passo enquanto o contexto cresce e depois desliza."""
    report.write("## G6 - Custo por token e a janela de contexto")
    report.write()
    report.write(
        "**Pergunta:** quanto custa cada token gerado, e o que muda quando o "
        "texto passa do context_length?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG (context_length "
        f"{BASE_CONFIG.context_length}), {G6_NEW_TOKENS} tokens gulosos a partir "
        f"do prompt de {len(bpe.encode(PROMPT))} tokens. Cada passo é uma "
        f"chamada de `generate(..., max_new_tokens=1)`, cronometrada "
        f"{G6_REPEATS} vezes (mediana). Tempos em CPU: valem as razões."
    )
    report.write()

    contexto = text_to_token_ids(PROMPT, bpe)
    janela = BASE_CONFIG.context_length
    tempos, tamanhos = [], []
    for _ in range(G6_NEW_TOKENS):
        medidas = []
        for _ in range(G6_REPEATS):
            inicio = time.perf_counter()
            seguinte = generate(model, contexto, max_new_tokens=1)
            medidas.append((time.perf_counter() - inicio) * 1000)
        tempos.append(statistics.median(medidas))
        tamanhos.append(min(contexto.shape[1], janela))
        contexto = seguinte

    cheio = next(passo for passo, tamanho in enumerate(tamanhos, start=1) if tamanho == janela)

    linhas = []
    for inicio in range(0, G6_NEW_TOKENS, G6_BIN):
        fim = min(inicio + G6_BIN, G6_NEW_TOKENS)
        linhas.append(
            [
                f"{inicio + 1}-{fim}",
                f"{tamanhos[inicio]}-{tamanhos[fim - 1]}",
                f"{statistics.median(tempos[inicio:fim]):.1f}",
            ]
        )
    report.table(["passos", "tokens que o modelo enxerga", "mediana (ms/passo)"], linhas)

    processados = sum(tamanhos)
    report.write(
        f"O contexto enche no passo {cheio} ({janela} tokens). Total: "
        f"{len(tamanhos)} tokens gerados, {thousands(processados)} posições "
        f"processadas pelo modelo -- {processados / len(tamanhos):.0f} por token "
        f"novo. Texto final com {contexto.shape[1]} tokens, dos quais o modelo "
        f"só enxerga os últimos {janela}."
    )
    report.write()

    figura, eixo = new_axes(6.8, 3.2)
    passos = list(range(1, G6_NEW_TOKENS + 1))
    eixo.plot(passos, tempos, color=SERIES_COLORS[0], linewidth=1.5)
    eixo.axvline(cheio, color=INK_SECONDARY, linewidth=0.8)
    # Folga no topo para o rótulo da linha vertical não cobrir a curva.
    topo = max(tempos) * 1.35
    eixo.annotate(
        f"contexto cheio ({janela} tokens):\na janela passa a deslizar",
        xy=(cheio, topo),
        xytext=(6, -4),
        textcoords="offset points",
        va="top",
        fontsize=8,
        color=INK_SECONDARY,
    )
    eixo.set_ylim(0, topo)
    eixo.set_xlabel("passo de geração")
    eixo.set_ylabel("ms por passo")
    eixo.set_title("Custo de cada token gerado (BASE_CONFIG, CPU)", fontsize=10)
    caminho = save_figure(figura, "geracao-custo-por-passo.png")
    report.figure(caminho, "Custo por passo de geração")


def experiment_7_model_size(report: Report, bpe: BPETokenizer) -> None:
    """G7: o mesmo laço, modelos de tamanhos diferentes."""
    report.write("## G7 - Custo por token e tamanho do modelo")
    report.write()
    report.write(
        "**Pergunta:** quanto o tamanho do modelo pesa no tempo de cada token?"
    )
    report.write()
    report.write(
        f"**Configuração:** {G7_NEW_TOKENS} tokens gulosos a partir do prompt, "
        "depois de uma geração curta de aquecimento. Tempo total dividido pelo "
        "número de tokens."
    )
    report.write()

    ids = text_to_token_ids(PROMPT, bpe)
    modelos = {
        "BASE_CONFIG (4 blocos)": BASE_CONFIG,
        "BASE_CONFIG com 12 blocos": replace(BASE_CONFIG, n_layers=12),
        "GPT-2 small (124M)": GPT_CONFIG_124M,
    }
    linhas = []
    for nome, cfg in modelos.items():
        model = build(cfg)
        generate(model, ids, 2)
        inicio = time.perf_counter()
        generate(model, ids, G7_NEW_TOKENS)
        total = (time.perf_counter() - inicio) * 1000
        linhas.append(
            [
                nome,
                thousands(model.parameter_count()),
                f"{model.size_in_megabytes():.0f} MB",
                f"{total / G7_NEW_TOKENS:.1f}",
                f"{1000 * G7_NEW_TOKENS / total:.0f}",
            ]
        )
        del model
    report.table(["modelo", "parâmetros", "float32", "ms por token", "tokens por segundo"], linhas)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    bpe = BPETokenizer()
    model = build(BASE_CONFIG)

    report = Report(RESULTS_PATH)
    report.write("# Sprint 4 - Inferência e geração: resultados dos experimentos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint04/geracao_experimentos.py`. Não editar "
        "à mão: rode o script novamente."
    )
    report.write()
    report.write(
        f"Modelo: BASE_CONFIG com seed {SEED}, **sem treino** -- os textos "
        "gerados são incoerentes, e o que se mede é o mecanismo. Prompt: "
        f"`{PROMPT}` (seção 5.3 do livro). Sorteios com `torch.Generator` "
        f"semeado em {SEED}; a única fonte de variação entre execuções são as "
        "colunas de tempo. Figuras em `resultados/figuras/`."
    )
    report.write()

    experiment_1_one_step(report, model, bpe)
    experiment_2_loop(report, model, bpe)
    experiment_3_temperature(report, model, bpe)
    experiment_4_top_k(report, model, bpe)
    experiment_5_strategies(report, model, bpe)
    experiment_6_cost(report, model, bpe)
    experiment_7_model_size(report, bpe)

    report.save()
    print(f"\nResultados salvos em {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
