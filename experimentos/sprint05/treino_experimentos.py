"""Experimentos de treinamento da Sprint 5 (Entregáveis 3 e 4).

T1  O treino de referência: hiperparâmetros, log das perdas, amostras e a
    variação entre seeds
T2  Número de épocas: um treino de 30 épocas, e quanto o modelo recita do
    texto de treino e do de validação a cada etapa
T3  Taxa de aprendizado
T4  Tamanho do lote
T5  Regularização: dropout e weight decay
T6  Sobreajuste: treino x validação, e a validação tirada do começo do conto
T7  Janelas sobrepostas: o stride do treino e a memorização de posições
T8  Custo computacional: tempo por passo, memória, FLOPs, CPU x GPU

Modelo: BASE_CONFIG. Treino de referência: BASE_TRAINING_CONFIG (AdamW, lr
4e-4, weight decay 0,1, lotes de 2, 10 épocas, avaliação a cada 5 passos sobre
os conjuntos inteiros, seed 42). Cada experimento muda UM hiperparâmetro e
mantém os outros, com dataclasses.replace; com a mesma seed, os pesos iniciais
e a ordem dos lotes são os mesmos em todas as variantes.

Todo treino roda em CPU (ver experimentos/sprint05/comum.py); a GPU, se
houver, aparece só no T8. Rodar de novo reproduz todos os números, exceto as
colunas de tempo. Leva ~22 minutos.

Uso:
    python experimentos/sprint05/treino_experimentos.py
    python experimentos/sprint05/treino_experimentos.py --rapido   # versão curta,
        só para conferir que o script roda: relatório e figuras vão para uma
        pasta temporária
"""

import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import comum  # noqa: E402
from comum import (  # noqa: E402
    DEVICE,
    INK_MUTED,
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
from src.generation import generate  # noqa: E402
from src.model import BASE_CONFIG, GPTConfig, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.training import (  # noqa: E402
    BASE_TRAINING_CONFIG,
    BOOK_MODEL_CONFIG,
    TrainingConfig,
    TrainingHistory,
    TrainingRun,
    batch_loss,
    create_optimizer,
    create_train_val_loaders,
    evaluate,
    loader_loss,
    perplexity,
    split_text,
    train_gpt,
    train_model,
)

RAPIDO = "--rapido" in sys.argv

MODEL_CONFIG = BASE_CONFIG
BASELINE = BASE_TRAINING_CONFIG if not RAPIDO else replace(BASE_TRAINING_CONFIG, num_epochs=2)

T1_EXTRA_SEEDS = (123, 7)
T2_EPOCHS = 30 if not RAPIDO else 3
# A sonda gera 32 tokens para cada janela e custa alguns segundos: roda nas
# épocas da tabela, não em todas.
T2_TABLE_EPOCHS = (1, 2, 3, 5, 10, 15, 20, 25, 30) if not RAPIDO else (1, 2, 3)
T2_PROBE_EPOCHS = (0, *T2_TABLE_EPOCHS)
T3_LEARNING_RATES = (1e-5, 1e-4, 4e-4, 1e-3, 4e-3)
T4_BATCH_SIZES = (1, 2, 4, 8)
T5_DROPOUTS = (0.0, 0.1, 0.3)
T5_WEIGHT_DECAYS = (0.0, 0.1, 1.0)
# Sonda de memorização (T2): a partir de 32 tokens reais, a geração gulosa
# reproduz quantos dos 32 seguintes antes do primeiro erro? Em três grupos de
# trechos: do treino ALINHADOS com as janelas do treino (começam num múltiplo
# de 128, onde começa uma janela), do treino DESLOCADOS meia janela (o mesmo
# texto, 64 tokens adiante, em posições em que o modelo nunca o viu) e da
# validação.
PROBE_WINDOWS = 9
PROBE_PROMPT = 32
PROBE_CONTINUATION = 32
PROBE_SHIFT = 64
PROBE_GROUPS = ("treino alinhado", "treino deslocado", "validação")
# Janelas sobrepostas (T7): stride do treino, e o deslocamento da grade usada
# para medir a perda do texto de treino fora das posições em que ele foi visto.
T7_STRIDES = (128, 64, 32)
T7_SHIFT = 16
T7_EVAL_BATCH = 8
# Medição de custo (T8).
T8_WARMUP = 3
T8_STEPS = 10 if not RAPIDO else 3
BYTES_PER_PARAMETER_TRAINING = 16  # pesos + gradientes + 2 momentos do AdamW, float32


@dataclass
class Run:
    """Um treino e o que se mede dele."""

    label: str
    cfg: TrainingConfig
    run: TrainingRun
    seconds: float

    @property
    def history(self) -> TrainingHistory:
        return self.run.history

    @property
    def steps_per_epoch(self) -> int:
        return len(self.run.train_loader)


def train(label: str, cfg: TrainingConfig, text: str, bpe: BPETokenizer, model_cfg: GPTConfig = MODEL_CONFIG) -> Run:
    inicio = time.perf_counter()
    run = train_gpt(model_cfg, cfg, text, bpe, device=DEVICE, log=None)
    segundos = time.perf_counter() - inicio
    historico = run.history
    melhor = historico.best_evaluation()
    print(
        f"  {label}: {segundos:.0f} s, validação final {historico.val_losses[-1]:.3f}, "
        f"mínima {historico.val_losses[melhor]:.3f} (época {historico.epochs[melhor]})"
    )
    return Run(label, cfg, run, segundos)


def last_eval_of_epoch(history: TrainingHistory) -> dict[int, int]:
    """Época -> índice da última avaliação feita nela."""
    ultimo: dict[int, int] = {}
    for indice, epoca in enumerate(history.epochs):
        ultimo[epoca] = indice
    return ultimo


def training_seconds(history: TrainingHistory) -> float:
    """Tempo dos passos de treino, sem avaliações nem amostras."""
    return sum(history.epoch_seconds) - sum(history.monitor_seconds)


def continuation(sample: str, bpe: BPETokenizer, prompt: str) -> list[int]:
    return bpe.encode(sample)[len(bpe.encode(prompt)):]


def ppl(loss: float) -> str:
    """Perplexidade com uma casa e espaço como separador de milhar."""
    return f"{perplexity(loss):,.1f}".replace(",", " ")


def max_rise(values: list[float]) -> float:
    """Maior subida entre avaliações consecutivas: o tamanho do pior pico."""
    return max([0.0] + [b - a for a, b in zip(values, values[1:])])


def probe_starts(train_length: int, val_length: int) -> dict[str, list[int]]:
    """Inícios dos trechos da sonda, por grupo (ver PROBE_GROUPS).

    Alinhados: o começo de uma janela do treino a cada 4 (tokens 0, 512, ...).
    Deslocados: os mesmos, PROBE_SHIFT tokens adiante. Validação: espalhados
    pelo texto de validação inteiro, que o modelo nunca viu em posição nenhuma.
    """
    contexto = MODEL_CONFIG.context_length
    trecho = PROBE_PROMPT + PROBE_CONTINUATION
    janelas = (train_length - contexto - 1) // contexto + 1
    passo = max(1, janelas // PROBE_WINDOWS)
    alinhados = [contexto * k for k in range(0, janelas, passo)][:PROBE_WINDOWS]
    deslocados = [s + PROBE_SHIFT for s in alinhados if s + PROBE_SHIFT + trecho <= train_length]
    ultimo = val_length - trecho
    validacao = [round(i * ultimo / (PROBE_WINDOWS - 1)) for i in range(PROBE_WINDOWS)]
    return dict(zip(PROBE_GROUPS, (alinhados, deslocados, validacao)))


@torch.no_grad()
def recitation(model: GPTModel, token_ids: list[int], inicios: list[int]) -> float:
    """Média de tokens que a geração gulosa reproduz igual ao texto antes do
    primeiro erro, a partir de PROBE_PROMPT tokens reais que começam em cada um
    dos `inicios`. Gulosa: não sorteia."""
    prompts = torch.tensor([token_ids[s : s + PROBE_PROMPT] for s in inicios], device=DEVICE)
    reais = [token_ids[s + PROBE_PROMPT : s + PROBE_PROMPT + PROBE_CONTINUATION] for s in inicios]
    gerados = generate(model, prompts, PROBE_CONTINUATION)[:, PROBE_PROMPT:].tolist()

    acertos = []
    for gerado, real in zip(gerados, reais):
        n = 0
        while n < len(real) and gerado[n] == real[n]:
            n += 1
        acertos.append(n)
    return sum(acertos) / len(acertos)


def hyperparameter_rows(model_cfg: GPTConfig, cfg: TrainingConfig, optimizer: torch.optim.Optimizer, model: GPTModel) -> list[list[str]]:
    """As tabelas de registro: arquitetura, treino e otimizador, tirados dos
    próprios objetos -- o que está aqui é o que rodou."""
    linhas = [["modelo", nome, str(valor)] for nome, valor in asdict(model_cfg).items()]
    linhas.append(["modelo", "parâmetros", thousands(model.parameter_count())])
    for nome, valor in asdict(cfg).items():
        if nome == "eval_iter" and valor is None:
            valor = "None (conjuntos inteiros)"
        if nome == "stride" and valor is None:
            valor = f"None (= context_length = {model_cfg.context_length})"
        linhas.append(["treino", nome, str(valor)])
    grupo = optimizer.param_groups[0]
    linhas.append(["otimizador", "classe", type(optimizer).__name__])
    for nome in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
        linhas.append(["otimizador", nome, str(grupo[nome])])
    return linhas


def experiment_1_reference(report: Report, text: str, bpe: BPETokenizer) -> Run:
    """T1: o treino de referência, registrado por inteiro."""
    report.write("## T1 - O treino de referência")
    report.write()
    report.write(
        "**Pergunta:** o laço de treino está aprendendo a tarefa de prever o "
        "próximo token? Com que hiperparâmetros, e quanto varia de uma seed para outra?"
    )
    report.write()

    referencia = train("referência (seed 42)", BASELINE, text, bpe)
    historico = referencia.history
    passos = referencia.steps_per_epoch
    treino_loader, val_loader = referencia.run.train_loader, referencia.run.val_loader

    report.write("**Configuração registrada** (lida dos objetos usados no treino):")
    report.write()
    report.table(
        ["grupo", "hiperparâmetro", "valor"],
        hyperparameter_rows(MODEL_CONFIG, BASELINE, referencia.run.optimizer, referencia.run.model),
    )
    report.write(
        f"Dados: {len(treino_loader.dataset)} janelas de treino -> {passos} lotes por "
        f"época ({thousands(passos * BASELINE.batch_size * MODEL_CONFIG.context_length)} tokens), "
        f"{len(val_loader.dataset)} janelas de validação em {len(val_loader)} lotes. "
        f"{historico.global_step + 1} passos de otimização ao todo."
    )
    report.write()

    report.write("**Log do treino** (perdas avaliadas em `eval()`, sobre os conjuntos inteiros):")
    report.write()
    report.table(
        ["época", "passo", "tokens vistos", "treino", "validação", "perplexidade validação"],
        [
            [
                str(historico.epochs[i]),
                f"{passo:06d}",
                thousands(historico.tokens_seen[i]),
                f"{historico.train_losses[i]:.3f}",
                f"{historico.val_losses[i]:.3f}",
                ppl(historico.val_losses[i]),
            ]
            for i, passo in enumerate(historico.steps)
        ],
    )

    report.write("**Amostra gulosa do fim de cada época** (contexto `Every effort moves you`, 50 tokens):")
    report.write()
    report.table(
        ["época", "amostra"],
        [[str(e), compact(a)] for e, a in enumerate(historico.samples, start=1)],
    )

    figura, eixo = new_axes(7.0, 3.8)
    x_passos = [(passo + 1) / passos for passo in range(len(historico.batch_losses))]
    eixo.plot(x_passos, historico.batch_losses, color=INK_MUTED, linewidth=0.7, alpha=0.6,
              label="perda do lote (com dropout)")
    loss_curves(eixo, historico, passos, SERIES_COLORS[0])
    eixo.set_ylim(bottom=0)
    eixo.set_title("Treino de referência: BASE_CONFIG, lr 4e-4, lotes de 2", fontsize=10)
    legend(eixo, loc="upper right")
    report.figure(save_figure(figura, "treino-referencia.png"), "Perdas do treino de referência")

    treino_s = training_seconds(historico)
    tokens = historico.total_tokens
    report.write(
        f"Tempo: {referencia.seconds:.1f} s no total; {treino_s:.1f} s nos passos de treino "
        f"({thousands(round(tokens / treino_s))} tokens/s) e "
        f"{sum(historico.monitor_seconds):.1f} s em avaliações e amostras "
        f"({sum(historico.monitor_seconds) / sum(historico.epoch_seconds):.0%} do tempo das épocas)."
    )
    report.write()

    report.write("**Variação entre seeds** (mesma configuração; a seed muda os pesos iniciais, a ordem dos lotes e o dropout):")
    report.write()
    linhas = []
    resultados = [referencia]
    for seed in T1_EXTRA_SEEDS:
        resultados.append(train(f"seed {seed}", replace(BASELINE, seed=seed), text, bpe))
    for item in resultados:
        h = item.history
        melhor = h.best_evaluation()
        linhas.append(
            [
                str(item.cfg.seed),
                f"{h.val_losses[0]:.3f}",
                f"{h.val_losses[melhor]:.3f}",
                str(h.epochs[melhor]),
                f"{h.train_losses[-1]:.3f}",
                f"{h.val_losses[-1]:.3f}",
            ]
        )
    minimos = [r.history.val_losses[r.history.best_evaluation()] for r in resultados]
    finais = [r.history.val_losses[-1] for r in resultados]
    linhas.append(
        [
            "desvio entre seeds",
            "",
            f"{statistics.stdev(minimos):.3f}",
            "",
            f"{statistics.stdev([r.history.train_losses[-1] for r in resultados]):.3f}",
            f"{statistics.stdev(finais):.3f}",
        ]
    )
    report.table(
        ["seed", "validação no passo 0", "validação mínima", "época da mínima", "treino final", "validação final"],
        linhas,
    )
    return referencia


def train_with_probes(cfg: TrainingConfig, text: str, bpe: BPETokenizer):
    """Treina época a época (retomando o histórico) e mede a recitação nas
    épocas de T2_PROBE_EPOCHS, dentro de fork_rng. O treino sai idêntico ao de
    uma chamada só de train_gpt."""
    train_loader, val_loader = create_train_val_loaders(
        text, bpe, context_length=MODEL_CONFIG.context_length, batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio, stride=cfg.stride,
    )
    ids_treino = train_loader.dataset.token_ids.tolist()
    ids_validacao = val_loader.dataset.token_ids.tolist()

    torch.manual_seed(cfg.seed)
    model = GPTModel(MODEL_CONFIG).to(DEVICE)
    optimizer = create_optimizer(model, cfg)

    inicios = probe_starts(len(ids_treino), len(ids_validacao))
    textos = {"treino alinhado": ids_treino, "treino deslocado": ids_treino, "validação": ids_validacao}
    sondas: dict[int, dict[str, float]] = {}

    def sondar(epoca: int) -> None:
        if epoca in T2_PROBE_EPOCHS:
            with torch.random.fork_rng(devices=[]):
                sondas[epoca] = {
                    grupo: recitation(model, textos[grupo], inicios[grupo]) for grupo in PROBE_GROUPS
                }

    inicio = time.perf_counter()
    sondar(0)
    historico = None
    for epoca in range(1, cfg.num_epochs + 1):
        historico = train_model(
            model, train_loader, val_loader, optimizer,
            num_epochs=1, eval_freq=cfg.eval_freq, eval_iter=cfg.eval_iter,
            tokenizer=bpe, start_context=cfg.start_context, sample_tokens=cfg.sample_tokens,
            history=historico, log=None,
        )
        sondar(epoca)
    segundos = time.perf_counter() - inicio
    print(f"  {cfg.num_epochs} épocas com sondas: {segundos:.0f} s")
    return Run(f"{cfg.num_epochs} épocas", cfg, TrainingRun(model, optimizer, historico, train_loader, val_loader), segundos), sondas


def experiment_2_epochs(report: Report, text: str, bpe: BPETokenizer, referencia: Run):
    """T2: o mesmo treino, levado a 30 épocas."""
    report.write(f"## T2 - Número de épocas")
    report.write()
    report.write(
        "**Pergunta:** o que muda com mais épocas, no treino, na validação e no "
        "texto gerado? Até quando vale a pena treinar?"
    )
    report.write()

    cfg = replace(BASELINE, num_epochs=T2_EPOCHS)
    longo, sondas = train_with_probes(cfg, text, bpe)
    historico = longo.history

    n_ref = len(referencia.history.val_losses)
    diferenca = max(
        abs(a - b)
        for a, b in zip(
            historico.train_losses[:n_ref] + historico.val_losses[:n_ref],
            referencia.history.train_losses + referencia.history.val_losses,
        )
    )
    report.write(
        f"**Configuração:** a do T1 com {T2_EPOCHS} épocas. Sem agendamento da taxa "
        f"de aprendizado, as {referencia.cfg.num_epochs} primeiras épocas deste treino "
        f"SÃO o T1: a maior diferença entre as {n_ref} avaliações em comum é "
        f"{diferenca:.1e}. Treinar k épocas é parar este treino na época k."
    )
    report.write()
    report.write(
        f"Sonda de memorização: a partir de {PROBE_PROMPT} tokens reais, quantos dos "
        f"{PROBE_CONTINUATION} seguintes a geração gulosa reproduz antes do primeiro erro "
        f"(média de {PROBE_WINDOWS} trechos por grupo). \"Treino alinhado\": trechos que "
        "começam onde começa uma janela do treino (tokens 0, 512, 1 024, ...), nas mesmas "
        f"posições em que o modelo os viu. \"Treino deslocado\": os mesmos trechos {PROBE_SHIFT} "
        "tokens adiante -- o mesmo texto, em posições em que o modelo nunca o viu. "
        "\"Validação\": trechos espalhados pelo texto de validação."
    )
    report.write()

    ultimo = last_eval_of_epoch(historico)
    ids_treino = longo.run.train_loader.dataset.token_ids.tolist()
    sem_sonda = dict.fromkeys(PROBE_GROUPS, float("nan"))
    linhas = []
    for epoca in T2_TABLE_EPOCHS:
        i = ultimo[epoca]
        amostra = continuation(historico.samples[epoca - 1], bpe, cfg.start_context)
        sonda = sondas.get(epoca, sem_sonda)
        linhas.append(
            [
                str(epoca),
                f"{historico.train_losses[i]:.3f}",
                f"{historico.val_losses[i]:.3f}",
                f"{historico.val_losses[i] - historico.train_losses[i]:.3f}",
                ppl(historico.train_losses[i]),
                ppl(historico.val_losses[i]),
                *[f"{sonda[grupo]:.1f}" for grupo in PROBE_GROUPS],
                str(longest_shared_run(amostra, ids_treino)),
            ]
        )
    report.table(
        ["época", "treino", "validação", "diferença", "perplexidade treino", "perplexidade validação",
         *[f"recita: {grupo}" for grupo in PROBE_GROUPS], "maior cópia na amostra"],
        linhas,
    )
    inicial = sondas.get(0, sem_sonda)
    report.write(
        "Antes do treino, a sonda reproduz "
        + ", ".join(f"{inicial[grupo]:.1f} ({grupo})" for grupo in PROBE_GROUPS)
        + ". \"Maior cópia na amostra\": o maior trecho contíguo da continuação gerada "
        "no fim da época (50 tokens) que aparece, token a token, no texto de treino."
    )
    report.write()

    melhor = historico.best_evaluation()
    report.write(
        f"Menor perda de validação: {historico.val_losses[melhor]:.3f} (perplexidade "
        f"{ppl(historico.val_losses[melhor])}) no passo {historico.steps[melhor]}, "
        f"época {historico.epochs[melhor]}. No fim das {T2_EPOCHS} épocas: treino "
        f"{historico.train_losses[-1]:.3f}, validação {historico.val_losses[-1]:.3f}."
    )
    report.write()

    report.write("**Amostras ao longo do treino:**")
    report.write()
    report.table(
        ["época", "amostra"],
        [[str(e), compact(historico.samples[e - 1])] for e in T2_TABLE_EPOCHS],
    )

    figura, eixo = new_axes(7.0, 3.8)
    loss_curves(eixo, historico, longo.steps_per_epoch, SERIES_COLORS[0])
    x_melhor = (historico.steps[melhor] + 1) / longo.steps_per_epoch
    eixo.axvline(x_melhor, color=SERIES_COLORS[2], linestyle=":", linewidth=1.0,
                 label=f"validação mínima (época {historico.epochs[melhor]})")
    eixo.set_ylim(bottom=0)
    eixo.set_title(f"{T2_EPOCHS} épocas: treino e validação", fontsize=10)
    legend(eixo, loc="upper right")
    report.figure(save_figure(figura, "treino-epocas.png"), f"Perdas ao longo de {T2_EPOCHS} épocas")

    figura, eixo = new_axes(7.0, 3.4)
    epocas = sorted(sondas)
    for grupo, cor, estilo in zip(PROBE_GROUPS, SERIES_COLORS, ("-", "--", ":")):
        eixo.plot(epocas, [sondas[e][grupo] for e in epocas], color=cor, marker="o",
                  markersize=3, linewidth=1.6, linestyle=estilo, label=grupo)
    eixo.set_ylim(0, PROBE_CONTINUATION)
    eixo.set_xlabel("épocas")
    eixo.set_ylabel(f"tokens reproduzidos (de {PROBE_CONTINUATION})")
    eixo.set_title("Memorização: quanto a geração gulosa recita do texto", fontsize=10)
    legend(eixo, loc="center right")
    report.figure(save_figure(figura, "treino-memorizacao.png"),
                  "Tokens recitados: treino alinhado, treino deslocado e validação")
    return longo, sondas


def variant_rows(runs: list[Run], bpe: BPETokenizer, train_ids: list[int], extra=None) -> list[list[str]]:
    """Linha de resumo de cada variante: o que os T3 a T6 comparam."""
    linhas = []
    for item in runs:
        h = item.history
        melhor = h.best_evaluation()
        fim_epoca_1 = last_eval_of_epoch(h)[1]
        amostra = continuation(h.samples[-1], bpe, item.cfg.start_context)
        linha = [
            item.label,
            f"{h.val_losses[fim_epoca_1]:.3f}",
            f"{h.val_losses[melhor]:.3f}",
            str(h.epochs[melhor]),
            f"{h.train_losses[-1]:.3f}",
            f"{h.val_losses[-1]:.3f}",
            f"{h.val_losses[-1] - h.train_losses[-1]:.3f}",
            str(longest_shared_run(amostra, train_ids)),
            f"{item.seconds:.0f}",
        ]
        if extra is not None:
            linha += extra(item)
        linhas.append(linha)
    return linhas


VARIANT_HEADERS = [
    "variante", "validação fim da época 1", "validação mínima", "época da mínima",
    "treino final", "validação final", "diferença final", "maior cópia na amostra final",
    "tempo (s)",
]


def curves_figure(runs: list[Run], name: str, title: str, caption: str, report: Report, val_only: bool = False) -> None:
    figura, eixo = new_axes(7.0, 3.8)
    for item, cor in zip(runs, SERIES_COLORS):
        loss_curves(eixo, item.history, item.steps_per_epoch, cor, label=item.label, train=not val_only)
    eixo.set_ylim(bottom=0)
    eixo.set_title(title, fontsize=10)
    legend(eixo, loc="upper right", ncols=1 if val_only else 2)
    report.figure(save_figure(figura, name), caption)


def experiment_3_learning_rate(report: Report, text: str, bpe: BPETokenizer, referencia: Run) -> None:
    report.write("## T3 - Taxa de aprendizado")
    report.write()
    report.write(
        "**Pergunta:** como a taxa de aprendizado muda a velocidade de "
        "convergência, a estabilidade e o ponto em que o sobreajuste começa?"
    )
    report.write()
    report.write(
        f"**Configuração:** a do T1, com lr ∈ {{{', '.join(f'{lr:g}' for lr in T3_LEARNING_RATES)}}}. "
        "lr 4e-4 é o próprio T1 (o do livro)."
    )
    report.write()

    runs = []
    for lr in T3_LEARNING_RATES:
        if lr == BASELINE.learning_rate:
            runs.append(Run(f"lr {lr:g}", referencia.cfg, referencia.run, referencia.seconds))
        else:
            runs.append(train(f"lr {lr:g}", replace(BASELINE, learning_rate=lr), text, bpe))

    ids_treino = referencia.run.train_loader.dataset.token_ids.tolist()
    report.table(
        VARIANT_HEADERS + ["maior subida do treino"],
        variant_rows(runs, bpe, ids_treino, extra=lambda r: [f"{max_rise(r.history.train_losses):.3f}"]),
    )
    report.write(
        "\"Maior subida do treino\": o maior aumento da perda de treino entre duas "
        "avaliações seguidas -- 0 numa descida monótona, positivo quando o passo "
        "do otimizador piora o modelo."
    )
    report.write()
    report.table(
        ["variante", "amostra final"],
        [[r.label, compact(r.history.samples[-1])] for r in runs],
    )
    curves_figure(runs, "treino-lr-validacao.png", "Perda de validação por taxa de aprendizado",
                  "Validação por taxa de aprendizado", report, val_only=True)
    figura, eixo = new_axes(7.0, 3.8)
    for item, cor in zip(runs, SERIES_COLORS):
        loss_curves(eixo, item.history, item.steps_per_epoch, cor, label=item.label, val=False)
    eixo.set_ylim(bottom=0)
    eixo.set_title("Perda de treino por taxa de aprendizado", fontsize=10)
    legend(eixo, loc="upper right")
    report.figure(save_figure(figura, "treino-lr-treino.png"), "Treino por taxa de aprendizado")


def experiment_4_batch_size(report: Report, text: str, bpe: BPETokenizer, referencia: Run) -> None:
    report.write("## T4 - Tamanho do lote")
    report.write()
    report.write(
        "**Pergunta:** com a mesma taxa de aprendizado e o mesmo número de "
        "épocas, o que muda com lotes maiores: a convergência por época, o "
        "número de atualizações e o custo?"
    )
    report.write()
    report.write(
        f"**Configuração:** a do T1, com lotes de {', '.join(map(str, T4_BATCH_SIZES))} "
        "sequências; lr 4e-4 em todos. Lote 2 é o T1. Com `drop_last=True`, as "
        "janelas que não completam um lote ficam fora da época."
    )
    report.write()

    runs = []
    for tamanho in T4_BATCH_SIZES:
        if tamanho == BASELINE.batch_size:
            runs.append(Run(f"lote {tamanho}", referencia.cfg, referencia.run, referencia.seconds))
        else:
            runs.append(train(f"lote {tamanho}", replace(BASELINE, batch_size=tamanho), text, bpe))

    ids_treino = referencia.run.train_loader.dataset.token_ids.tolist()

    def custo(item: Run) -> list[str]:
        h = item.history
        segundos = training_seconds(h)
        return [
            str(item.steps_per_epoch),
            str(h.global_step + 1),
            f"{segundos / h.epochs_completed:.2f}",
            thousands(round(h.total_tokens / segundos)),
        ]

    report.table(
        VARIANT_HEADERS + ["passos por época", "passos no total", "s por época (só treino)", "tokens/s"],
        variant_rows(runs, bpe, ids_treino, extra=custo),
    )
    curves_figure(runs, "treino-lote.png", "Treino e validação por tamanho do lote",
                  "Perdas por tamanho do lote", report)


def experiment_5_regularization(report: Report, text: str, bpe: BPETokenizer, referencia: Run) -> None:
    report.write("## T5 - Regularização: dropout e weight decay")
    report.write()
    report.write(
        "**Pergunta:** os dois regularizadores do modelo reduzem o sobreajuste "
        "neste corpus? A que custo na perda de treino?"
    )
    report.write()
    report.write(
        f"**Configuração:** a do T1, variando um de cada vez: dropout (`drop_rate` do "
        f"modelo, nas três posições) ∈ {{{', '.join(map(str, T5_DROPOUTS))}}} e weight decay "
        f"do AdamW ∈ {{{', '.join(map(str, T5_WEIGHT_DECAYS))}}}. Dropout 0,1 com weight "
        "decay 0,1 é o T1."
    )
    report.write()

    ids_treino = referencia.run.train_loader.dataset.token_ids.tolist()
    for nome, valores, monta in (
        ("dropout", T5_DROPOUTS, lambda v: (replace(MODEL_CONFIG, drop_rate=v), BASELINE)),
        ("weight decay", T5_WEIGHT_DECAYS, lambda v: (MODEL_CONFIG, replace(BASELINE, weight_decay=v))),
    ):
        runs = []
        for valor in valores:
            model_cfg, cfg = monta(valor)
            rotulo = f"{nome} {valor}"
            if model_cfg == MODEL_CONFIG and cfg == BASELINE:
                runs.append(Run(rotulo, cfg, referencia.run, referencia.seconds))
            else:
                runs.append(train(rotulo, cfg, text, bpe, model_cfg=model_cfg))
        report.write(f"**{nome.capitalize()}:**")
        report.write()
        report.table(VARIANT_HEADERS, variant_rows(runs, bpe, ids_treino))
        arquivo = "treino-dropout.png" if nome == "dropout" else "treino-weight-decay.png"
        curves_figure(runs, arquivo, f"Treino e validação por {nome}", f"Perdas por {nome}", report)


def experiment_6_overfitting(report: Report, text: str, bpe: BPETokenizer, referencia: Run) -> None:
    report.write("## T6 - Sobreajuste: treino x validação")
    report.write()
    report.write(
        "**Pergunta:** a diferença entre as perdas de treino e validação é "
        "sobreajuste, ou parte dela vem de a validação ser o desfecho do conto, "
        "um trecho diferente do resto?"
    )
    report.write()

    # O mesmo conto girado: os 10% iniciais passam para o fim e viram a validação.
    giro = len(split_text(text, BASELINE.train_ratio)[1])
    girado = text[giro:] + text[:giro]
    report.write(
        "**Configuração:** a do T1 com a validação tirada do COMEÇO do conto. O "
        f"texto é girado em {thousands(giro)} caracteres (os 10% iniciais passam "
        "para o fim) e cortado como sempre: a validação vira o início da "
        "história, e o treino, o resto inteiro."
    )
    report.write()

    alternativo = train("validação no início", BASELINE, girado, bpe)
    runs = [Run("validação no fim (T1)", referencia.cfg, referencia.run, referencia.seconds), alternativo]
    ids_treino = referencia.run.train_loader.dataset.token_ids.tolist()

    report.table(
        VARIANT_HEADERS[:1] + ["validação no passo 0"] + VARIANT_HEADERS[1:-1],
        [
            [linha[0], f"{item.history.val_losses[0]:.3f}", *linha[1:-1]]
            for item, linha in zip(runs, variant_rows(runs, bpe, ids_treino))
        ],
    )
    curves_figure(runs, "treino-divisao.png", "Validação no fim e no início do conto",
                  "Perdas com a validação no fim e no início do conto", report)


def forward_flops_per_token(cfg: GPTConfig, tokens: int) -> float:
    """FLOPs do forward por token: as multiplicações de matriz, como no A7 da Sprint 4.

    projeções dos blocos  L · 2 · (4D² + 2·D·H)
    atenção (QKᵀ e P·V)   L · 4 · T · D
    cabeça de saída       2 · D · V
    """
    d, layers, h = cfg.emb_dim, cfg.n_layers, cfg.feed_forward_dim
    return layers * 2 * (4 * d * d + 2 * d * h) + layers * 4 * tokens * d + 2 * d * cfg.vocab_size


def measure_step(model_cfg: GPTConfig, batch_size: int, device: torch.device) -> tuple[float, float | None]:
    """Mediana do tempo de um passo de treino (forward, backward, AdamW), em
    segundos, e o pico de memória da GPU em MB (None na CPU)."""
    torch.manual_seed(comum.SEED)
    model = GPTModel(model_cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASELINE.learning_rate, weight_decay=BASELINE.weight_decay)
    entradas = torch.randint(0, model_cfg.vocab_size, (batch_size, model_cfg.context_length), device=device)
    alvos = torch.randint(0, model_cfg.vocab_size, (batch_size, model_cfg.context_length), device=device)

    def sincroniza() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tempos = []
    for passo in range(T8_WARMUP + T8_STEPS):
        sincroniza()
        inicio = time.perf_counter()
        optimizer.zero_grad()
        batch_loss(model, entradas, alvos).backward()
        optimizer.step()
        sincroniza()
        if passo >= T8_WARMUP:
            tempos.append(time.perf_counter() - inicio)
    pico = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return statistics.median(tempos), pico


def fixed_windows(token_ids: torch.Tensor, offset: int, batch: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Janelas sem sobreposição de context_length tokens, começando em `offset`,
    em lotes de `batch`: a grade do treino, ou ela deslocada."""
    length = MODEL_CONFIG.context_length
    ids = token_ids[offset:]
    n = (len(ids) - length - 1) // length + 1
    entradas = torch.stack([ids[i * length : (i + 1) * length] for i in range(n)])
    alvos = torch.stack([ids[i * length + 1 : (i + 1) * length + 1] for i in range(n)])
    return [(entradas[i : i + batch], alvos[i : i + batch]) for i in range(0, n, batch)]


def experiment_7_stride(report: Report, text: str, bpe: BPETokenizer, referencia: Run) -> None:
    report.write("## T7 - Janelas sobrepostas: o stride do treino")
    report.write()
    report.write(
        "**Pergunta:** com janelas sem sobreposição (stride = 128), cada token "
        "do conto aparece sempre na mesma posição da janela, época após época. "
        "O modelo aprende o texto ou aprende o texto naquelas posições? "
        "Janelas sobrepostas mudam isso, e a que custo?"
    )
    report.write()
    report.write(
        f"**Configuração:** a do T1 com stride de treino {', '.join(map(str, T7_STRIDES))} "
        "(128 é o T1); a validação não muda. Com stride menor, a época tem mais "
        "janelas e mais passos, e a avaliação roda a cada 5 x (128 / stride) "
        "passos, o mesmo número de vezes por época. Depois do treino, a perda do "
        f"texto de treino é medida em duas grades de janelas sem sobreposição: a "
        f"do treino (começando no token 0) e a mesma deslocada de {T7_SHIFT} tokens -- "
        "o mesmo texto, com cada token numa posição em que o modelo nunca o viu "
        f"({T7_SHIFT} não é múltiplo de nenhum dos três strides)."
    )
    report.write()

    runs = []
    for stride in T7_STRIDES:
        if stride == MODEL_CONFIG.context_length:
            runs.append(Run(f"stride {stride} (T1)", referencia.cfg, referencia.run, referencia.seconds))
        else:
            cfg = replace(
                BASELINE,
                stride=stride,
                eval_freq=BASELINE.eval_freq * MODEL_CONFIG.context_length // stride,
            )
            runs.append(train(f"stride {stride}", cfg, text, bpe))

    ids_treino = referencia.run.train_loader.dataset.token_ids
    grade = fixed_windows(ids_treino, 0, T7_EVAL_BATCH)
    deslocada = fixed_windows(ids_treino, T7_SHIFT, T7_EVAL_BATCH)
    # A última AVALIAÇÃO do T1, e não o total de tokens: o passo final (179)
    # não é múltiplo de eval_freq, e a última avaliação é a do passo 175.
    tokens_t1 = referencia.history.tokens_seen[-1]

    def extra(item: Run) -> list[str]:
        h = item.history
        na_grade = loader_loss(item.run.model, grade)
        fora = loader_loss(item.run.model, deslocada)
        mesmo_custo = next(i for i, t in enumerate(h.tokens_seen) if t >= tokens_t1)
        return [
            str(len(item.run.train_loader.dataset)),
            str(h.global_step + 1),
            f"{h.val_losses[mesmo_custo]:.3f}",
            f"{na_grade:.3f}",
            f"{fora:.3f}",
            f"{fora - na_grade:.3f}",
        ]

    report.table(
        VARIANT_HEADERS + [
            "janelas por época", "passos no total", f"validação com {thousands(tokens_t1)} tokens vistos",
            "treino, grade do treino", f"treino, grade deslocada {T7_SHIFT}", "diferença entre grades",
        ],
        variant_rows(runs, bpe, ids_treino.tolist(), extra=extra),
    )
    report.write(
        f"\"Validação com {thousands(tokens_t1)} tokens vistos\": a primeira avaliação depois de "
        "o treino ter processado tantos tokens quanto o T1 na sua última avaliação -- a "
        "comparação com o mesmo custo de treino."
    )
    report.write()
    curves_figure(runs, "treino-stride.png", "Treino e validação por stride das janelas de treino",
                  "Perdas por stride do treino", report)


def experiment_8_cost(report: Report, text: str, bpe: BPETokenizer, referencia: Run) -> None:
    report.write("## T8 - Custo computacional")
    report.write()
    report.write(
        "**Pergunta:** quanto custa treinar, em tempo, memória e operações, e "
        "como isso escala do modelo do projeto para o do livro?"
    )
    report.write()

    dispositivos = [torch.device("cpu")]
    if torch.cuda.is_available():
        dispositivos.append(torch.device("cuda"))
    nome_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    report.write(
        f"**Configuração:** um passo de treino (forward, backward, AdamW) com lotes "
        f"de 2 sequências cheias; mediana de {T8_STEPS} passos depois de {T8_WARMUP} de "
        f"aquecimento. CPU: {torch.get_num_threads()} threads do PyTorch"
        + (f"; GPU: {nome_gpu}." if nome_gpu else "; sem GPU nesta máquina.")
        + " Memória do estado de treino: pesos, gradientes e os dois momentos do "
        "AdamW, 16 bytes por parâmetro em float32 (sem as ativações)."
    )
    report.write()

    modelos = [("BASE_CONFIG (projeto)", MODEL_CONFIG), ("BOOK_MODEL_CONFIG (livro, contexto 256)", BOOK_MODEL_CONFIG)]
    linhas = []
    for nome, cfg in modelos:
        with torch.device("meta"):
            parametros = GPTModel(cfg).parameter_count()
        flops_passo = 3 * forward_flops_per_token(cfg, cfg.context_length) * 2 * cfg.context_length
        for dispositivo in dispositivos:
            segundos, pico = measure_step(cfg, 2, dispositivo)
            linhas.append(
                [
                    nome,
                    dispositivo.type.upper(),
                    thousands(parametros),
                    f"{parametros * BYTES_PER_PARAMETER_TRAINING / 2**20:,.0f}".replace(",", " "),
                    f"{pico:,.0f}".replace(",", " ") if pico is not None else "-",
                    f"{flops_passo / 1e9:.1f}",
                    f"{segundos * 1000:.0f}",
                    thousands(round(2 * cfg.context_length / segundos)),
                    f"{flops_passo / segundos / 1e9:.0f}",
                ]
            )
            print(f"  {nome} em {dispositivo.type}: {segundos * 1000:.0f} ms por passo")
    report.table(
        ["modelo", "device", "parâmetros", "estado de treino (MB)", "pico na GPU (MB)",
         "GFLOP por passo", "ms por passo", "tokens/s", "GFLOP/s"],
        linhas,
    )
    report.write(
        "GFLOP por passo: 3 x os FLOPs do forward (o backward custa ~2 forwards), "
        "contando só as multiplicações de matriz."
    )
    report.write()

    # O custo de observar: avaliar sobre 5 lotes (livro) ou sobre tudo (projeto).
    modelo = referencia.run.model
    treino_loader, val_loader = referencia.run.train_loader, referencia.run.val_loader
    linhas = []
    for rotulo, lotes in (("5 lotes (livro)", 5), ("conjuntos inteiros (projeto)", None)):
        tempos = []
        with torch.random.fork_rng(devices=[]):
            for _ in range(3):
                inicio = time.perf_counter()
                evaluate(modelo, treino_loader, val_loader, lotes)
                tempos.append(time.perf_counter() - inicio)
        linhas.append([rotulo, f"{statistics.median(tempos) * 1000:.0f}"])
    h = referencia.history
    passos_s = training_seconds(h) / (h.global_step + 1)
    report.table(["avaliação (treino + validação)", "ms por avaliação"], linhas)
    report.write(
        f"No T1, um passo de treino custa {passos_s * 1000:.0f} ms em média, e a avaliação "
        f"roda a cada {BASELINE.eval_freq} passos: avaliações e amostras ficaram com "
        f"{sum(h.monitor_seconds) / sum(h.epoch_seconds):.0%} do tempo das épocas."
    )
    report.write()


def main() -> None:
    utf8_console()
    if RAPIDO:
        destino = Path(tempfile.mkdtemp(prefix="sprint05-rapido-"))
        comum.FIGURES_DIR = destino / "figuras"
        report = Report(destino / "treino.md")
        print(f"modo rápido: resultados em {destino}")
    else:
        report = Report(comum.RESULTS_DIR / "treino.md")
    bpe = BPETokenizer()
    texto = load_verdict()

    report.write("# Sprint 5 - Treinamento: resultados dos experimentos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint05/treino_experimentos.py`. Não editar à mão: rode o script novamente."
    )
    report.write()
    report.write(
        f"Modelo: BASE_CONFIG. Treino de referência: BASE_TRAINING_CONFIG (AdamW, lr "
        f"{BASELINE.learning_rate:g}, weight decay {BASELINE.weight_decay}, lotes de "
        f"{BASELINE.batch_size}, {BASELINE.num_epochs} épocas, avaliação a cada "
        f"{BASELINE.eval_freq} passos sobre os conjuntos inteiros, seed {BASELINE.seed}). Cada "
        "experimento muda um hiperparâmetro e mantém os outros. Dados: The Verdict, 90% "
        "treino / 10% validação, janelas de 128 tokens sem sobreposição. Perdas em nats, "
        "médias por token; \"validação mínima\" é a menor perda entre todas as "
        "avaliações; \"final\", a da última avaliação (passo 175 de 180 no treino de "
        "10 épocas). Treinos em CPU, seeds fixas: rodar de novo reproduz tudo, "
        "exceto as colunas de tempo. Figuras em `resultados/figuras/`."
    )
    report.write()

    # O relatório é salvo depois de cada experimento: uma falha no meio de ~20
    # minutos de treino não leva junto o que já foi medido.
    print("T1")
    referencia = experiment_1_reference(report, texto, bpe)
    report.save()
    for nome, experimento in (
        ("T2", experiment_2_epochs),
        ("T3", experiment_3_learning_rate),
        ("T4", experiment_4_batch_size),
        ("T5", experiment_5_regularization),
        ("T6", experiment_6_overfitting),
        ("T7", experiment_7_stride),
        ("T8", experiment_8_cost),
    ):
        print(nome)
        experimento(report, texto, bpe, referencia)
        report.save()


if __name__ == "__main__":
    main()
