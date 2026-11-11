"""Experimentos de estratégias de geração da Sprint 5 (Entregável 5).

D1  A distribuição que cada estratégia recebe, antes e depois do treino
D2  Estratégias lado a lado no modelo treinado
D3  Temperatura
D4  Top-k
D5  Estratégia x treino: o que muda com a estratégia e o que muda com os pesos

A Sprint 4 mediu a decodificação no modelo SEM treino (G1 a G7): ali os logits
são quase planos e todas as estratégias produzem texto incoerente. Aqui as
mesmas funções de src/generation/ rodam sobre o modelo treinado, e o desenho é
uma grade de duas entradas: PESOS x ESTRATÉGIA.

Os pesos são três estados do mesmo treino (BASE_CONFIG, BASE_TRAINING_CONFIG,
seed 42):

- sem treino: os pesos sorteados pela seed, o modelo dos G1 a G7 da Sprint 4;
- 10 épocas: o treino de referência (T1), com a validação perto do mínimo;
- 30 épocas: o fim do T2, com o texto de treino decorado.

Um treino só produz os três: 10 épocas, cópia dos pesos, mais 20 épocas com o
mesmo otimizador e o mesmo histórico. O script confere as perdas com as
publicadas em resultados/treino.md.

Toda estratégia parte do MESMO contexto ("Every effort moves you") e gera 50
tokens. As que sorteiam geram 20 continuações num lote só, com um
torch.Generator semeado (seed 42); a gulosa gera uma, porque não sorteia.
Rodar de novo reproduz todos os números. Leva ~15 minutos em CPU.

Uso:
    python experimentos/sprint05/geracao_experimentos.py
    python experimentos/sprint05/geracao_experimentos.py --rapido   # versão curta,
        só para conferir que o script roda: relatório e figuras vão para uma
        pasta temporária
"""

import copy
import math
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import comum  # noqa: E402
from comum import (  # noqa: E402
    DEVICE,
    INK_SECONDARY,
    SEED,
    SERIES_COLORS,
    Report,
    compact,
    legend,
    longest_shared_run,
    new_axes,
    save_figure,
    thousands,
    utf8_console,
)
from src.corpus import load_verdict  # noqa: E402
from src.generation import generate, next_token_probabilities, text_to_token_ids  # noqa: E402
from src.model import BASE_CONFIG, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.training import (  # noqa: E402
    BASE_TRAINING_CONFIG,
    TrainingHistory,
    loader_loss,
    perplexity,
    train_gpt,
    train_model,
)

RAPIDO = "--rapido" in sys.argv

MODEL_CONFIG = BASE_CONFIG
BASELINE = BASE_TRAINING_CONFIG
VOCAB = MODEL_CONFIG.vocab_size

# Épocas dos dois estados treinados: o T1 e o fim do T2.
EPOCHS = (10, 30) if not RAPIDO else (2, 3)
# Última avaliação das épocas 10 e 30 em resultados/treino.md (treino, validação).
PUBLISHED_LOSSES = {10: (3.429, 6.140), 30: (0.043, 7.067)}
LOSS_PRINT_TOLERANCE = 5e-4  # meia unidade na 3ª casa

PROMPT = BASELINE.start_context
NEW_TOKENS = 50 if not RAPIDO else 12
SAMPLES = 20 if not RAPIDO else 4
# Lote da conta de log-probabilidade: (10, 81, V) em float32 são 160 MB.
LOG_PROB_BATCH = 10

# As cinco estratégias do G5 da Sprint 4, medidas lá no modelo sem treino, mais
# a que o livro usa para gerar texto na seção 5.3.3. Da mais para a menos
# determinística.
STRATEGIES = {
    "gulosa (T = 0)": {"temperature": 0.0},
    "top-k 5, T = 1": {"temperature": 1.0, "top_k": 5},
    "top-k 50, T = 0,7": {"temperature": 0.7, "top_k": 50},
    "top-k 25, T = 1,4 (livro)": {"temperature": 1.4, "top_k": 25},
    "T = 1, sem top-k": {"temperature": 1.0},
    "T = 1,5, sem top-k": {"temperature": 1.5},
}
STRATEGY_TICKS = ("gulosa", "k=5\nT=1", "k=50\nT=0,7", "k=25\nT=1,4", "T=1", "T=1,5")
GREEDY = "gulosa (T = 0)"
PURE_SAMPLING = "T = 1, sem top-k"

D1_TOP = 5
D1_MASS = 0.9
D1_TEMPERATURES = (0.5, 0.7, 1.0, 1.5, 2.0)
D1_TOP_K = (1, 5, 25, 50, 200, 1_000)
D3_TEMPERATURES = (0.0, 0.25, 0.5, 0.7, 1.0, 1.5, 2.0)
D4_TOP_K = (1, 5, 25, 50, 200, None)
D4_GRID_TEMPERATURES = (0.7, 1.0, 1.5)
D4_GRID_TOP_K = (5, 50, None)
D5_TEXT_STRATEGIES = (GREEDY, "top-k 50, T = 0,7", PURE_SAMPLING, "T = 1,5, sem top-k")
# Segundo contexto do D5: os primeiros tokens do texto de validação, que o
# modelo nunca viu no treino.
D5_SECOND_CONTEXT_TOKENS = 32


@dataclass
class State:
    """Um estado dos pesos: o modelo e as perdas dele sobre os dois conjuntos."""

    name: str
    epochs: int
    model: GPTModel
    train_loss: float
    val_loss: float


@dataclass
class Context:
    """O texto de partida de uma geração."""

    name: str
    ids: torch.Tensor  # (1, T)


@dataclass
class References:
    """O conto em Token IDs, contra o qual as continuações são medidas."""

    train_ids: list[int]
    train_vocab: set[int]
    bigrams: set[tuple[int, int]]


@dataclass
class Sample:
    """As continuações de uma estratégia num estado dos pesos, e as medidas de cada uma."""

    tokens: list[list[int]]
    values: dict[str, list[float]]

    def mean(self, measure: str) -> float:
        return statistics.fmean(self.values[measure])

    def deviation(self, measure: str) -> float:
        valores = self.values[measure]
        return statistics.stdev(valores) if len(valores) > 1 else 0.0

    @property
    def distinct_continuations(self) -> int:
        return len({tuple(continuacao) for continuacao in self.tokens})

    @property
    def distinct_first_tokens(self) -> int:
        return len({continuacao[0] for continuacao in self.tokens})


def share(value: float) -> str:
    return f"{100 * value:.1f}%"


def count(value: float) -> str:
    return f"{value:.1f}"


# Medida -> (título da coluna, formatação). A ordem é a das tabelas:
# diversidade, repetição, coerência.
MEASURES = {
    "distinct_1": ("tokens distintos", share),
    "distinct_2": ("bigramas distintos", share),
    "run": ("maior repetição", count),
    "repeat_3": ("trigramas repetidos", share),
    "log_prob": ("log-prob média", lambda value: f"{value:.3f}"),
    "unseen": ("tokens nunca vistos no treino", share),
    "corpus_2": ("bigramas no conto", share),
    "copy": ("maior cópia do treino", count),
}


def ngrams(tokens: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_share(tokens: list[int], n: int) -> float:
    """n-gramas diferentes sobre n-gramas no total: 1 se nenhum se repete."""
    grams = ngrams(tokens, n)
    return len(set(grams)) / len(grams)


def repeated_share(tokens: list[int], n: int) -> float:
    """Fração dos n-gramas que já tinham aparecido antes na mesma continuação."""
    return 1.0 - distinct_share(tokens, n)


def longest_run(tokens: list[int]) -> int:
    """Maior sequência do mesmo token repetido em seguida (a medida do G5 da Sprint 4)."""
    maior = atual = 1
    for anterior, seguinte in zip(tokens, tokens[1:]):
        atual = atual + 1 if seguinte == anterior else 1
        maior = max(maior, atual)
    return maior


def corpus_bigram_share(tokens: list[int], previous: int, bigrams: set[tuple[int, int]]) -> float:
    """Fração dos bigramas da continuação que existem no conto.

    Conta também o bigrama que liga o último token do contexto ao primeiro
    gerado: são len(tokens) bigramas.
    """
    pares = ngrams([previous] + tokens, 2)
    return sum(par in bigrams for par in pares) / len(pares)


@torch.no_grad()
def own_log_probabilities(model: GPTModel, context: torch.Tensor, continuations: torch.Tensor) -> list[float]:
    """Média de ln p(token) de cada continuação, segundo o próprio modelo.

    Args:
        model: o GPTModel, em eval().
        context: (1, T) o contexto de partida.
        continuations: (n, N) os tokens gerados.

    Returns:
        n médias. A probabilidade é a do modelo sem ajuste (T = 1, sem top-k),
        numa passada sobre o texto final: mede o quanto o texto é provável
        para os pesos, qualquer que tenha sido a estratégia que o escolheu.
    """
    inicio = context.shape[1]
    medias: list[float] = []
    for lote in continuations.split(LOG_PROB_BATCH):
        texto = torch.cat([context.repeat(len(lote), 1), lote], dim=1)
        # A posição t prevê o token t + 1: as N últimas linhas, sem a final.
        logits = model(texto[:, :-1])[:, inicio - 1 :]
        log_probs = torch.log_softmax(logits, dim=-1)
        medias += log_probs.gather(-1, lote.unsqueeze(-1)).squeeze(-1).mean(dim=1).tolist()
    return medias


class Sampler:
    """Gera e mede as continuações, uma vez por (estado, contexto, estratégia).

    A mesma combinação aparece em mais de um experimento (T = 1 sem top-k está
    no D2, no D3, no D4 e no D5); o resultado fica guardado.
    """

    def __init__(self, references: References) -> None:
        self.references = references
        self._cache: dict[tuple, Sample] = {}

    def __call__(self, state: State, context: Context, options: dict) -> Sample:
        temperature = options.get("temperature", 0.0)
        chave = (state.name, context.name, temperature, options.get("top_k"))
        if chave not in self._cache:
            self._cache[chave] = self._sample(state, context, options, greedy=temperature == 0.0)
        return self._cache[chave]

    def _sample(self, state: State, context: Context, options: dict, greedy: bool) -> Sample:
        # A gulosa não sorteia: 20 execuções dariam 20 textos iguais.
        quantidade = 1 if greedy else SAMPLES
        saida = generate(
            state.model,
            context.ids.repeat(quantidade, 1),
            NEW_TOKENS,
            generator=torch.Generator().manual_seed(SEED),
            **options,
        )
        novos = saida[:, context.ids.shape[1] :]
        continuacoes = novos.tolist()
        anterior = int(context.ids[0, -1])
        refs = self.references

        valores = {
            "distinct_1": [distinct_share(c, 1) for c in continuacoes],
            "distinct_2": [distinct_share(c, 2) for c in continuacoes],
            "run": [float(longest_run(c)) for c in continuacoes],
            "repeat_3": [repeated_share(c, 3) for c in continuacoes],
            "log_prob": own_log_probabilities(state.model, context.ids, novos),
            "unseen": [sum(t not in refs.train_vocab for t in c) / len(c) for c in continuacoes],
            "corpus_2": [corpus_bigram_share(c, anterior, refs.bigrams) for c in continuacoes],
            "copy": [float(longest_shared_run(c, refs.train_ids)) for c in continuacoes],
        }
        return Sample(continuacoes, valores)


def measure_headers(first: str) -> list[str]:
    return [first] + [titulo for titulo, _ in MEASURES.values()] + ["continuações diferentes"]


def measure_row(label: str, sample: Sample) -> list[str]:
    return (
        [label]
        + [formato(sample.mean(medida)) for medida, (_, formato) in MEASURES.items()]
        + [f"{sample.distinct_continuations} de {len(sample.tokens)}"]
    )


def entropy(probabilities: torch.Tensor) -> float:
    """Entropia em nats; 0·log 0 tratado como 0."""
    seguras = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return max(0.0, float(-(probabilities * seguras.log()).sum(dim=-1)))


def token_label(bpe: BPETokenizer, token_id: int) -> str:
    """Texto do token entre crases, com o espaço inicial visível."""
    return f"`{bpe.decode([token_id])!r}`"


def text_of(bpe: BPETokenizer, context: Context, continuation: list[int]) -> str:
    return f"`{compact(bpe.decode(context.ids[0].tolist() + continuation))}`"


def last_eval_of_epoch(history: TrainingHistory, epoch: int) -> int:
    return max(i for i, e in enumerate(history.epochs) if e == epoch)


def train_states(text: str, bpe: BPETokenizer):
    """Os três estados dos pesos, de um treino só.

    Returns:
        (estados, histórico, train_loader, val_loader, segundos de treino).

    Os dois modelos extras só são criados depois do treino inteiro: criar um
    GPTModel sorteia pesos do gerador global, e um sorteio entre a época 10 e a
    11 mudaria a ordem dos lotes dali em diante -- o treino deixaria de ser o T2.
    """
    primeira, ultima = EPOCHS
    inicio = time.perf_counter()
    run = train_gpt(MODEL_CONFIG, replace(BASELINE, num_epochs=primeira), text, bpe, device=DEVICE, log=None)
    pesos_intermediarios = copy.deepcopy(run.model.state_dict())
    print(f"  {primeira} épocas: {time.perf_counter() - inicio:.0f} s")

    train_model(
        run.model,
        run.train_loader,
        run.val_loader,
        run.optimizer,
        num_epochs=ultima - primeira,
        eval_freq=BASELINE.eval_freq,
        eval_iter=BASELINE.eval_iter,
        tokenizer=bpe,
        start_context=BASELINE.start_context,
        sample_tokens=BASELINE.sample_tokens,
        history=run.history,
        log=None,
    )
    segundos = time.perf_counter() - inicio
    print(f"  {ultima} épocas: {segundos:.0f} s")

    torch.manual_seed(BASELINE.seed)
    sem_treino = GPTModel(MODEL_CONFIG).to(DEVICE)
    intermediario = GPTModel(MODEL_CONFIG).to(DEVICE)
    intermediario.load_state_dict(pesos_intermediarios)

    estados = []
    for nome, epocas, modelo in (
        ("sem treino", 0, sem_treino),
        (f"{primeira} épocas", primeira, intermediario),
        (f"{ultima} épocas", ultima, run.model),
    ):
        modelo.eval()
        estados.append(
            State(
                nome,
                epocas,
                modelo,
                loader_loss(modelo, run.train_loader),
                loader_loss(modelo, run.val_loader),
            )
        )
    return estados, run.history, run.train_loader, run.val_loader, segundos


def write_states(report: Report, states: list[State], history: TrainingHistory, seconds: float) -> None:
    """A tabela dos três estados e a conferência com resultados/treino.md."""
    report.write("## Os três estados dos pesos")
    report.write()
    report.table(
        ["estado", "épocas", "perda de treino", "perda de validação", "perplexidade de validação"],
        [
            [
                estado.name,
                str(estado.epochs),
                f"{estado.train_loss:.3f}",
                f"{estado.val_loss:.3f}",
                f"{perplexity(estado.val_loss):,.0f}".replace(",", " "),
            ]
            for estado in states
        ],
    )
    report.write(
        "Perdas dos pesos usados na geração, medidas em `eval()` sobre os conjuntos "
        f"inteiros depois do último passo de cada estado. Treino das {EPOCHS[1]} épocas: "
        f"{seconds:.0f} s em CPU."
    )
    report.write()

    if RAPIDO:
        return
    linhas = []
    for epoca, (treino, validacao) in PUBLISHED_LOSSES.items():
        i = last_eval_of_epoch(history, epoca)
        confere = (
            abs(history.train_losses[i] - treino) < LOSS_PRINT_TOLERANCE
            and abs(history.val_losses[i] - validacao) < LOSS_PRINT_TOLERANCE
        )
        linhas.append(
            [
                str(epoca),
                f"{history.steps[i]}",
                f"{history.train_losses[i]:.3f} / {history.val_losses[i]:.3f}",
                f"{treino:.3f} / {validacao:.3f}",
                "confere" if confere else "NÃO confere",
            ]
        )
    report.write(
        "Conferência com `resultados/treino.md` (última avaliação das épocas do T1 e "
        "do T2, treino / validação): é o mesmo treino, e portanto os mesmos pesos."
    )
    report.write()
    report.table(["época", "passo", "este script", "treino.md", "resultado"], linhas)


def first_unseen_rank(probabilities: torch.Tensor, unseen: torch.Tensor) -> torch.Tensor:
    """Posição no ranking, contando do 1, do token nunca visto mais provável.

    Args:
        probabilities: (..., V).
        unseen: máscara (V,) dos tokens que não aparecem no texto de treino.

    Returns:
        Tensor (...) de inteiros: quantos tokens têm probabilidade maior que a
        do melhor nunca visto, mais 1. Um top-k com k abaixo desse número não
        deixa passar nenhum token que o treino não mostrou.
    """
    melhor = probabilities[..., unseen].max(dim=-1, keepdim=True).values
    return (probabilities > melhor).sum(dim=-1) + 1


@torch.no_grad()
def validation_statistics(model: GPTModel, val_loader, unseen: torch.Tensor) -> dict[str, float]:
    """Médias, sobre todas as posições da validação, da forma da distribuição
    do próximo token e de onde o token certo cai no ranking.

    `unseen` é a máscara (V,) dos tokens que não aparecem no texto de treino.
    """
    soma: dict[str, float] = {}
    posicoes = 0
    primeiros_nunca_vistos: list[int] = []

    def acumula(nome: str, valores: torch.Tensor) -> None:
        soma[nome] = soma.get(nome, 0.0) + float(valores.sum())

    for inputs, targets in val_loader:
        probs = torch.softmax(model(inputs.to(DEVICE)), dim=-1).flatten(0, 1)  # (B·T, V)
        alvos = targets.to(DEVICE).flatten()
        ordenadas = probs.sort(dim=-1, descending=True).values
        acumulada = ordenadas.cumsum(dim=-1)
        seguras = probs.clamp_min(torch.finfo(probs.dtype).tiny)
        # Posição do alvo no ranking, contando do 0: quantos tokens têm
        # probabilidade maior que a dele.
        p_alvo = probs.gather(-1, alvos.unsqueeze(-1))
        ranking = (probs > p_alvo).sum(dim=-1)

        acumula("p_max", ordenadas[:, 0])
        acumula("unseen", probs[:, unseen].sum(dim=-1))
        acumula("entropy", -(probs * seguras.log()).sum(dim=-1))
        acumula("n_mass", (acumulada < D1_MASS).sum(dim=-1) + 1)
        for k in D1_TOP_K:
            acumula(f"mass_{k}", acumulada[:, k - 1])
            acumula(f"target_{k}", ranking < k)
        primeiros_nunca_vistos += first_unseen_rank(probs, unseen).tolist()
        posicoes += len(alvos)

    medias = {nome: valor / posicoes for nome, valor in soma.items()}
    medias["first_unseen_min"] = min(primeiros_nunca_vistos)
    medias["first_unseen_median"] = statistics.median(primeiros_nunca_vistos)
    return medias


def experiment_1_distribution(
    report: Report, states: list[State], context: Context, val_loader, references: References, bpe: BPETokenizer
) -> None:
    """D1: a matéria-prima das estratégias, em cada estado dos pesos."""
    report.write("## D1 - A distribuição que cada estratégia recebe")
    report.write()
    report.write(
        "**Pergunta:** toda estratégia parte da mesma distribuição do próximo token. "
        f"Que forma ela tem antes do treino, depois de {EPOCHS[0]} épocas e depois de {EPOCHS[1]}? "
        "Quanto dela a temperatura e o top-k alteram em cada caso?"
    )
    report.write()
    report.write(
        f"**Configuração:** distribuição do próximo token (T = 1, sem top-k) na última "
        f"posição de `{PROMPT}`, e a média das mesmas medidas sobre as posições do "
        "conjunto de validação. Candidatos efetivos = e^entropia: o tamanho de um "
        "vocabulário uniforme com a mesma entropia. Massa do top-k = soma das "
        f"probabilidades dos k tokens mais prováveis. Referência: ln V = {math.log(VOCAB):.3f} nats. "
        f"O texto de treino usa {thousands(len(references.train_vocab))} tokens diferentes dos "
        f"{thousands(VOCAB)} do vocabulário; \"massa nunca vista\" é a probabilidade somada dos "
        f"outros {thousands(VOCAB - len(references.train_vocab))}, que não aparecem nele nenhuma vez."
    )
    report.write()

    nunca_vistos = torch.ones(VOCAB, dtype=torch.bool, device=DEVICE)
    nunca_vistos[sorted(references.train_vocab)] = False

    with torch.no_grad():
        logits = {estado.name: estado.model(context.ids)[0, -1] for estado in states}
    probs = {nome: torch.softmax(z, dim=-1) for nome, z in logits.items()}

    report.write(f"**Na última posição de `{PROMPT}`:**")
    report.write()
    linhas = []
    for estado in states:
        p = probs[estado.name]
        ordenadas = p.sort(descending=True).values
        acumulada = ordenadas.cumsum(dim=-1)
        h = entropy(p)
        linhas.append(
            [
                estado.name,
                token_label(bpe, int(p.argmax())),
                f"{float(ordenadas[0]):.4f}",
                f"{h:.3f}",
                thousands(round(math.exp(h))),
                share(float(acumulada[4])),
                share(float(acumulada[49])),
                thousands(int((acumulada < D1_MASS).sum()) + 1),
                share(float(p[nunca_vistos].sum())),
            ]
        )
    report.table(
        ["estado", "token mais provável", "maior probabilidade", "entropia (nats)", "candidatos efetivos",
         "massa do top-5", "massa do top-50", f"tokens para {D1_MASS:.0%} da massa", "massa nunca vista"],
        linhas,
    )

    report.write(f"Os {D1_TOP} tokens mais prováveis em cada estado:")
    report.write()
    tops = {nome: torch.topk(p, D1_TOP) for nome, p in probs.items()}
    report.table(
        ["posição"] + [estado.name for estado in states],
        [
            [str(i + 1)]
            + [
                f"{token_label(bpe, int(tops[estado.name].indices[i]))} {float(tops[estado.name].values[i]):.4f}"
                for estado in states
            ]
            for i in range(D1_TOP)
        ],
    )

    figura, eixo = new_axes(6.8, 3.6)
    posicoes = range(1, VOCAB + 1)
    for estado, cor in zip(states, SERIES_COLORS):
        eixo.plot(posicoes, probs[estado.name].sort(descending=True).values.tolist(), color=cor,
                  linewidth=1.8, label=estado.name)
    eixo.axhline(1 / VOCAB, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    eixo.annotate("uniforme (1/V)", xy=(1, 1 / VOCAB), xytext=(0, 4), textcoords="offset points",
                  fontsize=8, color=INK_SECONDARY)
    eixo.set_xscale("log")
    eixo.set_yscale("log")
    eixo.set_ylim(bottom=1e-9)
    eixo.set_xlabel("posição do token no ranking (escala log)")
    eixo.set_ylabel("probabilidade (escala log)")
    eixo.set_title(f"A distribuição do próximo token depois de `{PROMPT}`", fontsize=10)
    legend(eixo, loc="lower left")
    report.figure(save_figure(figura, "geracao-distribuicao.png"), "Probabilidade por posição no ranking, nos três estados")

    report.write("**Média sobre as posições do conjunto de validação:**")
    report.write()
    estatisticas = {estado.name: validation_statistics(estado.model, val_loader, nunca_vistos) for estado in states}
    report.table(
        ["estado", "maior probabilidade", "entropia (nats)", "candidatos efetivos",
         f"tokens para {D1_MASS:.0%} da massa", "massa nunca vista"],
        [
            [
                estado.name,
                f"{estatisticas[estado.name]['p_max']:.4f}",
                f"{estatisticas[estado.name]['entropy']:.3f}",
                thousands(round(math.exp(estatisticas[estado.name]["entropy"]))),
                thousands(round(estatisticas[estado.name]["n_mass"])),
                share(estatisticas[estado.name]["unseen"]),
            ]
            for estado in states
        ],
    )

    vistos = len(references.train_vocab)
    report.write(
        "Onde começa a cauda: a posição, no ranking, do token nunca visto mais provável, e "
        f"quantos dos {thousands(vistos)} primeiros do ranking são tokens do texto de treino."
    )
    report.write()
    report.table(
        ["estado", f"melhor nunca visto, em `{PROMPT}`", "na validação: menor posição", "na validação: mediana",
         f"tokens do treino entre os {thousands(vistos)} primeiros, em `{PROMPT}`"],
        [
            [
                estado.name,
                thousands(int(first_unseen_rank(probs[estado.name], nunca_vistos))),
                thousands(int(estatisticas[estado.name]["first_unseen_min"])),
                thousands(round(estatisticas[estado.name]["first_unseen_median"])),
                thousands(int((~nunca_vistos)[probs[estado.name].topk(vistos).indices].sum())),
            ]
            for estado in states
        ],
    )

    report.write(
        "O que o top-k mantém e o que ele custa, na validação. \"Massa\" é a probabilidade "
        "somada dos k tokens mantidos; \"alvo dentro\" é a fração das posições em que o "
        "token que de fato vem a seguir está entre os k."
    )
    report.write()
    report.table(
        ["top_k"] + [f"{estado.name}: massa / alvo dentro" for estado in states],
        [
            [thousands(k)]
            + [
                f"{share(estatisticas[estado.name][f'mass_{k}'])} / {share(estatisticas[estado.name][f'target_{k}'])}"
                for estado in states
            ]
            for k in D1_TOP_K
        ],
    )

    report.write(f"Candidatos efetivos por temperatura, na última posição de `{PROMPT}`:")
    report.write()
    report.table(
        ["temperatura"] + [estado.name for estado in states],
        [
            [f"{temperatura:g}"]
            + [
                thousands(round(math.exp(entropy(next_token_probabilities(logits[estado.name], temperature=temperatura)))))
                for estado in states
            ]
            for temperatura in D1_TEMPERATURES
        ],
    )


def experiment_2_side_by_side(report: Report, sampler: Sampler, trained: State, context: Context, bpe: BPETokenizer) -> None:
    """D2: as estratégias sobre os mesmos pesos e o mesmo contexto."""
    report.write(f"## D2 - Estratégias lado a lado ({trained.name})")
    report.write()
    report.write(
        "**Pergunta:** com os mesmos pesos e o mesmo contexto, o que muda no texto "
        "quando só a estratégia de decodificação muda?"
    )
    report.write()
    report.write(
        f"**Configuração:** modelo de {trained.name}, contexto `{PROMPT}`, {NEW_TOKENS} tokens novos. "
        f"As estratégias que sorteiam geram {SAMPLES} continuações (um lote, gerador com seed "
        f"{SEED}), e a tabela traz a média das {SAMPLES}; a gulosa gera uma só. Medidas, "
        "todas sobre a continuação, sem o contexto:"
    )
    report.write()
    report.write("- **tokens distintos**, **bigramas distintos**: diferentes sobre o total (diversidade dentro do texto);")
    report.write(f"- **continuações diferentes**: quantas das {SAMPLES} são textos distintos (diversidade entre execuções);")
    report.write("- **maior repetição**: a maior sequência do mesmo token em seguida;")
    report.write("- **trigramas repetidos**: fração dos trigramas que já tinham aparecido antes no mesmo texto;")
    report.write(
        "- **log-prob média**: média de ln p(token) segundo o próprio modelo (T = 1, sem top-k), "
        "numa passada sobre o texto final;"
    )
    report.write(
        "- **tokens nunca vistos no treino**: fração dos tokens gerados que não aparecem "
        "nenhuma vez no texto de treino;"
    )
    report.write(
        "- **bigramas no conto**: fração dos pares de tokens vizinhos que existem em algum "
        "lugar do The Verdict (treino + validação) -- o texto é localmente plausível;"
    )
    report.write(
        "- **maior cópia do treino**: o maior trecho contíguo que aparece, token a token, "
        "no texto de treino -- separa o plausível do copiado."
    )
    report.write()

    amostras = {nome: sampler(trained, context, opcoes) for nome, opcoes in STRATEGIES.items()}
    report.table(measure_headers("estratégia"), [measure_row(nome, a) for nome, a in amostras.items()])

    report.write(
        f"Desvio-padrão de cada medida entre as {SAMPLES} continuações (a régua para "
        "comparar duas linhas da tabela acima):"
    )
    report.write()
    sorteadas = {nome: a for nome, a in amostras.items() if nome != GREEDY}
    report.table(
        ["estratégia"] + [titulo for titulo, _ in MEASURES.values()],
        [
            [nome] + [formato(a.deviation(medida)) for medida, (_, formato) in MEASURES.items()]
            for nome, a in sorteadas.items()
        ],
    )

    report.write(f"Primeiro token gerado: quantos diferentes nas {SAMPLES} continuações.")
    report.write()
    report.table(
        ["estratégia", f"primeiros tokens distintos (de {SAMPLES})"],
        [[nome, str(a.distinct_first_tokens)] for nome, a in sorteadas.items()],
    )

    report.write("Textos gerados (a primeira continuação de cada estratégia; o contexto é o começo de todos):")
    report.write()
    for nome, amostra in amostras.items():
        report.write(f"- **{nome}:** {text_of(bpe, context, amostra.tokens[0])}")
    report.write()


def sweep_figure(report: Report, x: list[float], samples: list[Sample], name: str, title: str,
                 xlabel: str, caption: str, log_x: bool = False, ticks: list[str] | None = None) -> None:
    """Diversidade, repetição e coerência local ao longo de um parâmetro."""
    figura, eixo = new_axes(6.8, 3.6)
    series = (
        ("distinct_2", "bigramas distintos (diversidade)", "-"),
        ("repeat_3", "trigramas repetidos (repetição)", "--"),
        ("corpus_2", "bigramas no conto (coerência local)", ":"),
    )
    for (medida, rotulo, estilo), cor in zip(series, SERIES_COLORS):
        eixo.plot(x, [100 * a.mean(medida) for a in samples], color=cor, linewidth=1.8,
                  linestyle=estilo, marker="o", markersize=4, label=rotulo)
    if log_x:
        eixo.set_xscale("log")
    if ticks is not None:
        eixo.set_xticks(x)
        eixo.set_xticklabels(ticks)
        eixo.minorticks_off()
    eixo.set_ylim(-3, 103)
    eixo.set_xlabel(xlabel)
    eixo.set_ylabel("% da continuação")
    eixo.set_title(title, fontsize=10)
    legend(eixo, loc="center right")
    report.figure(save_figure(figura, name), caption)


def experiment_3_temperature(report: Report, sampler: Sampler, trained: State, context: Context) -> None:
    """D3: a temperatura, sem top-k."""
    report.write(f"## D3 - Temperatura ({trained.name})")
    report.write()
    report.write(
        "**Pergunta:** como diversidade, repetição e coerência variam com a temperatura, "
        "e existe uma faixa em que o texto é diverso sem deixar de ser plausível?"
    )
    report.write()
    report.write(
        f"**Configuração:** modelo de {trained.name}, sem top-k, temperaturas de 0 (gulosa) a "
        f"{D3_TEMPERATURES[-1]:g}. {SAMPLES} continuações de {NEW_TOKENS} tokens por temperatura; "
        "medidas do D2."
    )
    report.write()

    amostras = [sampler(trained, context, {"temperature": t}) for t in D3_TEMPERATURES]
    report.table(
        measure_headers("temperatura"),
        [
            measure_row("0 (gulosa)" if t == 0 else f"{t:g}", a)
            for t, a in zip(D3_TEMPERATURES, amostras)
        ],
    )
    sweep_figure(
        report, list(D3_TEMPERATURES), amostras, "geracao-temperatura.png",
        f"Temperatura no modelo de {trained.name}, sem top-k", "temperatura (0 = gulosa)",
        "Diversidade, repetição e coerência local por temperatura",
    )


def experiment_4_top_k(report: Report, sampler: Sampler, trained: State, context: Context) -> None:
    """D4: o top-k, sozinho e combinado com a temperatura."""
    report.write(f"## D4 - Top-k ({trained.name})")
    report.write()
    report.write(
        "**Pergunta:** o que o corte do top-k muda no texto, e o que ele faz que a "
        "temperatura não faz?"
    )
    report.write()
    report.write(
        f"**Configuração:** modelo de {trained.name}, T = 1, k de 1 a 200 e sem top-k (o "
        f"vocabulário inteiro, {thousands(VOCAB)} tokens). {SAMPLES} continuações de {NEW_TOKENS} "
        "tokens por valor; medidas do D2. Com k = 1 só o mais provável sobrevive: é a gulosa "
        "por outro caminho, e as continuações saem todas iguais."
    )
    report.write()

    def opcoes(temperature: float, k: int | None) -> dict:
        return {"temperature": temperature} if k is None else {"temperature": temperature, "top_k": k}

    amostras = [sampler(trained, context, opcoes(1.0, k)) for k in D4_TOP_K]
    rotulos = ["sem top-k" if k is None else str(k) for k in D4_TOP_K]
    report.table(measure_headers("top_k"), [measure_row(r, a) for r, a in zip(rotulos, amostras)])
    sweep_figure(
        report, [VOCAB if k is None else k for k in D4_TOP_K], amostras, "geracao-top-k.png",
        f"Top-k no modelo de {trained.name}, T = 1", "top_k (escala log)",
        "Diversidade, repetição e coerência local por top-k", log_x=True,
        ticks=[r if k is not None else "sem\n(V)" for r, k in zip(rotulos, D4_TOP_K)],
    )

    report.write(
        "Temperatura x top-k: em cada célula, bigramas no conto / bigramas distintos / "
        "maior cópia do treino."
    )
    report.write()
    linhas = []
    for temperatura in D4_GRID_TEMPERATURES:
        linha = [f"T = {temperatura:g}"]
        for k in D4_GRID_TOP_K:
            a = sampler(trained, context, opcoes(temperatura, k))
            linha.append(f"{share(a.mean('corpus_2'))} / {share(a.mean('distinct_2'))} / {count(a.mean('copy'))}")
        linhas.append(linha)
    report.table(
        ["temperatura"] + ["sem top-k" if k is None else f"top_k = {k}" for k in D4_GRID_TOP_K],
        linhas,
    )


def variance_shares(matrix: list[list[float]]) -> tuple[float, float, float]:
    """Decompõe a variação de uma tabela estratégia x estado em três parcelas.

    Args:
        matrix: uma linha por estratégia, uma coluna por estado (médias).

    Returns:
        Frações da soma de quadrados total explicadas pelos pesos (as médias
        das colunas), pela estratégia (as médias das linhas) e pela interação
        (o que sobra: a estratégia agindo de forma diferente em cada estado).
        Somam 1. É a decomposição da análise de variância de dois fatores, sem
        teste de hipótese: descreve as 18 médias, não generaliza.
    """
    linhas, colunas = len(matrix), len(matrix[0])
    geral = statistics.fmean(valor for linha in matrix for valor in linha)
    medias_linha = [statistics.fmean(linha) for linha in matrix]
    medias_coluna = [statistics.fmean(matrix[i][j] for i in range(linhas)) for j in range(colunas)]

    total = sum((valor - geral) ** 2 for linha in matrix for valor in linha)
    if total == 0:
        return 0.0, 0.0, 0.0
    pesos = linhas * sum((media - geral) ** 2 for media in medias_coluna)
    estrategia = colunas * sum((media - geral) ** 2 for media in medias_linha)
    return pesos / total, estrategia / total, (total - pesos - estrategia) / total


def strategy_figure(report: Report, states: list[State], grid: dict, measure: str, name: str, title: str, caption: str) -> None:
    figura, eixo = new_axes(6.8, 3.6)
    x = range(len(STRATEGIES))
    for estado, cor, estilo in zip(states, SERIES_COLORS, ("-", "--", ":")):
        eixo.plot(x, [100 * grid[nome][estado.name].mean(measure) for nome in STRATEGIES], color=cor,
                  linewidth=1.8, linestyle=estilo, marker="o", markersize=4, label=estado.name)
    eixo.set_xticks(list(x))
    eixo.set_xticklabels(STRATEGY_TICKS, fontsize=8)
    eixo.set_ylim(-3, 103)
    eixo.set_ylabel("% da continuação")
    eixo.set_title(title, fontsize=10)
    legend(eixo, loc="center right")
    report.figure(save_figure(figura, name), caption)


def experiment_5_strategy_and_training(
    report: Report, sampler: Sampler, states: list[State], context: Context, second: Context, bpe: BPETokenizer
) -> None:
    """D5: a grade completa, e a parte de cada fator."""
    report.write("## D5 - Estratégia x treino")
    report.write()
    report.write(
        "**Pergunta:** do que se vê no texto gerado, o que é efeito da estratégia de "
        "decodificação e o que é efeito do treino dos pesos?"
    )
    report.write()
    report.write(
        f"**Configuração:** as {len(STRATEGIES)} estratégias do D2 em cada um dos {len(states)} estados dos "
        f"pesos, com o mesmo contexto e a mesma seed nos sorteios: {len(STRATEGIES) * len(states)} "
        "combinações. Uma tabela por medida, com a estratégia nas linhas e o estado nas colunas."
    )
    report.write()

    grade = {
        nome: {estado.name: sampler(estado, context, opcoes) for estado in states}
        for nome, opcoes in STRATEGIES.items()
    }

    for medida, (titulo, formato) in MEASURES.items():
        report.write(f"**{titulo[0].upper()}{titulo[1:]}:**")
        report.write()
        report.table(
            ["estratégia"] + [estado.name for estado in states],
            [
                [nome] + [formato(grade[nome][estado.name].mean(medida)) for estado in states]
                for nome in STRATEGIES
            ],
        )

    report.write(
        "**De onde vem a variação.** A soma de quadrados das "
        f"{len(STRATEGIES) * len(states)} médias de cada medida, repartida entre os pesos (as "
        "médias das colunas), a estratégia (as médias das linhas) e a interação (a "
        "estratégia agindo de forma diferente em cada estado):"
    )
    report.write()
    linhas = []
    for medida, (titulo, _) in MEASURES.items():
        matriz = [[grade[nome][estado.name].mean(medida) for estado in states] for nome in STRATEGIES]
        pesos, estrategia, interacao = variance_shares(matriz)
        linhas.append([titulo, share(pesos), share(estrategia), share(interacao)])
    report.table(["medida", "pesos", "estratégia", "interação"], linhas)

    strategy_figure(
        report, states, grade, "corpus_2", "geracao-estrategia-treino-coerencia.png",
        "Bigramas que existem no conto, por estratégia e estado dos pesos",
        "Coerência local por estratégia, nos três estados",
    )
    strategy_figure(
        report, states, grade, "distinct_2", "geracao-estrategia-treino-diversidade.png",
        "Bigramas distintos, por estratégia e estado dos pesos",
        "Diversidade por estratégia, nos três estados",
    )

    report.write(
        "**O mesmo sorteio em pesos diferentes.** Fração dos tokens gerados que são "
        "os mesmos do modelo sem treino, na mesma posição da mesma continuação "
        f"({SAMPLES} x {NEW_TOKENS} posições). O gerador tem a mesma seed nos três estados: "
        "quando o token coincide, quem o escolheu foi o sorteio, não os pesos."
    )
    report.write()
    base = states[0].name
    linhas = []
    for nome in STRATEGIES:
        if nome == GREEDY:
            continue
        referencia = grade[nome][base].tokens
        linha = [nome]
        for estado in states[1:]:
            tokens = grade[nome][estado.name].tokens
            iguais = sum(a == b for c, r in zip(tokens, referencia) for a, b in zip(c, r))
            linha.append(share(iguais / (len(tokens) * NEW_TOKENS)))
        linhas.append(linha)
    report.table(["estratégia"] + [estado.name for estado in states[1:]], linhas)

    report.write("**Textos** (a primeira continuação de cada combinação):")
    report.write()
    for nome in D5_TEXT_STRATEGIES:
        for estado in states:
            report.write(f"- **{nome}, {estado.name}:** {text_of(bpe, context, grade[nome][estado.name].tokens[0])}")
    report.write()

    report.write(
        f"**Outro contexto.** A mesma grade a partir dos {second.ids.shape[1]} primeiros tokens do "
        "texto de validação, que o modelo não viu no treino: "
        f"{text_of(bpe, second, [])}. Em cada célula, bigramas no conto / bigramas distintos / "
        "maior cópia do treino."
    )
    report.write()
    linhas = []
    for nome, opcoes in STRATEGIES.items():
        linha = [nome]
        for estado in states:
            a = sampler(estado, second, opcoes)
            linha.append(f"{share(a.mean('corpus_2'))} / {share(a.mean('distinct_2'))} / {count(a.mean('copy'))}")
        linhas.append(linha)
    report.table(["estratégia"] + [estado.name for estado in states], linhas)


def main() -> None:
    utf8_console()
    if RAPIDO:
        destino = Path(tempfile.mkdtemp(prefix="sprint05-geracao-rapido-"))
        comum.FIGURES_DIR = destino / "figuras"
        report = Report(destino / "geracao.md")
        print(f"modo rápido: resultados em {destino}")
    else:
        report = Report(comum.RESULTS_DIR / "geracao.md")
    bpe = BPETokenizer()
    texto = load_verdict()

    report.write("# Sprint 5 - Estratégias de geração: resultados dos experimentos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint05/geracao_experimentos.py`. Não editar à mão: rode o script novamente."
    )
    report.write()
    report.write(
        f"Modelo: BASE_CONFIG, treinado com BASE_TRAINING_CONFIG (seed {BASELINE.seed}) em três "
        f"estados: sem treino, {EPOCHS[0]} épocas e {EPOCHS[1]} épocas, tirados de um treino só. "
        f"Contexto: `{PROMPT}`, {NEW_TOKENS} tokens novos por continuação. Estratégias que sorteiam: "
        f"{SAMPLES} continuações num lote, com `torch.Generator` semeado em {SEED}; as tabelas trazem a "
        "média delas. A gulosa gera uma continuação. Decodificação de `src/generation/`, a mesma da "
        "Sprint 4. Treino e geração em CPU, seeds fixas: rodar de novo reproduz tudo. Figuras em "
        "`resultados/figuras/`."
    )
    report.write()

    print("treino")
    estados, historico, train_loader, val_loader, segundos = train_states(texto, bpe)
    write_states(report, estados, historico, segundos)
    report.save()

    ids_treino = train_loader.dataset.token_ids.tolist()
    ids_validacao = val_loader.dataset.token_ids.tolist()
    ids_conto = bpe.encode(texto)
    referencias = References(ids_treino, set(ids_treino), set(zip(ids_conto, ids_conto[1:])))
    sampler = Sampler(referencias)

    contexto = Context("livro", text_to_token_ids(PROMPT, bpe).to(DEVICE))
    segundo = Context(
        "validação",
        torch.tensor([ids_validacao[:D5_SECOND_CONTEXT_TOKENS]], dtype=torch.long, device=DEVICE),
    )
    treinado = estados[1]

    # O relatório é salvo depois de cada experimento, como no treino_experimentos.py.
    for nome, experimento in (
        ("D1", lambda: experiment_1_distribution(report, estados, contexto, val_loader, referencias, bpe)),
        ("D2", lambda: experiment_2_side_by_side(report, sampler, treinado, contexto, bpe)),
        ("D3", lambda: experiment_3_temperature(report, sampler, treinado, contexto)),
        ("D4", lambda: experiment_4_top_k(report, sampler, treinado, contexto)),
        ("D5", lambda: experiment_5_strategy_and_training(report, sampler, estados, contexto, segundo, bpe)),
    ):
        print(nome)
        experimento()
        report.save()

    print(f"\nResultados salvos em {report.path}")


if __name__ == "__main__":
    main()
