"""Experimentos de avaliação do modelo da Sprint 5 (Entregável 2).

A perda (cross-entropy) e a perplexidade medidas no modelo do projeto, antes e
durante o treino de referência: o que cada número significa e como se
relaciona com acertar o próximo token.

AV1  A perda passo a passo numa sequência de validação, no modelo sem treino
AV2  O estado inicial: 5 inicializações contra o chute uniforme
AV3  A evolução durante o treino: perda, perplexidade e acerto, época a época
AV4  A perda por posição na janela: quanto o contexto vale, antes e depois
AV5  A distribuição da probabilidade dada ao token certo: de onde vem a perda

Modelo: BASE_CONFIG (28,9 milhões de parâmetros). Treino: BASE_TRAINING_CONFIG
(os hiperparâmetros do livro com seed 42, avaliação sobre os conjuntos
inteiros), em CPU. Dados: The Verdict, 90% treino e 10% validação, janelas de
128 tokens sem sobreposição, lotes de 2.

As medidas de cada época (AV3 a AV5) são tiradas entre chamadas de
train_model, retomando o histórico, e dentro de torch.random.fork_rng: medir
não consome o gerador global, e o treino segue idêntico ao de uma chamada só --
o mesmo do T1 em treino_experimentos.py.

Uso:
    python experimentos/sprint05/avaliacao_experimentos.py
"""

import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comum import (  # noqa: E402
    DEVICE,
    RESULTS_DIR,
    SERIES_COLORS,
    Report,
    legend,
    new_axes,
    save_figure,
    thousands,
    utf8_console,
)
from src.corpus import load_verdict  # noqa: E402
from src.dataset import create_dataloader  # noqa: E402
from src.model import BASE_CONFIG, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.training import (  # noqa: E402
    BASE_TRAINING_CONFIG,
    batch_loss,
    create_optimizer,
    create_train_val_loaders,
    evaluate,
    perplexity,
    split_text,
    target_log_probabilities,
    train_model,
)

RESULTS_PATH = RESULTS_DIR / "avaliacao.md"

MODEL_CONFIG = BASE_CONFIG
TRAINING = BASE_TRAINING_CONFIG
VOCAB = MODEL_CONFIG.vocab_size

AV1_POSITIONS = 8
AV2_SEEDS = (42, 43, 44, 45, 46)
TOP_K = 5
# Faixas de posição na janela de 128 tokens (AV4): mais finas no começo, onde
# o contexto ainda é curto e cada token a mais pesa.
AV4_BINS = ((0, 1), (1, 4), (4, 16), (16, 32), (32, 64), (64, 128))
# Janelas do AV4 deslocadas de 16 em 16: cada token é previsto em até 8
# posições diferentes, e a média de cada posição deixa de ser a de poucos
# tokens fixos -- com o stride do treino (128), a posição 0 seria sempre um
# dos 36 tokens que abrem as janelas, e na validação só 4 tokens por posição.
AV4_STRIDE = 16
AV4_BATCH = 8
# Faixas de probabilidade dada ao alvo (AV5).
AV5_BINS = ((0.5, 1.0), (0.1, 0.5), (0.01, 0.1), (1e-3, 0.01), (1e-4, 1e-3), (0.0, 1e-4))
WORST_SHARE = 0.10
AV5_TOP_TOKENS = 10


@dataclass
class TokenStats:
    """Medidas por token de um conjunto inteiro."""

    log_probs: torch.Tensor  # (N,) log p(alvo) de cada token avaliado
    targets: torch.Tensor  # (N,) o Token ID alvo de cada um, alinhado com log_probs
    top1: float  # fração de tokens em que o alvo é o argmax
    topk: float  # fração em que o alvo está entre os TOP_K mais prováveis
    position_loss: torch.Tensor  # (T,) perda média em cada posição da janela

    @property
    def loss(self) -> float:
        return float(-self.log_probs.mean())


@torch.no_grad()
def token_stats(model: GPTModel, loader) -> TokenStats:
    """Percorre o loader em eval() e guarda log p(alvo), acertos e perda por posição.

    Dentro de fork_rng: iterar um DataLoader sorteia (o embaralhamento e a
    semente dos workers), e a medida não pode mexer no gerador do treino.
    """
    was_training = model.training
    model.eval()
    log_probs, alvos, top1, topk, total = [], [], 0, 0, 0
    position_sum = torch.zeros(MODEL_CONFIG.context_length, dtype=torch.float64)
    sequences = 0
    try:
        with torch.random.fork_rng(devices=[]):
            for inputs, targets in loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                logits = model(inputs)
                lp = target_log_probabilities(logits, targets)
                ranking = logits.topk(TOP_K, dim=-1).indices

                log_probs.append(lp.flatten().double().cpu())
                alvos.append(targets.flatten().cpu())
                top1 += int((ranking[..., 0] == targets).sum())
                topk += int((ranking == targets.unsqueeze(-1)).any(dim=-1).sum())
                total += targets.numel()
                position_sum += (-lp).double().sum(dim=0).cpu()
                sequences += targets.shape[0]
    finally:
        model.train(was_training)

    return TokenStats(
        log_probs=torch.cat(log_probs),
        targets=torch.cat(alvos),
        top1=top1 / total,
        topk=topk / total,
        position_loss=position_sum / sequences,
    )


def token_label(bpe: BPETokenizer, token_id: int) -> str:
    """Token entre crases, com o espaço inicial visível e o `|` escapado."""
    return "`" + repr(bpe.decode([token_id])).replace("|", "\\|") + "`"


def experiment_1_loss_by_hand(report: Report, model: GPTModel, val_loader, bpe: BPETokenizer) -> None:
    """AV1: a cross-entropy de uma sequência, token a token."""
    report.write("## AV1 - A perda passo a passo, no modelo sem treino")
    report.write()
    report.write(
        "**Pergunta:** o que a cross-entropy mede, token a token? E por que o "
        "modelo sem treino fica perto de ln V?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG com seed {TRAINING.seed}, sem treino, em "
        "`eval()`. O primeiro lote de validação (2 sequências de 128 tokens); a "
        f"tabela mostra as {AV1_POSITIONS} primeiras posições da primeira sequência."
    )
    report.write()

    entradas, alvos = next(iter(val_loader))
    with torch.no_grad():
        logits = model(entradas)
        perda_lote = batch_loss(model, entradas, alvos).item()
    log_do_alvo = target_log_probabilities(logits, alvos)
    logit_do_alvo = logits.gather(-1, alvos.unsqueeze(-1))
    # Posição do alvo no ranking: 1 + quantos tokens receberam logit maior.
    ranking = (logits > logit_do_alvo).sum(dim=-1) + 1

    linhas = []
    for t in range(AV1_POSITIONS):
        p = log_do_alvo[0, t].exp().item()
        linhas.append(
            [
                str(t),
                token_label(bpe, int(entradas[0, t])),
                token_label(bpe, int(alvos[0, t])),
                f"{p:.3e}",
                f"{p * VOCAB:.2f}",
                f"{-log_do_alvo[0, t].item():.3f}",
                thousands(int(ranking[0, t])),
            ]
        )
    report.table(
        ["posição", "entrada", "alvo", "p(alvo)", "p(alvo) · V", "-log p(alvo)", "alvo no ranking"],
        linhas,
    )

    media_seq = -log_do_alvo[0].mean().item()
    report.table(
        ["grandeza", "valor"],
        [
            [f"média de -log p(alvo) nas 128 posições da sequência 0", f"{media_seq:.4f}"],
            ["média nas 2 x 128 posições do lote (`batch_loss`)", f"{perda_lote:.4f}"],
            ["mesma média pelo passo a passo (`-target_log_probabilities(...).mean()`)",
             f"{-log_do_alvo.mean().item():.4f}"],
            ["ln V (chute uniforme entre 50 257 tokens)", f"{math.log(VOCAB):.4f}"],
            ["perplexidade do lote, exp(perda)", thousands(round(perplexity(perda_lote)))],
            ["V", thousands(VOCAB)],
            ["posição mediana do alvo no ranking (lote inteiro)", thousands(int(ranking.float().median()))],
        ],
    )
    report.write(
        "A coluna p(alvo) · V compara cada probabilidade com a do chute "
        "uniforme (1/V): valores perto de 1 querem dizer que o modelo não "
        "favorece o token certo nem o rejeita."
    )
    report.write()


def experiment_2_initial_state(report: Report, train_loader, val_loader) -> None:
    """AV2: o ponto de partida não depende da semente nem do conjunto."""
    report.write("## AV2 - O estado inicial")
    report.write()
    report.write(
        "**Pergunta:** de onde o treino parte? A perda inicial depende da "
        "inicialização, ou do conjunto (treino ou validação)?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_CONFIG com as seeds {', '.join(map(str, AV2_SEEDS))}, "
        "sem treino, em `eval()`; perdas sobre os conjuntos inteiros. O desvio "
        "dos logits é medido no primeiro lote de validação."
    )
    report.write()

    entradas, _ = next(iter(val_loader))
    linhas, treinos, validacoes, variancias = [], [], [], []
    for seed in AV2_SEEDS:
        torch.manual_seed(seed)
        modelo = GPTModel(MODEL_CONFIG).eval()
        perda_treino, perda_validacao = evaluate(modelo, train_loader, val_loader)
        with torch.no_grad():
            desvio = modelo(entradas).std().item()
        treinos.append(perda_treino)
        validacoes.append(perda_validacao)
        variancias.append(desvio**2)
        linhas.append(
            [
                str(seed),
                f"{perda_treino:.4f}",
                f"{perda_validacao:.4f}",
                thousands(round(perplexity(perda_treino))),
                thousands(round(perplexity(perda_validacao))),
                f"{desvio:.4f}",
            ]
        )

    def media_desvio(valores: list[float]) -> str:
        media = sum(valores) / len(valores)
        desvio = (sum((v - media) ** 2 for v in valores) / (len(valores) - 1)) ** 0.5
        return f"{media:.4f} ± {desvio:.4f}"

    linhas.append(["média ± desvio", media_desvio(treinos), media_desvio(validacoes), "", "", ""])
    report.table(
        ["seed", "perda de treino", "perda de validação", "perplexidade treino",
         "perplexidade validação", "desvio dos logits"],
        linhas,
    )

    variancia = sum(variancias) / len(variancias)
    report.table(
        ["referência", "perda", "perplexidade"],
        [
            ["chute uniforme: ln V", f"{math.log(VOCAB):.4f}", thousands(VOCAB)],
            [
                f"logits aleatórios com variância σ² = {variancia:.4f}: ln V + σ²/2",
                f"{math.log(VOCAB) + variancia / 2:.4f}",
                thousands(round(math.exp(math.log(VOCAB) + variancia / 2))),
            ],
        ],
    )
    report.write(
        "Para logits independentes com variância σ², a média de -log softmax "
        "no alvo é E[logsumexp] - E[logit do alvo] ≈ (ln V + σ²/2) - 0: o "
        "modelo sem treino perde um pouco MAIS que o chute uniforme, porque "
        "logits aleatórios já concentram probabilidade em tokens arbitrários."
    )
    report.write()


def experiment_3_evolution(report: Report, text: str, bpe: BPETokenizer):
    """AV3: o treino de referência, medido no fim de cada época."""
    report.write("## AV3 - A evolução durante o treino")
    report.write()
    report.write(
        "**Pergunta:** como perda, perplexidade e acerto do próximo token "
        "evoluem durante o treino, no treino e na validação?"
    )
    report.write()
    report.write(
        f"**Configuração:** BASE_TRAINING_CONFIG -- AdamW, lr {TRAINING.learning_rate}, "
        f"weight decay {TRAINING.weight_decay}, lotes de {TRAINING.batch_size}, "
        f"{TRAINING.num_epochs} épocas, seed {TRAINING.seed}. Medidas no fim de cada "
        f"época, nos conjuntos inteiros, em `eval()`. Acerto top-1: o alvo é o "
        f"argmax; top-{TOP_K}: o alvo está entre os {TOP_K} mais prováveis."
    )
    report.write()

    train_loader, val_loader = create_train_val_loaders(
        text,
        bpe,
        context_length=MODEL_CONFIG.context_length,
        batch_size=TRAINING.batch_size,
        train_ratio=TRAINING.train_ratio,
        stride=TRAINING.stride,
    )
    # Janelas deslocadas de 16 tokens, só para a perda por posição (AV4).
    por_posicao_loaders = {
        nome: create_dataloader(
            parte, bpe, context_length=MODEL_CONFIG.context_length, stride=AV4_STRIDE,
            batch_size=AV4_BATCH, shuffle=False, drop_last=False,
        )
        for nome, parte in zip(("treino", "validação"), split_text(text, TRAINING.train_ratio))
    }

    def perda_por_posicao() -> dict[str, torch.Tensor]:
        return {nome: token_stats(model, loader).position_loss for nome, loader in por_posicao_loaders.items()}

    # A mesma ordem de train_gpt: seed, modelo, otimizador.
    torch.manual_seed(TRAINING.seed)
    model = GPTModel(MODEL_CONFIG).to(DEVICE)
    optimizer = create_optimizer(model, TRAINING)

    por_posicao = {"sem treino": perda_por_posicao()}
    por_epoca = [(0, token_stats(model, train_loader), token_stats(model, val_loader))]
    historico = None
    for epoca in range(1, TRAINING.num_epochs + 1):
        historico = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            num_epochs=1,
            eval_freq=TRAINING.eval_freq,
            eval_iter=TRAINING.eval_iter,
            tokenizer=bpe,
            start_context=TRAINING.start_context,
            sample_tokens=TRAINING.sample_tokens,
            history=historico,
            log=None,
        )
        por_epoca.append((epoca, token_stats(model, train_loader), token_stats(model, val_loader)))
        print(f"  época {epoca}: treino {por_epoca[-1][1].loss:.3f}, validação {por_epoca[-1][2].loss:.3f}")
    por_posicao[f"época {TRAINING.num_epochs}"] = perda_por_posicao()
    janelas = {nome: len(loader.dataset) for nome, loader in por_posicao_loaders.items()}

    linhas = []
    for epoca, treino, validacao in por_epoca:
        linhas.append(
            [
                str(epoca),
                f"{treino.loss:.3f}",
                f"{validacao.loss:.3f}",
                f"{validacao.loss - treino.loss:.3f}",
                f"{perplexity(treino.loss):,.1f}".replace(",", " "),
                f"{perplexity(validacao.loss):,.1f}".replace(",", " "),
                f"{treino.top1:.1%}",
                f"{validacao.top1:.1%}",
                f"{validacao.topk:.1%}",
            ]
        )
    report.table(
        ["época", "perda treino", "perda validação", "diferença", "perplexidade treino",
         "perplexidade validação", "top-1 treino", "top-1 validação", f"top-{TOP_K} validação"],
        linhas,
    )

    melhor = min(por_epoca[1:], key=lambda item: item[2].loss)
    report.write(
        f"Menor perda de validação no fim de época: {melhor[2].loss:.3f} "
        f"(perplexidade {perplexity(melhor[2].loss):.1f}), na época {melhor[0]}. "
        f"Em {thousands(len(por_epoca[0][2].log_probs))} tokens de validação, "
        f"o acerto top-1 de 1/V seria {1 / VOCAB:.4%}."
    )
    report.write()

    epocas = [e for e, _, _ in por_epoca]
    figura, eixo = new_axes(7.0, 3.6)
    eixo.plot(epocas, [perplexity(t.loss) for _, t, _ in por_epoca], color=SERIES_COLORS[0],
              marker="o", markersize=3, linewidth=1.6, label="treino")
    eixo.plot(epocas, [perplexity(v.loss) for _, _, v in por_epoca], color=SERIES_COLORS[1],
              marker="o", markersize=3, linewidth=1.6, linestyle="--", label="validação")
    eixo.axhline(VOCAB, color=SERIES_COLORS[2], linewidth=1.0, linestyle=":", label="V (chute uniforme)")
    eixo.set_yscale("log")
    eixo.set_xlabel("épocas")
    eixo.set_ylabel("perplexidade (escala log)")
    eixo.set_title("Perplexidade no fim de cada época", fontsize=10)
    legend(eixo, loc="lower left")
    report.figure(save_figure(figura, "avaliacao-perplexidade.png"), "Perplexidade de treino e validação por época")

    figura, eixo = new_axes(7.0, 3.6)
    eixo.plot(epocas, [t.top1 for _, t, _ in por_epoca], color=SERIES_COLORS[0],
              marker="o", markersize=3, linewidth=1.6, label="top-1 treino")
    eixo.plot(epocas, [v.top1 for _, _, v in por_epoca], color=SERIES_COLORS[1],
              marker="o", markersize=3, linewidth=1.6, linestyle="--", label="top-1 validação")
    eixo.plot(epocas, [v.topk for _, _, v in por_epoca], color=SERIES_COLORS[3],
              marker="o", markersize=3, linewidth=1.6, linestyle=":", label=f"top-{TOP_K} validação")
    eixo.set_ylim(0, 1)
    eixo.set_xlabel("épocas")
    eixo.set_ylabel("fração de tokens")
    eixo.set_title("Acerto do próximo token por época", fontsize=10)
    legend(eixo, loc="upper left")
    report.figure(save_figure(figura, "avaliacao-acerto.png"), "Acerto top-1 e top-5 por época")

    return por_epoca, historico, por_posicao, janelas


def experiment_4_position_loss(report: Report, por_posicao: dict, janelas: dict[str, int]) -> None:
    """AV4: perda média em cada posição da janela de 128 tokens."""
    report.write("## AV4 - A perda por posição na janela")
    report.write()
    report.write(
        "**Pergunta:** a perda é a média sobre as 128 posições de cada janela. "
        "Todas as posições contribuem igual? Quanto o contexto anterior ajuda, "
        "antes e depois do treino?"
    )
    report.write()
    report.write(
        "**Configuração:** a perda média de cada posição t (o token t+1 previsto "
        "a partir dos tokens 0..t), no modelo sem treino e no fim do treino do "
        f"AV3. Janelas de {MODEL_CONFIG.context_length} tokens deslocadas de {AV4_STRIDE} em "
        f"{AV4_STRIDE} ({janelas['treino']} de treino, {janelas['validação']} de validação): cada "
        f"token é previsto em até {MODEL_CONFIG.context_length // AV4_STRIDE} posições diferentes. "
        "Com as janelas do treino (stride 128), a posição 0 seria sempre um dos "
        "36 tokens que abrem as janelas, e a validação teria só 4 tokens por posição."
    )
    report.write()

    series = [
        (f"{conjunto}, {estado}", perdas)
        for estado, por_conjunto in por_posicao.items()
        for conjunto, perdas in por_conjunto.items()
    ]

    def faixa(perdas: torch.Tensor, inicio: int, fim: int) -> str:
        return f"{float(perdas[inicio:fim].mean()):.3f}"

    report.table(
        ["posições", *[nome for nome, _ in series]],
        [
            [f"{inicio}" if fim - inicio == 1 else f"{inicio}-{fim - 1}",
             *[faixa(perdas, inicio, fim) for _, perdas in series]]
            for inicio, fim in AV4_BINS
        ],
    )

    figura, eixo = new_axes(7.0, 3.6)
    posicoes = list(range(MODEL_CONFIG.context_length))
    estilos = [(SERIES_COLORS[0], ":"), (SERIES_COLORS[1], ":"), (SERIES_COLORS[0], "-"), (SERIES_COLORS[1], "--")]
    for (nome, perdas), (cor, estilo) in zip(series, estilos):
        eixo.plot(posicoes, perdas.tolist(), color=cor, linestyle=estilo, linewidth=1.3, label=nome)
    eixo.set_xlabel("posição na janela (tokens de contexto - 1)")
    eixo.set_ylabel("perda média (nats)")
    eixo.set_ylim(bottom=0)
    eixo.set_title("Perda por posição na janela", fontsize=10)
    legend(eixo, loc="center right")
    report.figure(save_figure(figura, "avaliacao-perda-por-posicao.png"), "Perda média por posição na janela")


def experiment_5_target_probability(report: Report, por_epoca, bpe: BPETokenizer) -> None:
    """AV5: como a probabilidade do alvo se distribui, e quanto da perda vem dos piores tokens."""
    report.write("## AV5 - A probabilidade dada ao token certo")
    report.write()
    report.write(
        "**Pergunta:** a perda é a média de -log p(alvo). Como p(alvo) se "
        "distribui pelos tokens, e quanto da perda vem dos tokens em que o "
        "modelo mais erra?"
    )
    report.write()
    report.write(
        f"**Configuração:** p(alvo) de cada token, no modelo sem treino e no fim "
        f"da época {por_epoca[-1][0]} do AV3. \"Piores {WORST_SHARE:.0%}\": os tokens "
        "de maior -log p(alvo)."
    )
    report.write()

    _, _, validacao_0 = por_epoca[0]
    _, treino_n, validacao_n = por_epoca[-1]
    series = [
        ("validação, sem treino", validacao_0),
        ("treino, treinado", treino_n),
        ("validação, treinado", validacao_n),
    ]

    def fracao(log_probs: torch.Tensor, baixo: float, alto: float) -> str:
        p = log_probs.exp()
        dentro = (p >= baixo) & ((p < alto) if alto < 1.0 else (p <= alto))
        return f"{float(dentro.double().mean()):.1%}"

    def rotulo(baixo: float, alto: float) -> str:
        if baixo == 0.0:
            return f"p < {alto:g}"
        return f"{baixo:g} ≤ p" + (" ≤ 1" if alto == 1.0 else f" < {alto:g}")

    linhas = [
        [rotulo(baixo, alto), *[fracao(stats.log_probs, baixo, alto) for _, stats in series]]
        for baixo, alto in AV5_BINS
    ]

    def piores(log_probs: torch.Tensor) -> str:
        perdas = (-log_probs).sort(descending=True).values
        corte = max(1, int(len(perdas) * WORST_SHARE))
        return f"{float(perdas[:corte].sum() / perdas.sum()):.1%}"

    linhas += [
        ["p < 1/V (abaixo do chute uniforme)",
         *[f"{float((s.log_probs.exp() < 1 / VOCAB).double().mean()):.1%}" for _, s in series]],
        ["p(alvo) mediana", *[f"{float(s.log_probs.exp().median()):.2e}" for _, s in series]],
        ["média geométrica de p(alvo) = 1/perplexidade",
         *[f"{math.exp(-s.loss):.2e}" for _, s in series]],
        [f"parte da perda nos piores {WORST_SHARE:.0%} dos tokens", *[piores(s.log_probs) for _, s in series]],
    ]
    report.table(["faixa de p(alvo)", *[nome for nome, _ in series]], linhas)

    # Que tokens ficam em cada ponta, na validação do modelo treinado.
    p = validacao_n.log_probs.exp()
    pontas = [
        ("p(alvo) ≥ 0,1", Counter(validacao_n.targets[p >= 0.1].tolist())),
        ("p(alvo) < 1e-4", Counter(validacao_n.targets[p < 1e-4].tolist())),
    ]
    report.write(
        f"**Que tokens ficam em cada ponta** (validação, fim do treino; os {AV5_TOP_TOKENS} alvos "
        "mais frequentes de cada grupo, com quantas vezes aparecem):"
    )
    report.write()
    colunas = [
        [f"{token_label(bpe, token)} ({vezes})" for token, vezes in contagem.most_common(AV5_TOP_TOKENS)]
        for _, contagem in pontas
    ]
    altura = max(len(coluna) for coluna in colunas)
    report.table(
        [f"{nome}: {sum(contagem.values())} tokens, {len(contagem)} distintos" for nome, contagem in pontas],
        [[coluna[i] if i < len(coluna) else "" for coluna in colunas] for i in range(altura)],
    )

    figura, eixo = new_axes(7.0, 3.6)
    bordas = torch.linspace(-12, 0, 49).tolist()
    for (nome, stats), cor in zip(series, (SERIES_COLORS[2], SERIES_COLORS[0], SERIES_COLORS[1])):
        valores = (stats.log_probs / math.log(10)).clamp(min=-12).tolist()
        eixo.hist(valores, bins=bordas, histtype="step", linewidth=1.5, color=cor, label=nome,
                  weights=[1 / len(valores)] * len(valores))
    eixo.axvline(math.log10(1 / VOCAB), color=SERIES_COLORS[4], linestyle=":", linewidth=1.0, label="1/V")
    eixo.set_xlabel("log10 p(alvo)")
    eixo.set_ylabel("fração dos tokens")
    eixo.set_title("Distribuição da probabilidade dada ao token certo", fontsize=10)
    legend(eixo, loc="upper left")
    report.figure(save_figure(figura, "avaliacao-probabilidade-do-alvo.png"),
                  "Distribuição de log10 p(alvo), antes e depois do treino")


def main() -> None:
    utf8_console()
    report = Report(RESULTS_PATH)
    bpe = BPETokenizer()
    texto = load_verdict()

    report.write("# Sprint 5 - Avaliação do modelo: resultados dos experimentos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint05/avaliacao_experimentos.py`. Não editar à mão: rode o script novamente."
    )
    report.write()
    report.write(
        f"Modelo: BASE_CONFIG ({thousands(GPTModel(MODEL_CONFIG).parameter_count())} parâmetros). "
        f"Dados: The Verdict, {TRAINING.train_ratio:.0%} treino / {1 - TRAINING.train_ratio:.0%} "
        f"validação nos caracteres, janelas de {MODEL_CONFIG.context_length} tokens sem "
        f"sobreposição, lotes de {TRAINING.batch_size}. Perdas em nats (log natural), "
        "médias por token. Treino em CPU, seeds fixas: rodar de novo reproduz todos os números."
    )
    report.write()

    train_loader, val_loader = create_train_val_loaders(
        texto, bpe, context_length=MODEL_CONFIG.context_length, batch_size=TRAINING.batch_size
    )
    report.write(
        f"Conjuntos: treino com {len(train_loader.dataset)} janelas ({len(train_loader)} lotes, "
        f"{thousands(len(train_loader.dataset.token_ids))} tokens); validação com "
        f"{len(val_loader.dataset)} janelas ({len(val_loader)} lotes, "
        f"{thousands(len(val_loader.dataset.token_ids))} tokens)."
    )
    report.write()

    torch.manual_seed(TRAINING.seed)
    sem_treino = GPTModel(MODEL_CONFIG).eval()
    experiment_1_loss_by_hand(report, sem_treino, val_loader, bpe)
    del sem_treino
    experiment_2_initial_state(report, train_loader, val_loader)
    por_epoca, _, por_posicao, janelas = experiment_3_evolution(report, texto, bpe)
    experiment_4_position_loss(report, por_posicao, janelas)
    experiment_5_target_probability(report, por_epoca, bpe)

    report.save()


if __name__ == "__main__":
    main()
