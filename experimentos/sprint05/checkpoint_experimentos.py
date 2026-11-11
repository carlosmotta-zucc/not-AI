"""Experimentos de salvamento e carregamento da Sprint 5 (Entregável 6).

C1  O que vai no arquivo: partes, tamanho e tempo
C2  Ida e volta: o modelo carregado é o mesmo e gera o mesmo texto
C3  Retomar o treino do arquivo: o que o otimizador e o gerador mudam
C4  O que o formato recusa, e o que um state_dict sozinho deixa passar

O treino é o de referência (BASE_CONFIG, BASE_TRAINING_CONFIG, seed 42), feito
em duas metades: 5 épocas, um arquivo com o treino pausado, mais 5 épocas com
os mesmos objetos. As 10 épocas são o T1 de resultados/treino.md, e o script
confere. O modelo final fica em

    checkpoints/sprint05-base-10-epocas.pth

com pesos, estado do AdamW, as duas configurações, o histórico e o estado do
gerador (331 MB; arquivos .pth não vão para o repositório). O C3 retoma o
arquivo da pausa de três maneiras e compara com o treino que não parou.

Leva ~6 minutos em CPU. Rodar de novo reproduz todos os números, exceto os
tempos.

Uso:
    python experimentos/sprint05/checkpoint_experimentos.py
    python experimentos/sprint05/checkpoint_experimentos.py --gerar    # só carrega o
        arquivo salvo e gera texto, sem treinar
    python experimentos/sprint05/checkpoint_experimentos.py --rapido   # versão curta,
        só para conferir que o script roda: tudo vai para uma pasta temporária
"""

import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import comum  # noqa: E402
from comum import (  # noqa: E402
    DEVICE,
    INK_MUTED,
    SEED,
    SERIES_COLORS,
    Report,
    compact,
    legend,
    loss_curves,
    new_axes,
    save_figure,
    thousands,
    utf8_console,
)
from src.corpus import load_verdict  # noqa: E402
from src.generation import generate_text  # noqa: E402
from src.model import BASE_CONFIG, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.training import (  # noqa: E402
    BASE_TRAINING_CONFIG,
    Checkpoint,
    TrainingHistory,
    create_optimizer,
    create_train_val_loaders,
    load_checkpoint,
    loader_loss,
    save_checkpoint,
    train_gpt,
    train_model,
)

RAPIDO = "--rapido" in sys.argv
GERAR = "--gerar" in sys.argv

MODEL_CONFIG = BASE_CONFIG
BASELINE = BASE_TRAINING_CONFIG if not RAPIDO else replace(BASE_TRAINING_CONFIG, num_epochs=2)
# Época em que o treino é pausado e salvo: a metade.
PAUSE_EPOCH = BASELINE.num_epochs // 2

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_NAME = f"sprint05-base-{BASE_TRAINING_CONFIG.num_epochs}-epocas.pth"

# Última avaliação do T1 em resultados/treino.md (treino, validação).
PUBLISHED_LOSSES = (3.429, 6.140)
LOSS_PRINT_TOLERANCE = 5e-4
# Desvio da validação entre seeds, medido no T1: a régua do C3.
SEED_NOISE = 0.027

PROMPT = BASELINE.start_context
NEW_TOKENS = 50
# O que o modelo carregado gera no C2 e no modo --gerar.
STRATEGIES = {
    "gulosa": {"temperature": 0.0},
    "top-k 50, T = 0,7": {"temperature": 0.7, "top_k": 50},
}
TIMING_REPEATS = 3
BYTES_PER_MEGABYTE = 1024 * 1024
# C4: o trecho da mensagem do weights_only que nomeia o objeto recusado, e o
# tamanho máximo de uma mensagem na tabela.
UNPICKLER_PREFIX = "WeightsUnpickler error: "
MESSAGE_LIMIT = 160


def megabytes(size: int) -> str:
    return f"{size / BYTES_PER_MEGABYTE:.1f} MB"


def tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def optimizer_tensors(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    """Os tensores do estado do AdamW: step, m e v de cada parâmetro."""
    return [
        valor
        for estado in optimizer.state_dict()["state"].values()
        for valor in estado.values()
        if torch.is_tensor(valor)
    ]


def error_summary(error: Exception) -> str:
    """A linha da mensagem que diz o que deu errado.

    O PyTorch abre as mensagens com um cabeçalho: no load_state_dict, o motivo
    vem na linha seguinte; no weights_only, depois de "WeightsUnpickler error:".
    """
    linhas = [linha.strip() for linha in str(error).splitlines() if linha.strip()]
    for linha in linhas:
        if linha.startswith(UNPICKLER_PREFIX):
            return linha.removeprefix(UNPICKLER_PREFIX).split(". Please", 1)[0]
    if len(linhas) > 1 and linhas[0].endswith(":"):
        return linhas[1]
    return linhas[0]


def max_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maior diferença absoluta entre dois tensores do mesmo shape. Em float:
    o state_dict tem as máscaras causais, que são booleanas e não subtraem."""
    return float((a.float() - b.float()).abs().max())


def median_seconds(function) -> float:
    tempos = []
    for _ in range(TIMING_REPEATS):
        inicio = time.perf_counter()
        function()
        tempos.append(time.perf_counter() - inicio)
    return statistics.median(tempos)


def generate_texts(model: GPTModel, bpe: BPETokenizer) -> dict[str, str]:
    """As duas gerações de STRATEGIES a partir de PROMPT, com o gerador semeado."""
    return {
        nome: generate_text(
            model, bpe, PROMPT, NEW_TOKENS, generator=torch.Generator().manual_seed(SEED), **opcoes
        )
        for nome, opcoes in STRATEGIES.items()
    }


def resume(checkpoint: Checkpoint, optimizer: torch.optim.Optimizer, text: str, bpe: BPETokenizer) -> None:
    """Treina as épocas que faltam para o num_epochs da configuração salva.

    A configuração guarda o treino planejado e o histórico, o que já foi feito:
    a diferença é o que falta. Os DataLoaders são refeitos da configuração;
    criá-los não sorteia nada.
    """
    cfg = checkpoint.train_cfg
    train_loader, val_loader = create_train_val_loaders(
        text,
        bpe,
        context_length=checkpoint.model_cfg.context_length,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        stride=cfg.stride,
    )
    train_model(
        checkpoint.model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=cfg.num_epochs - checkpoint.history.epochs_completed,
        eval_freq=cfg.eval_freq,
        eval_iter=cfg.eval_iter,
        tokenizer=bpe,
        start_context=cfg.start_context,
        sample_tokens=cfg.sample_tokens,
        history=checkpoint.history,
        log=None,
    )


def train_in_two_halves(text: str, bpe: BPETokenizer, pause_path: Path):
    """O treino de referência com uma pausa salva no meio.

    Returns:
        (run, segundos de treino). O arquivo da pausa fica em pause_path, com o
        otimizador e o estado do gerador daquele instante. Salvar não sorteia
        nada: as épocas seguintes saem iguais às de um treino sem o arquivo.
    """
    inicio = time.perf_counter()
    run = train_gpt(MODEL_CONFIG, replace(BASELINE, num_epochs=PAUSE_EPOCH), text, bpe, device=DEVICE, log=None)
    segundos = time.perf_counter() - inicio
    save_checkpoint(pause_path, run.model, BASELINE, run.history, run.optimizer, torch.get_rng_state())
    print(f"  {PAUSE_EPOCH} épocas e arquivo da pausa: {segundos:.0f} s")

    inicio = time.perf_counter()
    train_model(
        run.model,
        run.train_loader,
        run.val_loader,
        run.optimizer,
        num_epochs=BASELINE.num_epochs - PAUSE_EPOCH,
        eval_freq=BASELINE.eval_freq,
        eval_iter=BASELINE.eval_iter,
        tokenizer=bpe,
        start_context=BASELINE.start_context,
        sample_tokens=BASELINE.sample_tokens,
        history=run.history,
        log=None,
    )
    segundos += time.perf_counter() - inicio
    print(f"  {BASELINE.num_epochs} épocas: {segundos:.0f} s")
    return run, segundos


def experiment_1_contents(report: Report, run, final_path: Path, scratch: Path, training_seconds: float) -> None:
    """C1: as partes do arquivo, quanto pesam e quanto custa gravar e ler."""
    report.write("## C1 - O que vai no arquivo")
    report.write()
    report.write(
        "**Pergunta:** o que precisa ser salvo para o treino poder ser usado e "
        "continuado depois, quanto cada parte ocupa, e quanto custa gravar e ler?"
    )
    report.write()

    modelo, otimizador, historico = run.model, run.optimizer, run.history
    gerador = torch.get_rng_state()
    salvar_completo = median_seconds(
        lambda: save_checkpoint(final_path, modelo, BASELINE, historico, otimizador, gerador)
    )
    so_pesos = save_checkpoint(scratch / "so-pesos.pth", modelo, BASELINE, historico)

    conteudo = torch.load(final_path, weights_only=True)
    estado = conteudo["model_state_dict"]
    parametros = {nome for nome, _ in modelo.named_parameters()}
    buffers = [nome for nome in estado if nome not in parametros]
    momentos = optimizer_tensors(otimizador)

    report.write(
        f"**Configuração:** o modelo do treino de referência depois de {BASELINE.num_epochs} "
        f"épocas ({historico.global_step + 1} passos), salvo por `save_checkpoint` com o "
        "otimizador e o estado do gerador. Tamanho de cada parte = bytes dos tensores "
        "(float32, 4 bytes por número) ou do JSON equivalente."
    )
    report.write()
    report.table(
        ["chave", "conteúdo", "tamanho", "para quê"],
        [
            [
                "`model_state_dict`",
                f"{len(estado)} tensores: {thousands(modelo.parameter_count())} parâmetros em "
                f"{len(parametros)} tensores e {len(buffers)} buffers (as máscaras causais)",
                megabytes(tensor_bytes(estado.values())),
                "gerar texto e avaliar",
            ],
            [
                "`optimizer_state_dict`",
                f"{len(momentos)} tensores: passo, m e v de cada um dos {len(parametros)} tensores de parâmetros",
                megabytes(tensor_bytes(momentos)),
                "retomar o treino",
            ],
            [
                "`history`",
                f"{len(historico.steps)} avaliações, {len(historico.batch_losses)} perdas de lote, "
                f"{len(historico.samples)} amostras",
                f"{len(json.dumps(conteudo['history']).encode()) / 1024:.1f} kB",
                "curvas de perda e contagem de passos",
            ],
            [
                "`model_config`",
                f"{len(conteudo['model_config'])} campos do GPTConfig",
                f"{len(json.dumps(conteudo['model_config']).encode())} bytes",
                "remontar a arquitetura",
            ],
            [
                "`training_config`",
                f"{len(conteudo['training_config'])} campos do TrainingConfig",
                f"{len(json.dumps(conteudo['training_config']).encode())} bytes",
                "recriar o otimizador e os dados",
            ],
            [
                "`rng_state`",
                f"tensor de {conteudo['rng_state'].numel()} bytes",
                f"{conteudo['rng_state'].numel() / 1024:.1f} kB",
                "retomada idêntica (lotes e dropout)",
            ],
            [
                "`format_version`, `torch_version`",
                f"{conteudo['format_version']}, `{conteudo['torch_version']}`",
                "-",
                "recusar formato desconhecido; registro",
            ],
        ],
    )

    ler_completo = median_seconds(lambda: load_checkpoint(final_path))
    ler_pesos = median_seconds(lambda: load_checkpoint(so_pesos))
    salvar_pesos = median_seconds(lambda: save_checkpoint(so_pesos, modelo, BASELINE, historico))
    report.table(
        ["arquivo", "tamanho em disco", "salvar (s)", "carregar (s)"],
        [
            ["só pesos (sem otimizador nem gerador)", megabytes(so_pesos.stat().st_size), f"{salvar_pesos:.2f}", f"{ler_pesos:.2f}"],
            ["completo", megabytes(final_path.stat().st_size), f"{salvar_completo:.2f}", f"{ler_completo:.2f}"],
        ],
    )
    report.write(
        f"Treinar as {BASELINE.num_epochs} épocas levou {training_seconds:.0f} s nesta máquina; "
        f"carregar o arquivo completo, {ler_completo:.2f} s: {training_seconds / ler_completo:.0f} "
        f"vezes menos. Tempos: mediana de {TIMING_REPEATS} execuções, com o arquivo já no "
        "cache do sistema."
    )
    report.write()


def experiment_2_round_trip(report: Report, run, final_path: Path, bpe: BPETokenizer, gerar_command: list[str]) -> None:
    """C2: o que sai do arquivo é o que entrou, e gera o mesmo texto."""
    report.write("## C2 - Ida e volta: o modelo carregado gerando texto")
    report.write()
    report.write(
        "**Pergunta:** o modelo reconstruído do arquivo é o mesmo que estava na "
        "memória? Ele gera o mesmo texto, inclusive num processo novo?"
    )
    report.write()
    report.write(
        "**Configuração:** `load_checkpoint` sobre o arquivo do C1, num `GPTModel` novo. "
        "Comparação com o modelo que ficou na memória depois do treino: pesos, perdas sobre "
        f"os conjuntos inteiros e {NEW_TOKENS} tokens gerados a partir de `{PROMPT}`, com a "
        f"decodificação gulosa e com top-k 50 e T = 0,7 (gerador com seed {SEED}). Por fim, o "
        "mesmo arquivo carregado por outro processo do Python (`--gerar`)."
    )
    report.write()

    carregado = load_checkpoint(final_path)
    original = run.model.state_dict()
    lido = carregado.model.state_dict()
    maior_diferenca = max(max_difference(original[nome], lido[nome]) for nome in original)

    def perdas_de(modelo: GPTModel) -> tuple[float, float]:
        # A mesma seed antes de cada medida: o loader de treino embaralha, e a
        # soma das perdas dos lotes em outra ordem muda as últimas casas.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            return loader_loss(modelo, run.train_loader), loader_loss(modelo, run.val_loader)

    perdas = {"na memória": perdas_de(run.model), "carregado": perdas_de(carregado.model)}
    textos = {
        "na memória": generate_texts(run.model, bpe),
        "carregado": generate_texts(carregado.model, bpe),
    }

    saida = subprocess.run(gerar_command, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    outro_processo = dict(linha.split(": ", 1) for linha in saida.splitlines() if ": " in linha)

    def igual(condicao: bool) -> str:
        return "igual" if condicao else "DIFERENTE"

    report.table(
        ["o que se compara", "na memória", "carregado do arquivo", "resultado"],
        [
            ["tensores do state_dict", str(len(original)), str(len(lido)), igual(original.keys() == lido.keys())],
            ["maior diferença entre pesos", "-", f"{maior_diferenca:.1e}", igual(maior_diferenca == 0.0)],
            [
                "perda de treino",
                f"{perdas['na memória'][0]:.6f}",
                f"{perdas['carregado'][0]:.6f}",
                igual(perdas["na memória"][0] == perdas["carregado"][0]),
            ],
            [
                "perda de validação",
                f"{perdas['na memória'][1]:.6f}",
                f"{perdas['carregado'][1]:.6f}",
                igual(perdas["na memória"][1] == perdas["carregado"][1]),
            ],
            ["GPTConfig", "BASE_CONFIG", "do arquivo", igual(carregado.model_cfg == MODEL_CONFIG)],
            ["TrainingConfig", "BASE_TRAINING_CONFIG", "do arquivo", igual(carregado.train_cfg == BASELINE)],
            [
                "histórico",
                f"{len(run.history.steps)} avaliações",
                f"{len(carregado.history.steps)} avaliações",
                igual(carregado.history == run.history),
            ],
            [
                "modo do modelo",
                "train() (como o treino deixa)",
                "eval()" if not carregado.model.training else "train()",
                "-",
            ],
        ],
    )

    report.write("**O modelo carregado gerando texto:**")
    report.write()
    linhas = []
    for nome in STRATEGIES:
        memoria, arquivo = textos["na memória"][nome], textos["carregado"][nome]
        processo = outro_processo.get(nome, "")
        linhas.append(
            [
                nome,
                f"`{compact(arquivo)}`",
                igual(arquivo == memoria),
                igual(compact(arquivo) == processo),
            ]
        )
    report.table(
        ["estratégia", "texto do modelo carregado", "contra o da memória", "contra o de outro processo"],
        linhas,
    )


def experiment_3_resume(report: Report, run, pause_path: Path, text: str, bpe: BPETokenizer) -> None:
    """C3: o que cada parte do arquivo muda na retomada."""
    report.write("## C3 - Retomar o treino do arquivo")
    report.write()
    report.write(
        "**Pergunta:** um treino pausado, salvo e retomado é o mesmo treino? O que "
        "muda se o estado do otimizador, ou o do gerador, não for restaurado?"
    )
    report.write()
    restantes = BASELINE.num_epochs - PAUSE_EPOCH
    report.write(
        f"**Configuração:** o treino de referência é salvo no fim da época {PAUSE_EPOCH} e segue até a "
        f"{BASELINE.num_epochs} sem parar (é o T1). O arquivo da pausa é então carregado três vezes, cada "
        f"uma num modelo novo, e treinado pelas {restantes} épocas que faltam:"
    )
    report.write()
    report.write("- **otimizador + gerador:** `load_checkpoint(..., restore_rng=True)` e o AdamW que veio do arquivo;")
    report.write(
        "- **sem o estado do otimizador:** os mesmos pesos e o mesmo gerador, com um AdamW "
        "novo (`create_optimizer`): m e v recomeçam do zero;"
    )
    report.write(
        f"- **sem o estado do gerador:** pesos e AdamW do arquivo, e `torch.manual_seed({BASELINE.seed})` "
        "no lugar do estado salvo -- o que faria quem guardou só a seed."
    )
    report.write()

    referencia = run.history
    pausa_avaliacoes = sum(1 for epoca in referencia.epochs if epoca <= PAUSE_EPOCH)
    pausa_passos = PAUSE_EPOCH * len(run.train_loader)

    def retomar(restaurar_gerador: bool, otimizador_do_arquivo: bool) -> Checkpoint:
        checkpoint = load_checkpoint(pause_path, device=DEVICE, restore_rng=restaurar_gerador)
        otimizador = checkpoint.optimizer if otimizador_do_arquivo else create_optimizer(checkpoint.model, checkpoint.train_cfg)
        if not restaurar_gerador:
            torch.manual_seed(checkpoint.train_cfg.seed)
        inicio = time.perf_counter()
        resume(checkpoint, otimizador, text, bpe)
        print(f"    retomada: {time.perf_counter() - inicio:.0f} s")
        return checkpoint

    variantes = {
        "otimizador + gerador": retomar(True, True),
        "sem o estado do otimizador": retomar(True, False),
        "sem o estado do gerador": retomar(False, True),
    }

    def linha(nome: str, historico: TrainingHistory, modelo: GPTModel | None) -> list[str]:
        depois = slice(pausa_avaliacoes, None)
        if modelo is None:
            pesos = perdas = "-"
        else:
            pesos = f"{max(max_difference(a, b) for a, b in zip(modelo.parameters(), run.model.parameters())):.1e}"
            perdas = f"{max(abs(a - b) for a, b in zip(historico.val_losses[depois], referencia.val_losses[depois])):.3f}"
        return [
            nome,
            f"{historico.batch_losses[pausa_passos]:.4f}",
            f"{historico.batch_losses[pausa_passos + 1]:.4f}",
            f"{min(historico.val_losses[depois]):.3f}",
            f"{historico.train_losses[-1]:.3f}",
            f"{historico.val_losses[-1]:.3f}",
            perdas,
            pesos,
        ]

    report.table(
        [
            "retomada",
            "perda do 1º lote depois da pausa",
            "perda do 2º lote",
            "validação mínima depois da pausa",
            "treino final",
            "validação final",
            "maior diferença de validação para o treino sem pausa",
            "maior diferença entre pesos",
        ],
        [linha("sem pausa (T1)", referencia, None)]
        + [linha(nome, c.history, c.model) for nome, c in variantes.items()],
    )
    report.write(
        f"Régua: entre seeds diferentes, a validação do treino de referência varia {SEED_NOISE} "
        "(desvio-padrão, T1). O histórico da retomada segue a contagem do arquivo: "
        f"{variantes['otimizador + gerador'].history.global_step + 1} passos e "
        f"{variantes['otimizador + gerador'].history.epochs_completed} épocas no fim, como no treino sem pausa."
    )
    report.write()

    amostras = [["sem pausa (T1)", f"`{compact(referencia.samples[-1])}`"]] + [
        [nome, f"`{compact(c.history.samples[-1])}`"] for nome, c in variantes.items()
    ]
    report.write(f"Amostra gulosa do fim da época {BASELINE.num_epochs}:")
    report.write()
    report.table(["retomada", "amostra"], amostras)

    figura, eixo = new_axes(7.0, 3.8)
    passos = len(run.train_loader)
    loss_curves(eixo, referencia, passos, INK_MUTED, label="sem pausa")
    desenhadas = [referencia]
    for (nome, checkpoint), cor in zip(list(variantes.items())[1:], SERIES_COLORS):
        loss_curves(eixo, checkpoint.history, passos, cor, label=nome)
        desenhadas.append(checkpoint.history)
    eixo.axvline(PAUSE_EPOCH, color=INK_MUTED, linestyle=":", linewidth=1.0)

    # Só a época anterior à pausa e as retomadas: antes disso as curvas são a mesma.
    primeira = sum(1 for epoca in referencia.epochs if epoca < PAUSE_EPOCH)
    visiveis = [
        perda
        for historico in desenhadas
        for perda in historico.train_losses[primeira:] + historico.val_losses[primeira:]
    ]
    faixa = max(visiveis) - min(visiveis)
    eixo.set_xlim(PAUSE_EPOCH - 1, BASELINE.num_epochs)
    # Folga em cima para a legenda não cobrir as curvas de validação.
    eixo.set_ylim(min(visiveis) - 0.05 * faixa, max(visiveis) + 0.45 * faixa)
    eixo.set_title(f"Retomada na época {PAUSE_EPOCH}: o que falta no arquivo aparece nas curvas", fontsize=10)
    legend(eixo, loc="upper center", ncols=2)
    report.figure(
        save_figure(figura, "checkpoint-retomada.png"),
        "Perdas das retomadas; a retomada com otimizador e gerador coincide com a curva sem pausa",
    )


def experiment_4_refusals(report: Report, run, final_path: Path, scratch: Path) -> None:
    """C4: os erros que o formato acusa, e o que um state_dict sozinho não acusa."""
    report.write("## C4 - O que o formato recusa")
    report.write()
    report.write(
        "**Pergunta:** que enganos ao carregar pesos viram erro, e quais passariam "
        "em silêncio se o arquivo tivesse só o `state_dict`, como no livro?"
    )
    report.write()
    report.write(
        "**Configuração:** cada linha tenta uma carga. As quatro primeiras fazem como o "
        "livro: montam um `GPTModel` com uma configuração escrita à mão e chamam "
        "`load_state_dict` com os pesos do arquivo. As outras usam `load_checkpoint`."
    )
    report.write()

    estado = torch.load(final_path, weights_only=True)["model_state_dict"]
    so_pesos = scratch / "so-pesos.pth"

    def a_moda_do_livro(**mudancas):
        return lambda: GPTModel(replace(MODEL_CONFIG, **mudancas)).load_state_dict(estado)

    def arquivo_com(conteudo) -> Path:
        caminho = scratch / "recusa.pth"
        torch.save(conteudo, caminho)
        return caminho

    valido = torch.load(final_path, weights_only=True)
    casos = [
        ("`load_state_dict` num modelo com `emb_dim` 128 em vez de 256", a_moda_do_livro(emb_dim=128)),
        ("`load_state_dict` num modelo com 2 blocos em vez de 4", a_moda_do_livro(n_layers=2)),
        ("`load_state_dict` num modelo com `context_length` 256 em vez de 128", a_moda_do_livro(context_length=256)),
        ("`load_state_dict` num modelo com `drop_rate` 0,5 em vez de 0,1", a_moda_do_livro(drop_rate=0.5)),
        ("`load_checkpoint` de um arquivo só com o `state_dict`", lambda: load_checkpoint(arquivo_com(estado))),
        (
            "`load_checkpoint` de um arquivo com `format_version` 2",
            lambda: load_checkpoint(arquivo_com({**valido, "format_version": 2})),
        ),
        (
            "`load_checkpoint(..., restore_rng=True)` de um arquivo salvo sem o gerador",
            lambda: load_checkpoint(so_pesos, restore_rng=True),
        ),
        (
            "`load_checkpoint` de um arquivo com um objeto que não é tensor",
            lambda: load_checkpoint(arquivo_com({**valido, "extra": comum.Report(scratch / "x.md")})),
        ),
    ]

    linhas = []
    for descricao, tentativa in casos:
        with torch.random.fork_rng(devices=[]):
            try:
                tentativa()
            except Exception as erro:  # noqa: BLE001 - o experimento é registrar o erro
                mensagem = error_summary(erro).replace(f"{scratch}/", "")
                linhas.append([descricao, type(erro).__name__, f"`{compact(mensagem, MESSAGE_LIMIT)}`"])
            else:
                linhas.append([descricao, "nenhum erro", "carrega e segue"])
    report.table(["tentativa", "erro", "mensagem (resumo)"], linhas)


def print_generation(path: Path) -> int:
    """Modo --gerar: carrega o arquivo e gera texto, sem treinar."""
    if not path.exists():
        print(f"{path} não existe. Rode o script sem --gerar para treinar e salvar o modelo.")
        return 1

    checkpoint = load_checkpoint(path)
    historico = checkpoint.history
    print(
        f"{path.name}: {thousands(checkpoint.model.parameter_count())} parâmetros, "
        f"{historico.epochs_completed} épocas, última avaliação com treino "
        f"{historico.train_losses[-1]:.3f} e validação {historico.val_losses[-1]:.3f}"
    )
    for nome, texto in generate_texts(checkpoint.model, BPETokenizer()).items():
        print(f"{nome}: {compact(texto)}")
    return 0


def main() -> int:
    utf8_console()
    if GERAR:
        destino = sys.argv[sys.argv.index("--gerar") + 1 :]
        return print_generation(Path(destino[0]) if destino else CHECKPOINT_DIR / CHECKPOINT_NAME)

    # Os arquivos intermediários (a pausa, o só de pesos, os recusados) somam
    # ~800 MB: ficam numa pasta que é apagada no fim.
    with tempfile.TemporaryDirectory(prefix="sprint05-checkpoint-") as pasta:
        return run_experiments(Path(pasta))


def run_experiments(scratch: Path) -> int:
    if RAPIDO:
        destino = Path(tempfile.mkdtemp(prefix="sprint05-checkpoint-rapido-"))
        comum.FIGURES_DIR = destino / "figuras"
        report = Report(destino / "checkpoint.md")
        final_path = scratch / CHECKPOINT_NAME
        print(f"modo rápido: resultados em {destino}")
    else:
        report = Report(comum.RESULTS_DIR / "checkpoint.md")
        final_path = CHECKPOINT_DIR / CHECKPOINT_NAME
    pause_path = scratch / "pausa.pth"
    bpe = BPETokenizer()
    texto = load_verdict()

    report.write("# Sprint 5 - Salvamento e carregamento: resultados dos experimentos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint05/checkpoint_experimentos.py`. Não editar à mão: rode o script novamente."
    )
    report.write()
    report.write(
        f"Modelo: BASE_CONFIG, treinado com BASE_TRAINING_CONFIG (seed {BASELINE.seed}) por "
        f"{BASELINE.num_epochs} épocas, em duas metades com uma pausa salva na época {PAUSE_EPOCH}. "
        "Arquivos gravados e lidos por `save_checkpoint` e `load_checkpoint` "
        "(`src/training/checkpoint.py`). O modelo final fica em "
        f"`checkpoints/{CHECKPOINT_NAME}`, fora do repositório (`*.pth` no `.gitignore`); os "
        "arquivos intermediários, numa pasta temporária. Treino em CPU, seeds fixas: rodar de "
        "novo reproduz tudo, exceto os tempos. Figuras em `resultados/figuras/`."
    )
    report.write()

    print("treino")
    run, segundos = train_in_two_halves(texto, bpe, pause_path)
    historico = run.history
    if not RAPIDO:
        treino, validacao = PUBLISHED_LOSSES
        confere = (
            abs(historico.train_losses[-1] - treino) < LOSS_PRINT_TOLERANCE
            and abs(historico.val_losses[-1] - validacao) < LOSS_PRINT_TOLERANCE
        )
        report.write(
            f"Conferência com `resultados/treino.md`: última avaliação (passo {historico.steps[-1]}) com "
            f"treino {historico.train_losses[-1]:.3f} e validação {historico.val_losses[-1]:.3f}; o T1 "
            f"publica {treino:.3f} e {validacao:.3f}: **{'confere' if confere else 'NÃO confere'}**. "
            "Salvar no meio não mudou o treino."
        )
        report.write()

    gerar = [sys.executable, str(Path(__file__).resolve()), "--gerar", str(final_path)]
    for nome, experimento in (
        ("C1", lambda: experiment_1_contents(report, run, final_path, scratch, segundos)),
        ("C2", lambda: experiment_2_round_trip(report, run, final_path, bpe, gerar)),
        ("C3", lambda: experiment_3_resume(report, run, pause_path, texto, bpe)),
        ("C4", lambda: experiment_4_refusals(report, run, final_path, scratch)),
    ):
        print(nome)
        experimento()
        report.save()

    print(f"\nResultados salvos em {report.path}")
    if not RAPIDO:
        print(f"Modelo salvo em {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
