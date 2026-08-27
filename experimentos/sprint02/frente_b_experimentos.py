"""Experimentos da Frente B da Sprint 2: do Token ID ao lote.

Cinco experimentos, cada um com pergunta declarada, configuracao explicita e
resultado numerico. A interpretacao de cada um esta em
docs/sprint02/frente-b-analise.md -- aqui ficam so os numeros que ela cita.

E1  Contexto e stride contra quantidade de amostras
E2  Batch size contra shape e numero de batches
E3  Dimensao do embedding contra custo
E4  Efeito da posicao
E5  Dimensoes ao longo do pipeline

Diferenca em relacao a Frente A: aqui HA aleatoriedade. As tabelas de
embeddings nascem de sorteio e o DataLoader pode embaralhar as janelas. Por
isso a seed e fixada em 42 (a mesma de check_environment.py) no inicio e de
novo antes de cada bloco que sorteia -- sem isso nenhuma tabela deste
relatorio seria reproduzivel. A unica fonte de variacao que sobra e o tempo
de forward medido no E3: duas execucoes podem divergir por ruido de
agendamento do sistema operacional, os parametros e shapes nao.

Uso:
    python experimentos/sprint02/frente_b_experimentos.py
"""

import sys
import time
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus import load_verdict  # noqa: E402
from src.dataset import GPTDataset, create_dataloader, expected_sample_count  # noqa: E402
from src.embeddings import InputEmbedding, PositionalEmbedding, TokenEmbedding  # noqa: E402
from src.tokenizer import BPETokenizer, Vocabulary  # noqa: E402

RESULTS_PATH = Path(__file__).parent / "resultados" / "frente-b.md"

BASE_CONFIG = {
    "context_length": 128,
    "stride": 128,
    "batch_size": 8,
    "emb_dim": 256,
    "seed": 42,
}

# E1: contextos crescendo por ordem de grandeza, cada um com tres strides que
# representam os tres regimes -- maxima sobreposicao, sobreposicao parcial e
# nenhuma sobreposicao.
CONTEXT_SWEEP = (8, 16, 32, 64, 128, 256)

# E2: fixa o contexto de BASE_CONFIG, varia so o batch_size.
BATCH_SIZE_SWEEP = (1, 2, 4, 8, 16, 32)

# E3: 768 e a dimensao do GPT-2 small -- fica na varredura para comparacao
# direta com o modelo real, nao so com os tamanhos didaticos menores.
EMB_DIM_SWEEP = (16, 32, 64, 128, 256, 768)

BYTES_PER_MEGABYTE = 1024 * 1024


def thousands(value: int) -> str:
    """Formata inteiro com espaco como separador de milhar (padrao pt-BR)."""
    return f"{value:,}".replace(",", " ")


def megabytes(tensor_elements: int, bytes_per_element: int = 4) -> str:
    """Converte contagem de elementos em tamanho legivel, assumindo float32."""
    return f"{tensor_elements * bytes_per_element / BYTES_PER_MEGABYTE:.1f} MB"


class Report:
    """Acumula o relatorio em markdown enquanto imprime no terminal."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def write(self, line: str = "") -> None:
        self._lines.append(line)
        print(line)

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Escreve uma tabela markdown com colunas alinhadas no terminal."""
        widths = [
            max(len(headers[column]), *(len(row[column]) for row in rows))
            for column in range(len(headers))
        ]

        self.write("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
        self.write("| " + " | ".join("-" * w for w in widths) + " |")
        for row in rows:
            self.write("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
        self.write()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")


def experiment_1_context_and_stride(report: Report, corpus: str, bpe: BPETokenizer) -> None:
    """E1: como contexto e stride, juntos, determinam o numero de amostras."""
    report.write("## E1 - Contexto e stride contra quantidade de amostras")
    report.write()
    report.write(
        "**Pergunta:** o numero de amostras de treino cresce so com o "
        "contexto, ou o stride pesa tanto quanto? E quanto de sobreposicao "
        "cada combinacao paga?"
    )
    report.write()
    report.write(
        f"**Configuracao:** The Verdict tokenizado pelo BPE, "
        f"context_length em {', '.join(str(c) for c in CONTEXT_SWEEP)}, "
        f"cada um com stride em {{1, context_length/2, context_length}}."
    )
    report.write()

    total_tokens = len(bpe.encode(corpus))
    report.write(f"O corpus tem {thousands(total_tokens)} Token IDs pelo BPE.")
    report.write()

    rows = []
    for context_length in CONTEXT_SWEEP:
        strides = list(dict.fromkeys((1, context_length // 2, context_length)))

        for stride in strides:
            samples = expected_sample_count(total_tokens, context_length, stride)
            tokens_per_epoch = samples * context_length
            overlap = max(context_length - stride, 0)

            rows.append(
                [
                    str(context_length),
                    str(stride),
                    thousands(samples),
                    thousands(tokens_per_epoch),
                    f"{overlap / context_length:.2f}",
                ]
            )

    report.table(
        [
            "context_length",
            "stride",
            "amostras",
            "tokens vistos por epoca",
            "fracao de sobreposicao",
        ],
        rows,
    )


def experiment_2_batch_size(report: Report, corpus: str, bpe: BPETokenizer) -> None:
    """E2: como o batch_size reparte as amostras de uma epoca em lotes."""
    report.write("## E2 - Batch size contra shape e numero de batches")
    report.write()
    report.write(
        "**Pergunta:** quantos lotes uma epoca produz para cada batch_size, "
        "e quantas amostras drop_last descarta em cada caso?"
    )
    report.write()

    context_length = BASE_CONFIG["context_length"]
    stride = BASE_CONFIG["stride"]

    report.write(
        f"**Configuracao:** The Verdict tokenizado pelo BPE, "
        f"context_length={context_length}, stride={stride} (fixos), "
        f"batch_size em {', '.join(str(b) for b in BATCH_SIZE_SWEEP)}, "
        f"drop_last=True."
    )
    report.write()

    total_tokens = len(bpe.encode(corpus))
    total_samples = expected_sample_count(total_tokens, context_length, stride)
    report.write(f"Amostras na epoca (independe de batch_size): {thousands(total_samples)}.")
    report.write()

    rows = []
    for batch_size in BATCH_SIZE_SWEEP:
        batches_per_epoch = total_samples // batch_size
        discarded = total_samples % batch_size

        rows.append(
            [
                str(batch_size),
                thousands(batches_per_epoch),
                f"({batch_size}, {context_length})",
                thousands(discarded),
            ]
        )

    report.table(
        ["batch_size", "batches por epoca", "shape do batch", "amostras descartadas"],
        rows,
    )

    # Confere a tabela contra um DataLoader real, nao so contra a formula.
    smallest_batch = min(BATCH_SIZE_SWEEP)
    loader = create_dataloader(
        corpus, bpe, context_length=context_length, stride=stride,
        batch_size=smallest_batch, shuffle=False, drop_last=True,
    )
    real_entrada, _ = next(iter(loader))
    report.write(
        f"Shape do primeiro lote real com batch_size={smallest_batch}: "
        f"{tuple(real_entrada.shape)} -- confere com a tabela."
    )
    report.write()


def experiment_3_embedding_dimension(report: Report, corpus: str, bpe: BPETokenizer) -> None:
    """E3: o que muda, em custo, ao variar emb_dim para os dois vocabularios."""
    report.write("## E3 - Dimensao do embedding contra custo")
    report.write()
    report.write(
        "**Pergunta:** quanto custa, em parametros, memoria e tempo de "
        "forward, cada aumento de emb_dim? O custo depende mais da dimensao "
        "ou do tamanho do vocabulario?"
    )
    report.write()

    context_length = BASE_CONFIG["context_length"]
    batch_size = BASE_CONFIG["batch_size"]
    own_vocabulary = Vocabulary.from_texts([corpus])

    report.write(
        f"**Configuracao:** emb_dim em {', '.join(str(d) for d in EMB_DIM_SWEEP)} "
        f"(768 e a dimensao do GPT-2 small, incluida para comparacao), "
        f"vocabulario do BPE ({thousands(len(bpe))} entradas) e vocabulario "
        f"proprio ({thousands(len(own_vocabulary))} entradas), "
        f"context_length={context_length}, forward sobre lote "
        f"({batch_size}, {context_length}), seed {BASE_CONFIG['seed']}."
    )
    report.write()

    def sweep(vocab_size: int) -> list[list[str]]:
        rows = []
        for emb_dim in EMB_DIM_SWEEP:
            torch.manual_seed(BASE_CONFIG["seed"])
            layer = InputEmbedding(vocab_size, emb_dim, context_length)
            token_params = layer.token_embedding.parameter_count()
            positional_params = layer.positional_embedding.weight.numel()

            batch = torch.randint(0, vocab_size, (batch_size, context_length))

            start = time.perf_counter()
            with torch.no_grad():
                layer(batch)
            forward_seconds = time.perf_counter() - start

            rows.append(
                [
                    str(emb_dim),
                    thousands(vocab_size),
                    thousands(token_params),
                    thousands(positional_params),
                    megabytes(token_params + positional_params),
                    f"{forward_seconds * 1000:.2f} ms",
                ]
            )
        return rows

    headers = [
        "emb_dim",
        "vocab_size",
        "parametros (tokens)",
        "parametros (posicoes)",
        "memoria estimada",
        "tempo de um forward",
    ]

    report.write("Vocabulario do BPE:")
    report.write()
    report.table(headers, sweep(len(bpe)))

    report.write("Vocabulario proprio:")
    report.write()
    report.table(headers, sweep(len(own_vocabulary)))


def experiment_4_position_effect(report: Report, corpus: str, bpe: BPETokenizer) -> None:
    """E4: o mesmo Token ID, em posicoes diferentes, para de ter o mesmo vetor."""
    report.write("## E4 - Efeito da posicao")
    report.write()
    report.write(
        "**Pergunta:** duas ocorrencias do mesmo Token ID tem o mesmo vetor "
        "de token? E o mesmo vetor final, depois da soma posicional?"
    )
    report.write()

    context_length = BASE_CONFIG["context_length"]
    emb_dim = BASE_CONFIG["emb_dim"]

    token_ids = bpe.encode(corpus)[:context_length]
    counts = Counter(token_ids)
    repeated_id = next(token_id for token_id, count in counts.items() if count >= 2)
    occurrences = [index for index, token_id in enumerate(token_ids) if token_id == repeated_id]
    first_position, second_position = occurrences[:2]

    report.write(
        f"**Configuracao:** primeiros {context_length} Token IDs de The "
        f"Verdict pelo BPE, emb_dim={emb_dim}, seed {BASE_CONFIG['seed']}. "
        f"Token ID {repeated_id} (`{bpe.decode([repeated_id])!r}`) se repete "
        f"nas posicoes {first_position} e {second_position}."
    )
    report.write()

    torch.manual_seed(BASE_CONFIG["seed"])
    layer = InputEmbedding(len(bpe), emb_dim, context_length)

    ids_tensor = torch.tensor([token_ids])
    token_vectors = layer.token_embedding(ids_tensor)[0]
    input_vectors = layer(ids_tensor)[0]

    token_distance = torch.dist(
        token_vectors[first_position], token_vectors[second_position]
    ).item()
    final_distance = torch.dist(
        input_vectors[first_position], input_vectors[second_position]
    ).item()

    report.table(
        ["vetor", "distancia entre as duas ocorrencias"],
        [
            ["token embedding (antes da posicao)", f"{token_distance:.6f}"],
            ["input embedding (depois da posicao)", f"{final_distance:.6f}"],
        ],
    )


def experiment_5_pipeline_shapes(report: Report, corpus: str, bpe: BPETokenizer) -> None:
    """E5: o shape em cada etapa, do texto bruto ate a entrada da Sprint 3."""
    report.write("## E5 - Dimensoes ao longo do pipeline")
    report.write()
    report.write(
        "**Pergunta:** os shapes fecham do texto bruto ate o tensor de "
        "entrada do modelo, para uma configuracao de referencia?"
    )
    report.write()
    report.write(
        "**Configuracao:** " + ", ".join(f"{key}={value}" for key, value in BASE_CONFIG.items())
    )
    report.write()

    context_length = BASE_CONFIG["context_length"]
    stride = BASE_CONFIG["stride"]
    batch_size = BASE_CONFIG["batch_size"]
    emb_dim = BASE_CONFIG["emb_dim"]

    token_ids = bpe.encode(corpus)
    dataset = GPTDataset(corpus, bpe, context_length=context_length, stride=stride)
    sample_input, sample_target = dataset[0]

    loader = create_dataloader(
        corpus, bpe, context_length=context_length, stride=stride,
        batch_size=batch_size, shuffle=False, drop_last=True,
    )
    batch_input, _ = next(iter(loader))

    torch.manual_seed(BASE_CONFIG["seed"])
    layer = InputEmbedding(len(bpe), emb_dim, context_length)

    token_vectors = layer.token_embedding(batch_input)
    position_vectors = layer.positional_embedding(context_length)
    input_vectors = layer(batch_input)

    report.table(
        ["etapa", "objeto", "shape"],
        [
            ["texto bruto", "str", f"{thousands(len(corpus))} caracteres"],
            ["tokenizacao (BPE)", "list[int]", f"{thousands(len(token_ids))} Token IDs"],
            ["janela deslizante", "GPTDataset", f"{thousands(len(dataset))} amostras"],
            ["uma amostra", "(x, y)", f"{tuple(sample_input.shape)} cada"],
            ["lote", "Tensor long", str(tuple(batch_input.shape))],
            ["token embedding", "Tensor float32", str(tuple(token_vectors.shape))],
            ["positional embedding", "Tensor float32", str(tuple(position_vectors.shape))],
            [
                "input embedding (entrada da Sprint 3)",
                "Tensor float32",
                str(tuple(input_vectors.shape)),
            ],
        ],
    )

    report.write(f"Parametros treinaveis do bloco de embeddings: {thousands(layer.parameter_count())}.")
    report.write()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(BASE_CONFIG["seed"])

    corpus = load_verdict()
    bpe = BPETokenizer()

    report = Report()
    report.write("# Sprint 2 - Frente B: resultados dos experimentos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint02/frente_b_experimentos.py`. "
        "Nao editar a mao: rode o script novamente."
    )
    report.write()
    report.write(f"Corpus: The Verdict, {thousands(len(corpus))} caracteres.")
    report.write()
    report.write(
        f"Seed fixa em {BASE_CONFIG['seed']} no inicio e antes de cada bloco "
        "que sorteia (tabelas de embeddings) -- sem isso nenhuma tabela "
        "deste relatorio seria reproduzivel."
    )
    report.write()

    experiment_1_context_and_stride(report, corpus, bpe)
    experiment_2_batch_size(report, corpus, bpe)
    experiment_3_embedding_dimension(report, corpus, bpe)
    experiment_4_position_effect(report, corpus, bpe)
    experiment_5_pipeline_shapes(report, corpus, bpe)

    report.save(RESULTS_PATH)
    print(f"\nResultados salvos em {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
