"""Experimentos da Frente A da Sprint 2: do texto ao Token ID.

Quatro experimentos, cada um com pergunta declarada, configuracao explicita e
resultado numerico. A interpretacao de cada um esta em
docs/sprint02/frente-a-analise.md -- aqui ficam so os numeros que ela cita.

E1  Contagem de tokens em textos de tamanhos diferentes
E2  Comportamento diante de palavra fora do vocabulario
E3  Portugues contra ingles no mesmo tokenizador
E4  Comparativo consolidado: tokenizador proprio contra BPE

Nao ha aleatoriedade em nenhum deles: as duas execucoes de um mesmo corpus
produzem exatamente a mesma tabela, entao nao ha seed a fixar. A unica fonte
de variacao e o tempo medido no E4.

Uso:
    python experimentos/sprint02/frente_a_experimentos.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus import load_verdict  # noqa: E402
from src.tokenizer import (  # noqa: E402
    UNKNOWN_TOKEN,
    BPETokenizer,
    SimpleTokenizer,
    Vocabulary,
    split_text,
    unique_tokens,
)

RESULTS_PATH = Path(__file__).parent / "resultados" / "frente-a.md"

# E1: prefixos do corpus, em caracteres. Crescem por ordem de grandeza para
# que a curva de vocabulario apareca mesmo com corpus pequeno.
PREFIX_SIZES = (500, 1_000, 2_500, 5_000, 10_000, 20_479)

# E2: frases construidas para forcar situacoes distintas de palavra ausente.
OUT_OF_VOCABULARY_CASES = (
    ("palavra comum ausente do conto", "Hello, do you like tea?"),
    ("nome proprio inexistente", "Akwirw ier"),
    ("termo tecnico moderno", "The transformer architecture uses attention."),
    ("duas palavras novas distintas", "quantum blockchain"),
)

# E3: pares equivalentes em conteudo, para que a diferenca medida seja do
# tokenizador e nao do que cada frase diz.
PARALLEL_SENTENCES = (
    (
        "O gato preto dormia tranquilamente sobre o telhado quente.",
        "The black cat slept peacefully on the warm roof.",
    ),
    (
        "A implementação do modelo exigiu atenção aos detalhes.",
        "The model implementation required attention to detail.",
    ),
    (
        "Não podemos afirmar que a conclusão está correta.",
        "We cannot claim that the conclusion is correct.",
    ),
    (
        "As informações são processadas simultaneamente pela rede.",
        "The information is processed simultaneously by the network.",
    ),
    (
        "Ele explicou que a manutenção seria concluída amanhã.",
        "He explained that the maintenance would be finished tomorrow.",
    ),
)


def thousands(value: int) -> str:
    """Formata inteiro com espaco como separador de milhar (padrao pt-BR)."""
    return f"{value:,}".replace(",", " ")


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


def experiment_1_token_counts(report: Report, corpus: str, bpe: BPETokenizer) -> None:
    """E1: como a contagem de tokens acompanha o tamanho do texto."""
    report.write("## E1 - Contagem de tokens em textos de tamanhos diferentes")
    report.write()
    report.write(
        "**Pergunta:** a contagem de tokens e propriedade do texto ou do par "
        "texto + tokenizador? E o vocabulario cresce junto com o corpus?"
    )
    report.write()
    report.write(
        "**Configuracao:** prefixos de The Verdict com "
        f"{', '.join(thousands(size) for size in PREFIX_SIZES)} caracteres, "
        "tokenizados pelo splitter proprio e pelo BPE do GPT-2."
    )
    report.write()

    rows = []
    for size in PREFIX_SIZES:
        prefix = corpus[:size]

        simple_tokens = split_text(prefix)
        distinct = len(unique_tokens(prefix))
        bpe_tokens = bpe.encode(prefix)

        rows.append(
            [
                thousands(size),
                thousands(len(simple_tokens)),
                thousands(distinct),
                f"{distinct / len(simple_tokens):.2f}",
                thousands(len(bpe_tokens)),
                f"{size / len(simple_tokens):.2f}",
                f"{size / len(bpe_tokens):.2f}",
            ]
        )

    report.table(
        [
            "caracteres",
            "tokens (proprio)",
            "tokens distintos",
            "distintos/total",
            "tokens (BPE)",
            "car./token proprio",
            "car./token BPE",
        ],
        rows,
    )


def experiment_2_out_of_vocabulary(report: Report, corpus: str, bpe: BPETokenizer) -> None:
    """E2: o que acontece quando o texto tem palavra que o vocabulario nao viu."""
    report.write("## E2 - Comportamento diante de palavra fora do vocabulario")
    report.write()
    report.write(
        "**Pergunta:** qual e o custo de o vocabulario ser fechado, e o que "
        "muda quando <|unk|> entra ou quando o tokenizador e subword?"
    )
    report.write()
    report.write(
        "**Configuracao:** vocabulario construido sobre The Verdict inteiro, "
        "em duas versoes -- com e sem <|unk|>. As frases de teste usam palavras "
        "que nao aparecem no conto."
    )
    report.write()

    strict_vocabulary = Vocabulary.from_texts([corpus], special_tokens=())
    tolerant_vocabulary = Vocabulary.from_texts([corpus])

    strict = SimpleTokenizer(strict_vocabulary)
    tolerant = SimpleTokenizer(tolerant_vocabulary)

    report.write(
        f"Vocabulario sem especiais: {len(strict_vocabulary)} entradas. "
        f"Com <|endoftext|> e <|unk|>: {len(tolerant_vocabulary)}."
    )
    report.write()

    rows = []
    for label, sentence in OUT_OF_VOCABULARY_CASES:
        tokens = split_text(sentence)
        missing = [token for token in tokens if token not in strict_vocabulary]

        try:
            strict.encode(sentence)
            strict_result = "encode ok"
        except KeyError as error:
            strict_result = f"KeyError em {error}"

        unk_ids = tolerant.encode(sentence)
        unk_count = unk_ids.count(tolerant_vocabulary.token_to_id(UNKNOWN_TOKEN))

        bpe_ids = bpe.encode(sentence)
        bpe_lossless = "sim" if bpe.decode(bpe_ids) == sentence else "nao"

        rows.append(
            [
                label,
                f"{len(tokens)}",
                f"{len(missing)}",
                strict_result,
                f"{unk_count}",
                f"{len(bpe_ids)}",
                bpe_lossless,
            ]
        )

    report.table(
        [
            "caso",
            "tokens",
            "ausentes",
            "sem <|unk|>",
            "viram <|unk|>",
            "tokens BPE",
            "BPE reconstroi",
        ],
        rows,
    )

    # O custo do <|unk|> nao e falhar: e apagar a diferenca entre palavras.
    ambiguous = "quantum blockchain"
    report.write(f"Colapso de informacao com <|unk|>, entrada `{ambiguous}`:")
    report.write()
    report.write("```")
    report.write(f"tokens        : {tolerant.tokenize(ambiguous)}")
    report.write(f"Token IDs     : {tolerant.encode(ambiguous)}")
    report.write(f"decode        : {tolerant.decode(tolerant.encode(ambiguous))}")
    report.write(f"BPE tokens    : {bpe.tokenize(ambiguous)}")
    report.write(f"BPE decode    : {bpe.decode(bpe.encode(ambiguous))}")
    report.write("```")
    report.write()


def experiment_3_portuguese_vs_english(report: Report, bpe: BPETokenizer) -> None:
    """E3: o mesmo conteudo custa o mesmo numero de tokens nas duas linguas?"""
    report.write("## E3 - Portugues contra ingles")
    report.write()
    report.write(
        "**Pergunta:** o BPE do GPT-2, treinado majoritariamente em ingles, "
        "gasta mais tokens para dizer a mesma coisa em portugues?"
    )
    report.write()
    report.write(
        f"**Configuracao:** {len(PARALLEL_SENTENCES)} pares de frases equivalentes "
        "em conteudo, tokenizadas pelo splitter proprio e pelo BPE."
    )
    report.write()

    rows = []
    totals = {"pt_simple": 0, "en_simple": 0, "pt_bpe": 0, "en_bpe": 0, "pt_chars": 0, "en_chars": 0}

    for index, (portuguese, english) in enumerate(PARALLEL_SENTENCES, start=1):
        pt_simple = len(split_text(portuguese))
        en_simple = len(split_text(english))
        pt_bpe = len(bpe.encode(portuguese))
        en_bpe = len(bpe.encode(english))

        totals["pt_simple"] += pt_simple
        totals["en_simple"] += en_simple
        totals["pt_bpe"] += pt_bpe
        totals["en_bpe"] += en_bpe
        totals["pt_chars"] += len(portuguese)
        totals["en_chars"] += len(english)

        rows.append(
            [
                f"{index}",
                f"{pt_simple}",
                f"{en_simple}",
                f"{pt_bpe}",
                f"{en_bpe}",
                f"{pt_bpe / en_bpe:.2f}x",
            ]
        )

    rows.append(
        [
            "total",
            f"{totals['pt_simple']}",
            f"{totals['en_simple']}",
            f"{totals['pt_bpe']}",
            f"{totals['en_bpe']}",
            f"{totals['pt_bpe'] / totals['en_bpe']:.2f}x",
        ]
    )

    report.table(
        ["par", "PT proprio", "EN proprio", "PT BPE", "EN BPE", "custo BPE PT/EN"],
        rows,
    )

    report.write(
        f"Caracteres por token no BPE: portugues "
        f"{totals['pt_chars'] / totals['pt_bpe']:.2f}, ingles "
        f"{totals['en_chars'] / totals['en_bpe']:.2f}."
    )
    report.write()

    # Onde a diferenca aparece: palavra acentuada fatiada em varios pedacos.
    report.write("Fragmentacao de palavras acentuadas pelo BPE:")
    report.write()
    report.write(
        "Os `\N{REPLACEMENT CHARACTER}` na saida nao sao erro de codificacao: o BPE opera sobre "
        "bytes, e um caractere acentuado ocupa dois bytes em UTF-8. Quando a "
        "fronteira entre dois tokens cai no meio desse par, nenhum dos dois "
        "tokens isolados forma um caractere valido -- so a concatenacao forma. "
        "E a prova de que a unidade do BPE e o byte, nao a letra."
    )
    report.write()
    report.write("```")
    for word in ("implementação", "atenção", "informações", "não", "implementation", "attention"):
        pieces = bpe.tokenize(word)
        report.write(f"{word:<16} -> {len(pieces)} tokens: {pieces}")
    report.write("```")
    report.write()


def experiment_4_comparison(report: Report, corpus: str, bpe: BPETokenizer) -> None:
    """E4: comparativo consolidado entre os dois tokenizadores."""
    report.write("## E4 - Comparativo: tokenizador proprio contra BPE")
    report.write()
    report.write(
        "**Pergunta:** quais propriedades separam os dois tokenizadores, e "
        "qual delas justifica levar o BPE para as sprints seguintes?"
    )
    report.write()
    report.write("**Configuracao:** The Verdict inteiro (20 479 caracteres).")
    report.write()

    vocabulary = Vocabulary.from_texts([corpus])
    simple = SimpleTokenizer(vocabulary)

    start = time.perf_counter()
    simple_ids = simple.encode(corpus)
    simple_seconds = time.perf_counter() - start

    start = time.perf_counter()
    bpe_ids = bpe.encode(corpus)
    bpe_seconds = time.perf_counter() - start

    simple_lossless = "sim" if simple.decode(simple_ids) == corpus else "nao"
    bpe_lossless = "sim" if bpe.decode(bpe_ids) == corpus else "nao"

    report.table(
        ["propriedade", "proprio (por palavras)", "BPE do GPT-2 (subword)"],
        [
            ["tamanho do vocabulario", thousands(len(vocabulary)), "50 257"],
            [
                "tokens no corpus",
                thousands(len(simple_ids)),
                thousands(len(bpe_ids)),
            ],
            [
                "caracteres por token",
                f"{len(corpus) / len(simple_ids):.2f}",
                f"{len(corpus) / len(bpe_ids):.2f}",
            ],
            ["palavra fora do vocabulario", "<|unk|> ou KeyError", "fatiada em subwords"],
            ["reconstroi o texto original", simple_lossless, bpe_lossless],
            ["espaco", "descartado", "parte do token"],
            ["precisa de <|unk|>", "sim", "nao"],
            ["vocabulario depende do corpus", "sim", "nao (ja treinado)"],
            [
                "tempo de encode do corpus",
                f"{simple_seconds * 1000:.1f} ms",
                f"{bpe_seconds * 1000:.1f} ms",
            ],
        ],
    )

    report.write("Primeiros e ultimos pares (token, ID) do vocabulario proprio:")
    report.write()
    report.write("```")
    report.write(f"inicio : {vocabulary.preview(5)}")
    report.write(f"fim    : {vocabulary.preview(5, from_end=True)}")
    report.write("```")
    report.write()

    # Amostra lado a lado da mesma frase nos dois tokenizadores.
    sample = "It's the last he painted, you know."
    report.write(f"Mesma frase nos dois tokenizadores, entrada `{sample}`:")
    report.write()
    report.write("```")
    report.write(f"proprio tokens : {simple.tokenize(sample)}")
    report.write(f"proprio IDs    : {simple.encode(sample)}")
    report.write(f"proprio decode : {simple.decode(simple.encode(sample))}")
    report.write(f"BPE tokens     : {bpe.tokenize(sample)}")
    report.write(f"BPE IDs        : {bpe.encode(sample)}")
    report.write(f"BPE decode     : {bpe.decode(bpe.encode(sample))}")
    report.write("```")
    report.write()


def main() -> int:
    # O terminal do Windows nao usa UTF-8 por padrao e as frases em portugues
    # quebrariam a impressao.
    sys.stdout.reconfigure(encoding="utf-8")

    corpus = load_verdict()
    bpe = BPETokenizer()

    report = Report()
    report.write("# Sprint 2 - Frente A: resultados dos experimentos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint02/frente_a_experimentos.py`. "
        "Nao editar a mao: rode o script novamente."
    )
    report.write()
    report.write(f"Corpus: The Verdict, {thousands(len(corpus))} caracteres.")
    report.write()

    experiment_1_token_counts(report, corpus, bpe)
    experiment_2_out_of_vocabulary(report, corpus, bpe)
    experiment_3_portuguese_vs_english(report, bpe)
    experiment_4_comparison(report, corpus, bpe)

    report.save(RESULTS_PATH)
    print(f"\nResultados salvos em {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
