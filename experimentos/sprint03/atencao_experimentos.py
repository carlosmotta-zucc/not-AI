"""Experimentos da Sprint 3: o que a atencao faz, alem de estar correta.

Os testes em tests/ provam que a implementacao esta certa. Estes quatro
experimentos mostram o comportamento dela. A interpretacao vai para
docs/sprint03/analise.md -- aqui ficam so os numeros que ela cita.

E1  Com e sem a divisao por raiz de d_k
E2  Dimensao do embedding e tamanho da sequencia: parametros, memoria e tempo
E3  Visualizacao: sem pesos treinaveis, para onde cada token olha?
E4  Sequencias diferentes: o vetor de um token depende do que vem depois?
E5  Mascara causal: o que ela muda nos pesos, e o que o dropout faz com eles
E6  Numero de cabecas, com d_out fixo
E7  Dimensao da cabeca, com o numero de cabecas fixo
E8  Uma cabeca contra quatro: os tres caminhos com a mesma saida

Tudo roda em eval() e sob torch.no_grad(), com duas excecoes: o bloco do E1,
que precisa do backward, e a tabela de dropout do E5, que so faz sentido em
train(). Seed fixa antes de cada tensor sorteado -- sem isso nenhuma tabela
daqui seria reproduzivel. A unica fonte de variacao entre execucoes sao as
colunas de tempo (E2, E6, E7 e E8).

Uso:
    python experimentos/sprint03/atencao_experimentos.py
"""

import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from heatmap import attention_heatmap  # noqa: E402
from src.attention import (  # noqa: E402
    CausalAttention,
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
    SelfAttention,
    causal_mask,
    scaled_dot_product_attention,
    simplified_attention,
)
from src.embeddings import InputEmbedding  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

RESULTS_PATH = Path(__file__).parent / "resultados" / "atencao.md"
FIGURES_DIR = Path(__file__).parent / "resultados" / "figuras"

SEED = 42

# A mesma varredura de dimensao da Sprint 2, para as duas sprints poderem ser
# lidas lado a lado. 768 e a dimensao do GPT-2 small.
DIM_SWEEP = (16, 32, 64, 128, 256, 768)

# E1: sequencia curta e fixa -- o que varia no experimento e d_k, nao T.
E1_TOKENS = 32

# 4096 esta fora da faixa do projeto e entra so no E1: em 768 a saturacao ja
# aparece mas ainda nao e gritante, e o limite so fica obvio com um ponto alem.
E1_DIM_SWEEP = DIM_SWEEP + (4096,)

# E2: a configuracao base da Sprint 2 fixa o lado que nao esta sendo variado.
E2_TOKENS = 128
E2_DIM = 256
E2_LENGTH_SWEEP = (8, 32, 128, 256, 512, 1024)
E2_WARMUP = 3
E2_REPEATS = 11

# E3 e E4: as frases declaradas no docstring de src/attention/__init__.py.
# Prefixo igual e um token final diferente: qualquer mudanca no vetor de
# "bank" so pode ter vindo do fim da frase.
PHRASE_RIVER = "The bank of the river"
PHRASE_CITY = "The bank of the city"

# E5: 0.0 como controle, 0.1 e a taxa da configuracao base do pacote, 0.5
# exagera o efeito para ele caber em cinco tokens.
E5_DROPOUT_SWEEP = (0.0, 0.1, 0.5)

# E6 e E7: as duas maneiras de mexer nas cabecas, uma de cada vez.
E6_HEAD_SWEEP = (1, 2, 4, 8, 16)
E7_HEAD_DIM_SWEEP = (16, 32, 64, 128)
E7_HEADS = 4

# E8: a configuracao base da Sprint 3 -- 4 cabecas de 64 sobre d_out 256.
E8_HEADS = 4

BYTES_PER_MEGABYTE = 1024 * 1024
BYTES_PER_FLOAT32 = 4


def thousands(value: int) -> str:
    """Formata inteiro com espaco como separador de milhar (padrao pt-BR)."""
    return f"{value:,}".replace(",", " ")


def megabytes(tensor_elements: int) -> str:
    """Converte contagem de elementos float32 em tamanho legivel."""
    return f"{tensor_elements * BYTES_PER_FLOAT32 / BYTES_PER_MEGABYTE:.2f} MB"


def row_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Entropia de cada linha de pesos, em nats.

    Args:
        weights: (..., T, T), linhas somando 1.

    Returns:
        Tensor (..., T) com a entropia de cada linha.

    Mede o quanto a atencao esta espalhada: 0 quando a linha e one-hot (olha um
    token so), ln(T) quando e uniforme. E a leitura que o peso maximo sozinho
    nao da -- maximo 0,3 pode ser tanto "tres candidatos" quanto "um favorito
    fraco num mar de zeros".
    """
    # clamp antes do log: peso exatamente zero (posicao mascarada) daria -inf, e
    # 0 * -inf = nan. O limite de x*ln(x) com x -> 0 e 0, que e o que se quer.
    seguro = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    # O + 0.0 no fim nao e decoracao: linha one-hot da -0.0, que imprime como
    # "-0.000", e clamp_min(0.0) nao corrige (-0.0 nao e MENOR que 0.0). Somar
    # zero positivo e o que troca o sinal, pela regra do IEEE 754.
    return (-(weights * seguro.log()).sum(dim=-1)).clamp_min(0.0) + 0.0


def softmax_sensitivity(weights: torch.Tensor) -> torch.Tensor:
    """Traco do jacobiano do softmax, linha a linha: soma de p(1-p).

    Args:
        weights: (..., T, T), linhas somando 1.

    Returns:
        Tensor (..., T), entre 0 e 1 - 1/T.

    E o fator pelo qual o softmax multiplica o gradiente que passa por ele:
    vale ~1 - 1/T na linha uniforme e vai a 0 na linha one-hot. Ao contrario da
    norma do gradiente medida na saida, nao depende da escala da perda, entao e
    esta coluna que isola a saturacao.
    """
    return (weights * (1 - weights)).sum(dim=-1)


def median_forward_ms(layer: torch.nn.Module, inputs: torch.Tensor) -> float:
    """Mediana do tempo de forward, em milissegundos.

    Mediana e nao media: a maquina e CPU compartilhada com o resto do sistema, e
    um pico do escalonador em uma das repeticoes levaria a media junto.
    """
    with torch.no_grad():
        for _ in range(E2_WARMUP):
            layer(inputs)

        tempos = []
        for _ in range(E2_REPEATS):
            inicio = time.perf_counter()
            layer(inputs)
            tempos.append((time.perf_counter() - inicio) * 1000)

    return statistics.median(tempos)


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


def experiment_1_scale(report: Report) -> None:
    """E1: a divisao por raiz de d_k muda o regime do softmax, ou so os numeros?"""
    report.write("## E1 - Com e sem a divisao por raiz de d_k")
    report.write()
    report.write(
        "**Pergunta:** a escala e cosmetica ou muda o regime em que o softmax "
        "opera? E o que acontece com o gradiente que volta por ele?"
    )
    report.write()
    report.write(
        f"**Configuracao:** Q, K e V sorteados de uma normal padrao, "
        f"shape (1, {E1_TOKENS}, d_k), d_k na varredura "
        f"{', '.join(str(d) for d in E1_DIM_SWEEP)}. Normal e nao uniforme: o "
        "argumento do raiz de d_k pressupoe componentes de media zero e "
        "variancia 1; com uniforme em [0, 1) os scores sairiam todos positivos "
        "e o efeito ficaria mascarado. O ultimo d_k esta fora da faixa do "
        "projeto e entra so para o limite ficar visivel."
    )
    report.write()
    report.write(
        "Duas colunas medem o gradiente, e elas nao dizem a mesma coisa. A "
        "**sensibilidade** e a soma de p(1-p) na linha, o fator pelo qual o "
        "softmax multiplica o gradiente que passa por ele: vale ~1 na linha "
        "uniforme e 0 na linha one-hot, e nao depende da escala da perda. A "
        "**norma do gradiente** e medida no caminho real da atencao "
        "(`(pesos @ values).pow(2).mean()`, derivada em relacao aos scores que "
        "entram no softmax). Nao e uma cross-entropy de proposito: o gradiente "
        "dela em relacao aos scores e `p - y`, que nao desaparece com o "
        "softmax saturado porque a conta cancela o jacobiano -- medir por ali "
        "daria a conclusao oposta a correta."
    )
    report.write()

    entropia_maxima = torch.tensor(float(E1_TOKENS)).log().item()
    report.write(
        f"Entropia maxima possivel com {E1_TOKENS} tokens: "
        f"ln({E1_TOKENS}) = {entropia_maxima:.3f} nats (atencao uniforme). "
        "Zero seria olhar um token so."
    )
    report.write()

    linhas = []
    for d_k in E1_DIM_SWEEP:
        for escala in (True, False):
            # Mesma seed dentro do par: as duas linhas de cada d_k comparam o
            # MESMO sorteio, entao a diferenca so pode vir da escala.
            torch.manual_seed(SEED)
            queries, keys, values = (torch.randn(1, E1_TOKENS, d_k) for _ in range(3))

            scores = queries @ keys.transpose(-2, -1)
            if escala:
                scores = scores / d_k**0.5

            # detach + requires_grad_ faz dos scores uma folha do grafo: e o que
            # permite ler .grad neles e isolar o que volta pelo softmax.
            scores = scores.detach().requires_grad_(True)
            pesos = torch.softmax(scores, dim=-1)
            pesos.retain_grad()
            (pesos @ values).pow(2).mean().backward()

            linhas.append(
                [
                    str(d_k),
                    "sim" if escala else "nao",
                    f"{scores.detach().std().item():.3f}",
                    f"{pesos.max(dim=-1).values.mean().item():.3f}",
                    f"{row_entropy(pesos).mean().item():.3f}",
                    f"{softmax_sensitivity(pesos).mean().item():.4f}",
                    f"{scores.grad.norm().item():.2e}",
                    f"{pesos.grad.norm().item():.2e}",
                ]
            )

    report.table(
        [
            "d_k",
            "escala",
            "desvio dos scores",
            "peso maximo medio",
            "entropia media (nats)",
            "sensibilidade Σp(1-p)",
            "norma do grad. nos scores",
            "norma do grad. nos pesos",
        ],
        linhas,
    )

    report.write(
        "A sensibilidade cai uma ordem de grandeza sem a escala e fica parada "
        "com ela -- e a saturacao, medida. Ja a norma do gradiente nos scores "
        "NAO desaba junto, e o motivo esta na ultima coluna: quando a atencao "
        "se concentra num token so, o vetor de contexto vira uma linha de V "
        "inteira em vez de uma media de 32, a perda cresce e o gradiente que "
        "CHEGA ao softmax cresce com ela. Os dois efeitos se cancelam em parte. "
        "Ler so a norma levaria a concluir que a escala nao muda nada; a "
        "conclusao certa e que ela muda o regime, e o preco da saturacao "
        "aparece quando a rede fica profunda e esse fator se multiplica camada "
        "apos camada."
    )
    report.write()


def experiment_2_cost(report: Report) -> None:
    """E2: o que cresce linear e o que cresce ao quadrado."""
    report.write("## E2 - Dimensao do embedding e tamanho da sequencia")
    report.write()
    report.write(
        "**Pergunta:** aumentar `emb_dim` e aumentar `context_length` custam a "
        "mesma coisa? Onde cada um aparece -- em parametros, em memoria ou em "
        "tempo?"
    )
    report.write()
    report.write(
        f"**Configuracao:** SelfAttention com d_in = d_out, lote de 1 "
        f"sequencia. Tempo: {E2_WARMUP} execucoes de aquecimento descartadas e "
        f"mediana de {E2_REPEATS} medidas, sob `torch.no_grad()` e em `eval()`. "
        "Maquina CPU-only (ver requirements.txt), entao os tempos absolutos "
        "valem pouco; o que interessa e a razao entre linhas."
    )
    report.write()

    report.write(f"### E2a - Dimensao, com T = {E2_TOKENS} fixo")
    report.write()

    linhas = []
    for dim in DIM_SWEEP:
        torch.manual_seed(SEED)
        camada = SelfAttention(d_in=dim, d_out=dim).eval()
        entrada = torch.randn(1, E2_TOKENS, dim)

        parametros = sum(p.numel() for p in camada.parameters())
        # A formula e 3 * d_in * d_out (tres projecoes, sem bias); conferir
        # contra o modulo evita a tabela descrever uma camada que nao e esta.
        assert parametros == 3 * dim * dim

        linhas.append(
            [
                str(dim),
                thousands(parametros),
                megabytes(parametros),
                f"{median_forward_ms(camada, entrada):.2f}",
            ]
        )

    report.table(
        ["d_in = d_out", "parametros (3·d_in·d_out)", "pesos em float32", "tempo (ms)"],
        linhas,
    )

    report.write(f"### E2b - Comprimento, com d_in = d_out = {E2_DIM} fixo")
    report.write()

    torch.manual_seed(SEED)
    camada = SelfAttention(d_in=E2_DIM, d_out=E2_DIM).eval()
    parametros = sum(p.numel() for p in camada.parameters())

    linhas = []
    for tokens in E2_LENGTH_SWEEP:
        entrada = torch.randn(1, tokens, E2_DIM)
        linhas.append(
            [
                str(tokens),
                thousands(parametros),
                thousands(tokens * tokens),
                megabytes(tokens * tokens),
                f"{median_forward_ms(camada, entrada):.2f}",
            ]
        )

    report.table(
        [
            "T",
            "parametros",
            "celulas da matriz T×T",
            "matriz T×T em float32",
            "tempo (ms)",
        ],
        linhas,
    )

    report.write(
        "A coluna de parametros do E2b e constante de proposito: a sequencia "
        "mais longa nao cria peso nenhum. O que ela cria e a matriz T×T, que e "
        "temporaria e quadratica -- e o custo que limita o `context_length`."
    )
    report.write()


def experiment_3_visualization(report: Report, bpe: BPETokenizer) -> None:
    """E3: sem pesos treinaveis, para onde cada token olha?"""
    report.write("## E3 - Visualizacao: com e sem pesos treinaveis")
    report.write()
    report.write(
        "**Pergunta:** sem pesos treinaveis, para onde cada token olha? E o que "
        "muda quando W_q, W_k e W_v entram?"
    )
    report.write()

    ids = bpe.encode(PHRASE_RIVER)
    tokens = bpe.tokenize(PHRASE_RIVER)

    report.write(
        f"**Configuracao:** frase `{PHRASE_RIVER}`, tokenizada pelo BPE em "
        f"{len(tokens)} tokens: "
        + ", ".join(f"`{token!r}`" for token in tokens)
        + f". InputEmbedding com emb_dim {E2_DIM} e os MESMOS vetores entrando "
        "nas duas versoes -- a unica diferenca e o que acontece depois deles."
    )
    report.write()

    torch.manual_seed(SEED)
    embedding = InputEmbedding(
        vocab_size=len(bpe), emb_dim=E2_DIM, context_length=E2_TOKENS
    ).eval()
    atencao = SelfAttention(d_in=E2_DIM, d_out=E2_DIM).eval()

    with torch.no_grad():
        vetores = embedding(torch.tensor([ids]))
        _, pesos_simples = simplified_attention(vetores)
        _, pesos_treinaveis = atencao(vetores)

    figuras = [
        (
            "sem pesos treinaveis (atencao simplificada)",
            pesos_simples[0],
            FIGURES_DIR / "e3-simplificada.png",
        ),
        (
            "com pesos treinaveis (SelfAttention)",
            pesos_treinaveis[0],
            FIGURES_DIR / "e3-self-attention.png",
        ),
    ]

    for titulo, pesos, caminho in figuras:
        attention_heatmap(pesos, tokens, f"{PHRASE_RIVER}\n{titulo}", caminho)
        report.write(f"![{titulo}]({caminho.relative_to(RESULTS_PATH.parent)})")
        report.write()

    # A tabela repete o essencial das figuras: o markdown tem que se sustentar
    # sem as imagens.
    linhas = []
    for posicao, token in enumerate(tokens):
        linha_simples = pesos_simples[0, posicao]
        linha_treinavel = pesos_treinaveis[0, posicao]

        linhas.append(
            [
                f"`{token!r}`",
                f"`{tokens[int(linha_simples.argmax())]!r}`",
                f"{linha_simples.max().item():.3f}",
                f"{row_entropy(linha_simples).item():.3f}",
                f"`{tokens[int(linha_treinavel.argmax())]!r}`",
                f"{linha_treinavel.max().item():.3f}",
                f"{row_entropy(linha_treinavel).item():.3f}",
            ]
        )

    report.table(
        [
            "token",
            "sem pesos: olha mais",
            "peso",
            "entropia",
            "com pesos: olha mais",
            "peso",
            "entropia",
        ],
        linhas,
    )

    diagonal_simples = int(
        sum(int(pesos_simples[0, i].argmax()) == i for i in range(len(tokens)))
    )
    diagonal_treinavel = int(
        sum(int(pesos_treinaveis[0, i].argmax()) == i for i in range(len(tokens)))
    )
    report.write(
        f"Tokens que olham mais para si mesmos: {diagonal_simples} de "
        f"{len(tokens)} sem pesos treinaveis, {diagonal_treinavel} de "
        f"{len(tokens)} com. Sem W_q e W_k o criterio e o produto escalar cru "
        "dos embeddings, e o vetor mais parecido com um token tende a ser ele "
        "mesmo."
    )
    report.write()
    report.write(
        "A diagonal da primeira figura nao e so parecida com o E1 -- e o mesmo "
        "fenomeno. A atencao simplificada roda com `scale=False` sobre vetores "
        f"de {E2_DIM} dimensoes, entao o produto de um embedding por ele mesmo "
        "e da ordem de centenas enquanto o dos outros fica perto de zero: e a "
        f"linha `d_k = {E2_DIM}, escala = nao` do E1, levada ao extremo. O peso "
        "1,000 com entropia 0,000 e softmax saturado, nao afinidade semantica. "
        "Com embeddings ainda nao treinados nao havia semantica alguma a "
        "encontrar."
    )
    report.write()


def experiment_4_sequences(report: Report, bpe: BPETokenizer) -> None:
    """E4: o vetor de um token depende do que vem depois dele?"""
    report.write("## E4 - Duas sequencias com o mesmo comeco")
    report.write()
    report.write(
        "**Pergunta:** o vetor de contexto de um token depende do que vem "
        "DEPOIS dele?"
    )
    report.write()

    ids_river = bpe.encode(PHRASE_RIVER)
    ids_city = bpe.encode(PHRASE_CITY)
    tokens = bpe.tokenize(PHRASE_RIVER)

    # O experimento so faz sentido se as duas frases tiverem o mesmo prefixo e o
    # mesmo comprimento -- e isso e propriedade da tokenizacao, nao da frase.
    assert len(ids_river) == len(ids_city)
    assert ids_river[:-1] == ids_city[:-1]
    prefixo = len(ids_river) - 1

    report.write(
        f"**Configuracao:** `{PHRASE_RIVER}` e `{PHRASE_CITY}`, as duas pelas "
        "MESMAS instancias de InputEmbedding e de atencao. Pelo BPE as duas "
        f"dao {len(ids_river)} tokens, iguais nas {prefixo} primeiras posicoes "
        f"e diferentes so na ultima (`{bpe.tokenize(PHRASE_RIVER)[-1]!r}` "
        f"contra `{bpe.tokenize(PHRASE_CITY)[-1]!r}`). Distancia L2 entre os "
        "vetores correspondentes."
    )
    report.write()

    torch.manual_seed(SEED)
    embedding = InputEmbedding(
        vocab_size=len(bpe), emb_dim=E2_DIM, context_length=E2_TOKENS
    ).eval()
    self_attention = SelfAttention(d_in=E2_DIM, d_out=E2_DIM).eval()

    with torch.no_grad():
        entrada_river = embedding(torch.tensor([ids_river]))
        entrada_city = embedding(torch.tensor([ids_city]))

        contexto_self_river, _ = self_attention(entrada_river)
        contexto_self_city, _ = self_attention(entrada_city)

    linhas = []
    for posicao in range(prefixo):
        distancia_entrada = torch.dist(
            entrada_river[0, posicao], entrada_city[0, posicao]
        ).item()
        # Mesmo Token ID na mesma posicao: as duas tabelas devolvem a mesma
        # linha, entao a soma e bit a bit identica. Zero exato, nao "pequeno".
        assert distancia_entrada == 0.0

        linhas.append(
            [
                str(posicao),
                f"`{tokens[posicao]!r}`",
                f"{distancia_entrada:.6f}",
                f"{torch.dist(contexto_self_river[0, posicao], contexto_self_city[0, posicao]).item():.6f}",
            ]
        )

    report.table(
        ["posicao", "token", "distancia na entrada", "distancia no contexto"],
        linhas,
    )

    report.write(
        "A coluna da entrada e zero exato: o mesmo Token ID na mesma posicao le "
        "as mesmas duas linhas das tabelas de embedding. Toda diferenca das "
        "outras colunas entrou na camada de atencao, e so pode ter vindo do "
        "ultimo token -- e o unico lugar em que as frases diferem."
    )
    report.write()
    report.write(
        "Esta camada nao tem mascara, entao a posicao 1 enxerga a posicao 4 e a "
        "troca de `' river'` por `' city'` chega em todo mundo. A mesma medida "
        "com a mascara causal esta no E5."
    )
    report.write()


def experiment_5_mask(report: Report, bpe: BPETokenizer) -> None:
    """E5: o que a mascara muda nos pesos, e o que o dropout faz com eles."""
    report.write("## E5 - Mascara causal e dropout")
    report.write()
    report.write(
        "**Pergunta:** o que a mascara muda nos pesos? Ela e mesmo equivalente "
        "a renormalizar a mao? E o que o dropout faz com o que sobrou?"
    )
    report.write()

    ids = bpe.encode(PHRASE_RIVER)
    tokens = bpe.tokenize(PHRASE_RIVER)
    num_tokens = len(ids)

    report.write(
        f"**Configuracao:** frase `{PHRASE_RIVER}` ({num_tokens} tokens) por uma "
        f"CausalAttention com d_in = d_out = {E2_DIM}. As duas versoes saem da "
        "MESMA instancia: as projecoes dela alimentam a funcao nucleo duas "
        "vezes, uma com `mask=None` e outra com a mascara. Instanciar uma "
        "SelfAttention separada para o 'antes' traria pesos diferentes, e a "
        "figura passaria a mostrar duas mudancas ao mesmo tempo."
    )
    report.write()

    torch.manual_seed(SEED)
    embedding = InputEmbedding(
        vocab_size=len(bpe), emb_dim=E2_DIM, context_length=E2_TOKENS
    ).eval()
    self_attention = SelfAttention(d_in=E2_DIM, d_out=E2_DIM).eval()
    causal = CausalAttention(
        d_in=E2_DIM, d_out=E2_DIM, context_length=E2_TOKENS
    ).eval()

    mascara = causal_mask(num_tokens)

    with torch.no_grad():
        vetores = embedding(torch.tensor([ids]))
        queries = causal.W_query(vetores)
        keys = causal.W_key(vetores)
        values = causal.W_value(vetores)

        _, pesos_sem = scaled_dot_product_attention(queries, keys, values)
        _, pesos_com = scaled_dot_product_attention(
            queries, keys, values, mask=mascara
        )

    figuras = [
        ("antes da mascara", pesos_sem[0], FIGURES_DIR / "e5-sem-mascara.png"),
        ("depois da mascara", pesos_com[0], FIGURES_DIR / "e5-com-mascara.png"),
    ]
    for titulo, pesos, caminho in figuras:
        attention_heatmap(pesos, tokens, f"{PHRASE_RIVER}\n{titulo}", caminho)
        report.write(f"![{titulo}]({caminho.relative_to(RESULTS_PATH.parent)})")
        report.write()

    linhas = []
    for posicao, token in enumerate(tokens):
        linhas.append(
            [
                str(posicao),
                f"`{token!r}`",
                str(posicao + 1),
                f"{pesos_sem[0, posicao].max().item():.3f}",
                f"{row_entropy(pesos_sem[0, posicao]).item():.3f}",
                f"{pesos_com[0, posicao].max().item():.3f}",
                f"{row_entropy(pesos_com[0, posicao]).item():.3f}",
            ]
        )

    report.table(
        [
            "posicao",
            "token",
            "tokens visiveis",
            "antes: peso maximo",
            "antes: entropia",
            "depois: peso maximo",
            "depois: entropia",
        ],
        linhas,
    )

    report.write(
        "A entropia da primeira linha cai a zero com a mascara, e isso nao e a "
        "atencao ficando mais decidida: e ela nao tendo escolha. O primeiro "
        "token ve um candidato so, entao gasta 1,000 nele. A cada posicao a "
        "escolha aumenta, e e por isso que a coluna 'tokens visiveis' precisa "
        "estar do lado -- sem ela a queda da entropia pareceria um efeito de "
        "aprendizado."
    )
    report.write()

    # Duas maneiras de chegar nos mesmos pesos, da secao 3.5.1. O teste afirma
    # que coincidem; aqui vira numero.
    with torch.no_grad():
        scores = queries @ keys.transpose(-2, -1) / keys.shape[-1] ** 0.5
        renormalizado = torch.softmax(scores, dim=-1).masked_fill(mascara, 0.0)
        renormalizado = renormalizado / renormalizado.sum(dim=-1, keepdim=True)
        diferenca = (renormalizado - pesos_com).abs().max().item()

    report.write(
        f"**-inf antes do softmax contra zerar e renormalizar depois:** "
        f"diferenca maxima de {diferenca:.2e} entre os dois caminhos. Nao e "
        "zero exato porque sao duas sequencias diferentes de operacoes em ponto "
        "flutuante, e e da ordem do epsilon do float32 (1.19e-07). Os dois "
        "resultados sao o mesmo; o pacote usa o primeiro por ser um passo a "
        "menos."
    )
    report.write()

    report.write("### O par de frases pela CausalAttention")
    report.write()

    ids_river = bpe.encode(PHRASE_RIVER)
    ids_city = bpe.encode(PHRASE_CITY)
    prefixo = len(ids_river) - 1

    with torch.no_grad():
        entrada_river = embedding(torch.tensor([ids_river]))
        entrada_city = embedding(torch.tensor([ids_city]))

        contexto_self_river, _ = self_attention(entrada_river)
        contexto_self_city, _ = self_attention(entrada_city)
        contexto_causal_river, _ = causal(entrada_river)
        contexto_causal_city, _ = causal(entrada_city)

    linhas = []
    for posicao in range(prefixo):
        distancia_causal = torch.dist(
            contexto_causal_river[0, posicao], contexto_causal_city[0, posicao]
        ).item()
        # Zero exato, nao "pequeno": a posicao do prefixo nao le a ultima
        # coluna, entao a conta dela e literalmente a mesma nas duas frases.
        assert distancia_causal == 0.0

        linhas.append(
            [
                str(posicao),
                f"`{tokens[posicao]!r}`",
                f"{torch.dist(contexto_self_river[0, posicao], contexto_self_city[0, posicao]).item():.6f}",
                f"{distancia_causal:.6f}",
            ]
        )

    report.table(
        ["posicao", "token", "SelfAttention (sem mascara)", "CausalAttention"],
        linhas,
    )

    report.write(
        "Esta e a coluna que faltava no E4, e e o que a mascara compra em "
        "numero: com ela, trocar o fim da frase nao move um bit em nada que "
        "veio antes. E a condicao para o modelo poder ser treinado em todas as "
        "posicoes de uma vez sem ler a resposta."
    )
    report.write()

    report.write("### Dropout sobre os pesos")
    report.write()
    report.write(
        "O dropout do pacote age nos PESOS, depois do softmax e depois da "
        "mascara. A tabela usa os mesmos `pesos_com` acima, com seed fixa antes "
        "de cada sorteio. As duas colunas de fracao existem porque a mascara "
        f"sozinha ja zera {num_tokens * (num_tokens - 1) // 2} das "
        f"{num_tokens * num_tokens} posicoes: so a segunda pode ser comparada "
        "com p."
    )
    report.write()

    permitidas = ~mascara
    linhas = []
    for probabilidade in E5_DROPOUT_SWEEP:
        torch.manual_seed(SEED)
        dropout = torch.nn.Dropout(probabilidade).train()

        with torch.no_grad():
            pesos_treino = dropout(pesos_com)

        sobrevive = pesos_treino > 0
        fator = (pesos_treino[sobrevive] / pesos_com[sobrevive]).max().item()

        linhas.append(
            [
                f"{probabilidade:.1f}",
                f"{(pesos_treino == 0).float().mean().item():.3f}",
                f"{(pesos_treino[:, permitidas] == 0).float().mean().item():.3f}",
                f"{fator:.4f}",
                f"{1 / (1 - probabilidade):.4f}",
                f"{pesos_treino.sum(dim=-1).mean().item():.3f}",
            ]
        )

    report.table(
        [
            "p",
            "fracao zerada (total)",
            "fracao zerada (so permitidas)",
            "fator medido",
            "1/(1-p)",
            "soma media das linhas em train()",
        ],
        linhas,
    )

    report.write(
        "O fator medido e exatamente 1/(1-p): o dropout nao so apaga, ele "
        "aumenta quem sobrou, para que o VALOR ESPERADO da soma continue 1. "
        "Esperado, nao realizado -- a ultima coluna mostra a soma de uma "
        "amostra so, e ela nao da 1. Em `eval()` as linhas somam 1 sempre, e e "
        "por isso que todo numero de peso deste relatorio foi medido la."
    )
    report.write()


def experiment_6_num_heads(report: Report) -> None:
    """E6: mais cabecas custa mais parametro, ou custa outra coisa?"""
    report.write(f"## E6 - Numero de cabecas, com d_out = {E2_DIM} fixo")
    report.write()
    report.write(
        "**Pergunta:** dobrar o numero de cabecas dobra o que, exatamente?"
    )
    report.write()
    report.write(
        f"**Configuracao:** MultiHeadAttention com d_in = d_out = {E2_DIM}, "
        f"T = {E2_TOKENS}, lote de 1, num_heads na varredura "
        f"{', '.join(str(h) for h in E6_HEAD_SWEEP)}."
    )
    report.write()

    torch.manual_seed(SEED)
    entrada = torch.randn(1, E2_TOKENS, E2_DIM)

    linhas = []
    for cabecas in E6_HEAD_SWEEP:
        torch.manual_seed(SEED)
        camada = MultiHeadAttention(
            d_in=E2_DIM,
            d_out=E2_DIM,
            context_length=E2_TOKENS,
            num_heads=cabecas,
        ).eval()

        celulas = cabecas * E2_TOKENS * E2_TOKENS
        linhas.append(
            [
                str(cabecas),
                str(camada.head_dim),
                thousands(sum(p.numel() for p in camada.parameters())),
                thousands(celulas),
                megabytes(celulas),
                f"{median_forward_ms(camada, entrada):.2f}",
            ]
        )

    report.table(
        [
            "num_heads",
            "head_dim",
            "parametros",
            "celulas de (B, H, T, T)",
            "scores em float32",
            "tempo (ms)",
        ],
        linhas,
    )

    report.write(
        "A coluna de parametros nao se mexe: a projecao continua sendo "
        f"{E2_DIM} -> {E2_DIM}, e o que muda e so em quantos blocos as colunas "
        "sao lidas. Mais cabecas com o mesmo d_out da cabecas MENORES, nao rede "
        "maior. Ja o tensor de scores cresce linear com o numero de cabecas: o "
        "custo das cabecas e memoria de ativacao, temporaria, e nao peso."
    )
    report.write()


def experiment_7_head_dim(report: Report) -> None:
    """E7: e quando e a cabeca que cresce?"""
    report.write(f"## E7 - Dimensao da cabeca, com {E7_HEADS} cabecas fixas")
    report.write()
    report.write(
        "**Pergunta:** o contraste do E6 -- e quando quem cresce e a cabeca?"
    )
    report.write()
    report.write(
        f"**Configuracao:** MultiHeadAttention com d_in = {E2_DIM} fixo, "
        f"num_heads = {E7_HEADS} fixo e head_dim na varredura "
        f"{', '.join(str(d) for d in E7_HEAD_DIM_SWEEP)}, entao "
        f"d_out = {E7_HEADS} x head_dim."
    )
    report.write()

    torch.manual_seed(SEED)
    entrada = torch.randn(1, E2_TOKENS, E2_DIM)

    linhas = []
    for head_dim in E7_HEAD_DIM_SWEEP:
        d_out = E7_HEADS * head_dim

        torch.manual_seed(SEED)
        camada = MultiHeadAttention(
            d_in=E2_DIM,
            d_out=d_out,
            context_length=E2_TOKENS,
            num_heads=E7_HEADS,
        ).eval()

        parametros = sum(p.numel() for p in camada.parameters())
        # Tres projecoes d_in -> d_out, mais a out_proj d_out -> d_out com bias.
        assert parametros == 3 * E2_DIM * d_out + d_out * d_out + d_out

        celulas = E7_HEADS * E2_TOKENS * E2_TOKENS
        linhas.append(
            [
                str(head_dim),
                str(d_out),
                thousands(parametros),
                thousands(celulas),
                f"{median_forward_ms(camada, entrada):.2f}",
            ]
        )

    report.table(
        [
            "head_dim",
            "d_out",
            "parametros",
            "celulas de (B, H, T, T)",
            "tempo (ms)",
        ],
        linhas,
    )

    report.write(
        "Espelho do E6: aqui o peso cresce e a ativacao nao. O tensor de scores "
        f"e o mesmo nas quatro linhas, porque nem H nem T mudaram -- head_dim "
        "nao aparece no shape (B, H, T, T). E os parametros crescem mais que "
        "linear, porque a out_proj e d_out x d_out."
    )
    report.write()


def experiment_8_one_head_against_four(report: Report, bpe: BPETokenizer) -> None:
    """E8: os tres caminhos com a mesma saida de 256."""
    report.write(f"## E8 - Uma cabeca contra {E8_HEADS}, com saida de {E2_DIM}")
    report.write()
    report.write(
        "**Pergunta:** com a mesma dimensao de saida, o que separa a atencao de "
        "uma cabeca, a lista de cabecas e o weight split?"
    )
    report.write()
    report.write(
        f"**Configuracao:** CausalAttention (1 cabeca de {E2_DIM}), "
        f"MultiHeadAttentionWrapper e MultiHeadAttention (as duas com "
        f"{E8_HEADS} cabecas de {E2_DIM // E8_HEADS}), todas com "
        f"d_in = d_out = {E2_DIM}, T = {E2_TOKENS} e lote de 1."
    )
    report.write()

    torch.manual_seed(SEED)
    entrada = torch.randn(1, E2_TOKENS, E2_DIM)

    torch.manual_seed(SEED)
    uma_cabeca = CausalAttention(
        d_in=E2_DIM, d_out=E2_DIM, context_length=E2_TOKENS
    ).eval()
    torch.manual_seed(SEED)
    wrapper = MultiHeadAttentionWrapper(
        d_in=E2_DIM, d_out=E2_DIM, context_length=E2_TOKENS, num_heads=E8_HEADS
    ).eval()
    torch.manual_seed(SEED)
    weight_split = MultiHeadAttention(
        d_in=E2_DIM, d_out=E2_DIM, context_length=E2_TOKENS, num_heads=E8_HEADS
    ).eval()

    camadas = [
        ("CausalAttention", uma_cabeca, "1", str(E2_DIM), "nao"),
        ("MultiHeadAttentionWrapper", wrapper, str(E8_HEADS), str(E2_DIM // E8_HEADS), "nao"),
        ("MultiHeadAttention", weight_split, str(E8_HEADS), str(E2_DIM // E8_HEADS), "sim"),
    ]

    linhas = []
    for nome, camada, cabecas, head_dim, tem_out_proj in camadas:
        linhas.append(
            [
                nome,
                cabecas,
                head_dim,
                thousands(sum(p.numel() for p in camada.parameters())),
                tem_out_proj,
                f"{median_forward_ms(camada, entrada):.2f}",
            ]
        )

    report.table(
        ["camada", "cabecas", "head_dim", "parametros", "out_proj", "tempo (ms)"],
        linhas,
    )

    report.write(
        f"As tres projecoes valem {thousands(3 * E2_DIM * E2_DIM)} nas tres "
        f"linhas: {E8_HEADS} cabecas de {E2_DIM // E8_HEADS} somam exatamente o "
        f"que uma de {E2_DIM} soma. A diferenca de parametros e so a out_proj, "
        "que a MultiHeadAttention tem e as outras duas nao. O Wrapper e mais "
        "lento porque e um laco: quatro passadas pelo mesmo tensor e quatro "
        "mascaras alocadas, onde o weight split faz um matmul so."
    )
    report.write()

    # A mesma copia de pesos do teste de equivalencia, agora medida.
    with torch.no_grad():
        for projecao in ("W_query", "W_key", "W_value"):
            empilhado = torch.cat(
                [getattr(cabeca, projecao).weight for cabeca in wrapper.heads], dim=0
            )
            getattr(weight_split, projecao).weight.copy_(empilhado)
        weight_split.out_proj.weight.copy_(torch.eye(E2_DIM))
        weight_split.out_proj.bias.zero_()

        contexto_wrapper, pesos_wrapper = wrapper(entrada)
        contexto_split, pesos_split = weight_split(entrada)

    report.write(
        "**Wrapper contra weight split, com os pesos copiados** (as tres "
        "projecoes empilhadas em `dim=0` e a out_proj como identidade): "
        f"diferenca maxima de "
        f"{(contexto_wrapper - contexto_split).abs().max().item():.2e} nos "
        "vetores de contexto e de "
        f"{(pesos_wrapper - pesos_split).abs().max().item():.2e} nos pesos de "
        "atencao. Ordem do epsilon do float32: as duas implementacoes sao a "
        "mesma conta, com as operacoes agrupadas de outro jeito."
    )
    report.write()

    report.write("### Um heatmap por cabeca")
    report.write()

    ids = bpe.encode(PHRASE_RIVER)
    tokens = bpe.tokenize(PHRASE_RIVER)

    torch.manual_seed(SEED)
    embedding = InputEmbedding(
        vocab_size=len(bpe), emb_dim=E2_DIM, context_length=E2_TOKENS
    ).eval()
    torch.manual_seed(SEED)
    multi_head = MultiHeadAttention(
        d_in=E2_DIM, d_out=E2_DIM, context_length=E2_TOKENS, num_heads=E8_HEADS
    ).eval()

    with torch.no_grad():
        _, pesos_por_cabeca = multi_head(embedding(torch.tensor([ids])))

    linhas = []
    for cabeca in range(E8_HEADS):
        pesos = pesos_por_cabeca[0, cabeca]
        caminho = FIGURES_DIR / f"e8-head-{cabeca}.png"
        attention_heatmap(
            pesos, tokens, f"{PHRASE_RIVER}\ncabeca {cabeca} de {E8_HEADS}", caminho
        )
        report.write(f"![cabeca {cabeca}]({caminho.relative_to(RESULTS_PATH.parent)})")
        report.write()

        ultima = pesos[-1]
        linhas.append(
            [
                str(cabeca),
                f"`{tokens[int(ultima.argmax())]!r}`",
                f"{ultima.max().item():.3f}",
                f"{row_entropy(ultima).item():.3f}",
                f"{row_entropy(pesos).mean().item():.3f}",
            ]
        )

    report.table(
        [
            "cabeca",
            f"`{tokens[-1]!r}` olha mais",
            "peso",
            "entropia da ultima linha",
            "entropia media",
        ],
        linhas,
    )

    report.write(
        "As quatro figuras sao triangulares inferiores -- a MultiHeadAttention e "
        "causal, entao a mascara vale em toda cabeca. E as quatro sao "
        "diferentes entre si, o que aqui ainda nao e especializacao: com os "
        "pesos recem-sorteados, o que a tabela mostra e que as cabecas COMECAM "
        "diferentes. Sem isso elas convergiriam para a mesma funcao e as quatro "
        "seriam uma so."
    )
    report.write()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(SEED)
    bpe = BPETokenizer()

    report = Report()
    report.write("# Sprint 3 - Atencao: resultados dos experimentos")
    report.write()
    report.write(
        "Gerado por `experimentos/sprint03/atencao_experimentos.py`. "
        "Nao editar a mao: rode o script novamente."
    )
    report.write()
    report.write(
        f"Todas as camadas em `eval()` e sob `torch.no_grad()`, com duas "
        "excecoes declaradas: o bloco do E1, que precisa do backward, e a "
        "tabela de dropout do E5, que so existe em `train()`. Seed fixa em "
        f"{SEED} antes de cada tensor sorteado. Figuras em "
        "`resultados/figuras/`."
    )
    report.write()

    experiment_1_scale(report)
    experiment_2_cost(report)
    experiment_3_visualization(report, bpe)
    experiment_4_sequences(report, bpe)
    experiment_5_mask(report, bpe)
    experiment_6_num_heads(report)
    experiment_7_head_dim(report)
    experiment_8_one_head_against_four(report, bpe)

    report.save(RESULTS_PATH)
    print(f"\nResultados salvos em {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figuras salvas em {FIGURES_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
