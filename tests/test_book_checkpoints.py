"""Checkpoints do livro: a implementação do projeto reproduz os números do capítulo 4.

Cada teste refaz um exemplo do livro com a implementação de src/ e compara com
o valor impresso, guardado em tests/book_reference.py com a página. Passar aqui
quer dizer mais que "o shape está certo": com a mesma seed, os pesos sorteados,
a ordem das operações e o resultado são os do livro.

O que fica protegido aqui:

- 4.1: o lote de "Every effort moves you" / "Every day holds a" e os logits do
  esqueleto (DummyGPTModel = GPTModel sem blocos e sem a norma final).
- 4.2: a saída da camada de exemplo, a normalização passo a passo e a LayerNorm.
- 4.4: o gradiente por camada da rede de demonstração, com e sem atalho.
- 4.5 e exercício 4.1: shape do bloco e parâmetros da atenção e da FeedForward.
- 4.6 e exercício 4.2: logits do GPTModel de 124M com seed 123, contagem de
  parâmetros, weight tying e tamanho em MB.
- 4.7: a geração gulosa de "Hello, I am", token a token.
- 5.3 e exercício 5.1: argmax, frequências de amostragem, temperatura e top-k no
  vocabulário de 9 palavras.

O GPTModel de 124M é montado uma vez (cache) e ocupa ~650 MB: este arquivo é o
mais lento da pasta, na casa de segundos.

Uso:
    python tests/test_book_checkpoints.py
    pytest tests/
"""

import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    LayerNorm,
    ShortcutDemoNetwork,
    TransformerBlock,
    mean_weight_gradients,
)
from tests import book_reference as livro  # noqa: E402


def lote_do_livro() -> torch.Tensor:
    tokenizer = BPETokenizer()
    return torch.tensor([tokenizer.encode(texto) for texto in livro.BATCH_TEXTS])


def confere_logits(logits: torch.Tensor, esperado: list) -> None:
    """Compara os três primeiros e os três últimos valores de cada linha."""
    for exemplo, linhas in enumerate(esperado):
        for posicao, (inicio, fim) in enumerate(linhas):
            linha = logits[exemplo, posicao]
            assert torch.allclose(linha[:3], torch.tensor(inicio), atol=livro.PRINT_TOLERANCE)
            assert torch.allclose(linha[-3:], torch.tensor(fim), atol=livro.PRINT_TOLERANCE)


@lru_cache(maxsize=1)
def gpt_124m_do_livro() -> tuple[GPTModel, torch.Tensor]:
    """O GPTModel da seção 4.6 e os logits da primeira passada.

    A passada é feita em train(), como no livro: o modelo recém-criado está em
    modo de treino e o dropout de 0,1 participa dos logits impressos. A
    geração da seção 4.7 usa este mesmo modelo, já em eval().
    """
    torch.manual_seed(livro.BOOK_SEED)
    modelo = GPTModel(GPT_CONFIG_124M)
    with torch.no_grad():
        logits = modelo(lote_do_livro())
    return modelo, logits


def test_4_1_lote_de_entrada() -> None:
    assert lote_do_livro().tolist() == livro.BATCH_IDS


def test_4_1_esqueleto_dummy_gpt() -> None:
    # O DummyGPTModel é o GPTModel sem blocos e com uma LayerNorm falsa. Zero
    # blocos não sorteiam nada, então a ordem dos pesos (tokens, posições,
    # saída) é a mesma do livro.
    torch.manual_seed(livro.BOOK_SEED)
    esqueleto = GPTModel(replace(GPT_CONFIG_124M, n_layers=0))
    esqueleto.final_norm = nn.Identity()

    with torch.no_grad():
        logits = esqueleto(lote_do_livro())

    assert logits.shape == (2, 4, 50_257)
    confere_logits(logits, livro.DUMMY_LOGITS)


def test_4_2_normalizacao_passo_a_passo() -> None:
    torch.manual_seed(livro.BOOK_SEED)
    entrada = torch.randn(2, 5)
    camada = nn.Sequential(nn.Linear(5, 6), nn.ReLU())

    with torch.no_grad():
        saida = camada(entrada)
    tol = livro.PRINT_TOLERANCE

    assert torch.allclose(saida, torch.tensor(livro.LAYER_OUTPUT), atol=tol)

    media = saida.mean(dim=-1, keepdim=True)
    variancia = saida.var(dim=-1, keepdim=True)  # n - 1, como no texto
    assert torch.allclose(media.squeeze(), torch.tensor(livro.LAYER_OUTPUT_MEAN), atol=tol)
    assert torch.allclose(variancia.squeeze(), torch.tensor(livro.LAYER_OUTPUT_VAR), atol=tol)

    normalizada = (saida - media) / torch.sqrt(variancia)
    assert torch.allclose(normalizada, torch.tensor(livro.LAYER_OUTPUT_NORMALIZED), atol=tol)

    # A classe: média 0 e variância (enviesada) 1 sobre o lote de entrada.
    with torch.no_grad():
        normalizado = LayerNorm(emb_dim=5)(entrada)
    assert torch.allclose(normalizado.mean(dim=-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalizado.var(dim=-1, unbiased=False), torch.ones(2), atol=1e-4)


def test_4_3_minimo_da_gelu() -> None:
    x = torch.linspace(-3, 3, 100_001)
    minimo = x[GELU()(x).argmin()].item()
    assert abs(minimo - livro.GELU_MINIMUM_X) < 0.01


def test_4_4_gradientes_com_e_sem_atalho() -> None:
    entrada = torch.tensor(livro.SHORTCUT_INPUT)
    casos = {
        False: livro.GRADIENTS_WITHOUT_SHORTCUT,
        True: livro.GRADIENTS_WITH_SHORTCUT,
    }

    for usa_atalho, esperado in casos.items():
        torch.manual_seed(livro.BOOK_SEED)
        rede = ShortcutDemoNetwork(livro.SHORTCUT_LAYER_SIZES, use_shortcut=usa_atalho)
        obtido = mean_weight_gradients(rede, entrada)

        # Tolerância relativa: os valores vão de 1e-4 a 1.
        assert torch.allclose(torch.tensor(obtido), torch.tensor(esperado), rtol=1e-5, atol=0)


def test_4_5_bloco_e_exercicio_4_1() -> None:
    torch.manual_seed(livro.BOOK_SEED)
    x = torch.rand(*livro.BLOCK_INPUT_SHAPE)
    cfg = GPT_CONFIG_124M
    bloco = TransformerBlock(
        emb_dim=cfg.emb_dim,
        context_length=cfg.context_length,
        n_heads=cfg.n_heads,
        drop_rate=cfg.drop_rate,
    )

    with torch.no_grad():
        assert bloco(x).shape == livro.BLOCK_INPUT_SHAPE

    assert sum(p.numel() for p in bloco.ff.parameters()) == livro.FEED_FORWARD_PARAMS
    assert sum(p.numel() for p in bloco.att.parameters()) == livro.ATTENTION_PARAMS


def test_4_6_logits_do_gpt_124m() -> None:
    _, logits = gpt_124m_do_livro()
    assert logits.shape == (2, 4, 50_257)
    confere_logits(logits, livro.GPT_LOGITS)


def test_4_6_parametros_e_memoria() -> None:
    modelo, _ = gpt_124m_do_livro()

    assert modelo.parameter_count() == livro.GPT_124M_PARAMS
    assert tuple(modelo.embedding.token_embedding.weight.shape) == livro.EMBEDDING_SHAPE
    assert tuple(modelo.out_head.weight.shape) == livro.EMBEDDING_SHAPE

    sem_cabeca = modelo.parameter_count() - sum(p.numel() for p in modelo.out_head.parameters())
    assert sem_cabeca == livro.GPT_124M_PARAMS_TIED
    assert f"{modelo.size_in_megabytes():.2f}" == livro.GPT_124M_MEGABYTES


def test_exercicio_4_2_gpt2_xl() -> None:
    with torch.device("meta"):
        xl = GPTModel(GPT2_CONFIGS["gpt2-xl"])

    assert xl.parameter_count() == livro.GPT2_XL_PARAMS
    assert xl.parameter_count() - xl.parameter_breakdown()["out_head"] == livro.GPT2_XL_PARAMS_TIED
    assert f"{xl.size_in_megabytes():.2f}" == livro.GPT2_XL_MEGABYTES


def test_4_7_geracao_de_hello_i_am() -> None:
    modelo, _ = gpt_124m_do_livro()
    tokenizer = BPETokenizer()

    ids = text_to_token_ids(livro.GENERATION_PROMPT, tokenizer)
    assert ids.tolist() == [livro.GENERATION_PROMPT_IDS]

    saida = generate(
        modelo,
        ids,
        max_new_tokens=livro.GENERATION_NEW_TOKENS,
        context_length=GPT_CONFIG_124M.context_length,
    )

    assert saida.tolist() == [livro.GENERATION_OUTPUT_IDS]
    assert token_ids_to_text(saida, tokenizer) == livro.GENERATION_OUTPUT_TEXT


def test_5_3_gulosa_e_amostragem() -> None:
    logits = torch.tensor(livro.TOY_LOGITS)

    escolhido = select_next_token(logits.unsqueeze(0)).item()
    assert livro.TOY_VOCAB[escolhido] == livro.TOY_GREEDY_WORD

    # O livro sorteia com a seed global; um Generator com a mesma seed produz a
    # mesma sequência de números, e é o que select_next_token aceita.
    gerador = torch.Generator().manual_seed(livro.BOOK_SEED)
    sorteios = [
        select_next_token(logits.unsqueeze(0), temperature=1.0, generator=gerador).item()
        for _ in range(livro.TOY_SAMPLES)
    ]
    contagem = torch.bincount(torch.tensor(sorteios), minlength=len(livro.TOY_VOCAB))

    for palavra, esperado in livro.TOY_SAMPLE_COUNTS.items():
        assert contagem[livro.TOY_VOCAB.index(palavra)].item() == esperado


def test_exercicio_5_1_pizza_com_temperatura_5() -> None:
    logits = torch.tensor(livro.TOY_LOGITS)
    pizza = livro.TOY_VOCAB.index("pizza")

    probabilidades = next_token_probabilities(logits, temperature=5.0)
    assert abs(probabilidades[pizza].item() - livro.TOY_PIZZA_PROBABILITY_T5) < 5e-4

    # Um sorteio por chamada, como no print_sampled_tokens do livro: a ordem
    # de consumo do gerador é parte do resultado.
    gerador = torch.Generator().manual_seed(livro.BOOK_SEED)
    sorteios = [
        torch.multinomial(probabilidades, 1, generator=gerador).item()
        for _ in range(livro.TOY_SAMPLES)
    ]
    assert sorteios.count(pizza) == livro.TOY_PIZZA_COUNT_T5


def test_5_3_top_k() -> None:
    logits = torch.tensor(livro.TOY_LOGITS)

    valores, posicoes = torch.topk(logits, livro.TOY_TOP_K)
    assert torch.allclose(valores, torch.tensor(livro.TOY_TOP3_LOGITS))
    assert posicoes.tolist() == livro.TOY_TOP3_POSITIONS

    filtrados = apply_top_k(logits, livro.TOY_TOP_K)
    assert int(torch.isfinite(filtrados).sum()) == livro.TOY_TOP_K

    probabilidades = next_token_probabilities(logits, top_k=livro.TOY_TOP_K)
    assert torch.allclose(
        probabilidades, torch.tensor(livro.TOY_TOP3_PROBABILITIES), atol=livro.PRINT_TOLERANCE
    )


def main() -> int:
    """Roda todos os testes deste arquivo sem depender de pytest."""
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]

    failures = 0
    for name, function in tests:
        try:
            function()
        except Exception as error:  # noqa: BLE001 - relatorio, nao tratamento
            failures += 1
            print(f"FALHOU  {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok      {name}")

    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
