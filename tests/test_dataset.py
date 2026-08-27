"""Testes da Sprint 2: da sequencia de Token IDs aos lotes (src/dataset/).

Cobrem as afirmacoes sobre a janela deslizante -- que o alvo e a entrada
deslocada de exatamente uma posicao, que o numero de amostras e uma funcao
fechada de (len(ids), context_length, stride) e nao depende de gerar as
amostras, que stride controla sozinho a sobreposicao entre amostras vizinhas,
e que texto curto demais nao e erro, e sim zero amostras. Se algum destes
testes cair, uma dessas afirmacoes deixou de ser verdadeira.

Uso:
    python tests/test_dataset.py
    pytest tests/
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (  # noqa: E402
    GPTDataset,
    create_dataloader,
    expected_sample_count,
    sliding_window_pairs,
)
from src.tokenizer import SimpleTokenizer, Vocabulary  # noqa: E402

SEED = 42


class SequentialTokenizer:
    """Tokenizador falso que numera os caracteres do texto, um ID por caractere.

    Existe para testar a janela deslizante sem depender do BPE: com IDs iguais
    a posicao, da para conferir o recorte a olho -- a janela que comeca em 8 e
    exatamente [8, 9, 10, 11]. Tambem prova o contrato de injecao do dataset,
    que so exige o metodo encode.
    """

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


FAKE_TEXT = "x" * 100
FAKE_IDS = list(range(100))


# --------------------------------------------------------------------------
# sequences.py -- janela deslizante em Python puro
# --------------------------------------------------------------------------


def test_alvo_e_entrada_deslocada_de_uma_posicao() -> None:
    for entrada, alvo in sliding_window_pairs(FAKE_IDS, context_length=4, stride=4):
        assert entrada[1:] == alvo[:-1]
        assert alvo[-1] == entrada[-1] + 1


def test_numero_de_amostras_bate_com_expected_sample_count() -> None:
    combinacoes = [
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
        (8, 3), (8, 8), (10, 7), (1, 1), (100, 1),
    ]

    for context_length, stride in combinacoes:
        pares = sliding_window_pairs(FAKE_IDS, context_length, stride)
        esperado = expected_sample_count(len(FAKE_IDS), context_length, stride)
        assert len(pares) == esperado, f"context_length={context_length}, stride={stride}"


def test_stride_igual_ao_contexto_nao_sobrepoe() -> None:
    pares = sliding_window_pairs(FAKE_IDS, context_length=4, stride=4)

    vistos: list[int] = []
    for entrada, _ in pares:
        vistos.extend(entrada)

    assert len(vistos) == len(set(vistos)), "stride == contexto nao deveria repetir token"


def test_stride_menor_que_o_contexto_sobrepoe() -> None:
    pares = sliding_window_pairs(FAKE_IDS, context_length=4, stride=1)

    primeira = set(pares[0][0])
    segunda = set(pares[1][0])

    assert len(primeira & segunda) == 3


def test_texto_curto_demais_produz_zero_amostras() -> None:
    assert expected_sample_count(id_count=3, context_length=4, stride=1) == 0
    assert sliding_window_pairs([0, 1, 2], context_length=4, stride=1) == []


def test_context_length_invalido_levanta_erro() -> None:
    try:
        expected_sample_count(id_count=100, context_length=0, stride=1)
    except ValueError:
        return
    raise AssertionError("esperava ValueError para context_length zero")


def test_stride_invalido_levanta_erro() -> None:
    try:
        expected_sample_count(id_count=100, context_length=4, stride=0)
    except ValueError:
        return
    raise AssertionError("esperava ValueError para stride zero")


# --------------------------------------------------------------------------
# loader.py -- GPTDataset (tensores) e create_dataloader (lotes)
# --------------------------------------------------------------------------


def test_dataset_concorda_com_a_janela_deslizante_em_python_puro() -> None:
    dataset = GPTDataset(FAKE_TEXT, SequentialTokenizer(), context_length=4, stride=3)
    pares = sliding_window_pairs(FAKE_IDS, context_length=4, stride=3)

    assert len(dataset) == len(pares)
    for indice, (entrada_esperada, alvo_esperado) in enumerate(pares):
        entrada, alvo = dataset[indice]
        assert entrada.tolist() == entrada_esperada
        assert alvo.tolist() == alvo_esperado


def test_dataset_texto_curto_demais_fica_com_tamanho_zero() -> None:
    dataset = GPTDataset("xxxx", SequentialTokenizer(), context_length=4, stride=1)
    assert len(dataset) == 0


def test_indice_fora_do_intervalo_levanta_erro() -> None:
    dataset = GPTDataset(FAKE_TEXT, SequentialTokenizer(), context_length=4, stride=4)
    try:
        dataset[len(dataset)]
    except IndexError:
        return
    raise AssertionError("esperava IndexError para janela inexistente")


def test_amostra_sai_como_par_de_tensores_inteiros() -> None:
    dataset = GPTDataset(FAKE_TEXT, SequentialTokenizer(), context_length=4, stride=4)
    entrada, alvo = dataset[0]

    for tensor in (entrada, alvo):
        assert tensor.shape == (4,)
        assert tensor.dtype == torch.long
        assert not tensor.dtype.is_floating_point


def test_tokenizador_entra_por_injecao() -> None:
    texto = "o gato mordeu o rato. o rato fugiu do gato."
    tokenizer = SimpleTokenizer(Vocabulary.from_texts([texto]))

    dataset = GPTDataset(texto, tokenizer, context_length=3, stride=3)
    assert len(dataset) > 0
    assert dataset[0][0].shape == (3,)


def test_lote_sai_com_shape_batch_por_contexto() -> None:
    torch.manual_seed(SEED)
    loader = create_dataloader(
        FAKE_TEXT, SequentialTokenizer(), context_length=4, stride=4, batch_size=8
    )
    entrada, alvo = next(iter(loader))

    assert entrada.shape == (8, 4)
    assert alvo.shape == (8, 4)
    assert entrada.dtype == torch.long
    assert alvo.dtype == torch.long


def test_drop_last_descarta_lote_incompleto() -> None:
    argumentos = dict(
        text=FAKE_TEXT,
        tokenizer=SequentialTokenizer(),
        context_length=4,
        stride=4,
        batch_size=5,
        shuffle=False,
    )

    assert len(list(create_dataloader(**argumentos, drop_last=True))) == 4
    assert len(list(create_dataloader(**argumentos, drop_last=False))) == 5


def test_mesma_seed_reproduz_a_mesma_ordem_de_lotes() -> None:
    def primeira_entrada() -> list[int]:
        torch.manual_seed(SEED)
        loader = create_dataloader(
            FAKE_TEXT, SequentialTokenizer(), context_length=4, stride=4,
            batch_size=8, shuffle=True,
        )
        return next(iter(loader))[0].flatten().tolist()

    assert primeira_entrada() == primeira_entrada()


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
