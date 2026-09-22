"""Testes da atencao simplificada, da funcao nucleo e da SelfAttention.

O que fica protegido aqui:

- Os pesos sao uma distribuicao: cada linha soma 1, inclusive com mascara --
  prova que masked_fill(-inf) antes do softmax dispensa renormalizar.
- A implementacao reproduz o vetor de contexto do exemplo do capitulo 3,
  conferido a mao antes de existir codigo.
- A funcao nucleo concorda com o PyTorch, com e sem mascara causal.
- A SelfAttention projeta com as matrizes na orientacao certa e e cega a ordem
  dos tokens -- o que justifica os positional embeddings da Sprint 2.

Uso:
    python tests/test_self_attention.py
    pytest tests/
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import (  # noqa: E402
    EXAMPLE_INPUTS,
    EXAMPLE_WORDS,
    SelfAttention,
    scaled_dot_product_attention,
    simplified_attention,
)

SEED = 42

# Valor do vetor de contexto de "journey" (indice 1) no exemplo do capitulo 3.
# Serve de referencia externa: foi conferido no calculo a mao, entao se esta
# linha deixar de bater a implementacao mudou, nao a conta.
Z2_ESPERADO = torch.tensor([0.4419, 0.6515, 0.5683])


def mascara_triangular(context_length: int) -> torch.Tensor:
    """Mascara causal na convencao do projeto: True acima da diagonal.

    True = bloqueado, entao a query em i nao enxerga chave em j > i. A versao
    definitiva, com buffer registrado, vem em src/attention/causal.py.
    """
    return torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)


def test_linhas_dos_pesos_somam_um() -> None:
    _, pesos = simplified_attention(EXAMPLE_INPUTS)

    assert pesos.shape == (len(EXAMPLE_WORDS), len(EXAMPLE_WORDS))
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(len(EXAMPLE_WORDS)))


def test_linhas_continuam_somando_um_com_mascara() -> None:
    contexto = len(EXAMPLE_WORDS)
    _, pesos = scaled_dot_product_attention(
        EXAMPLE_INPUTS,
        EXAMPLE_INPUTS,
        EXAMPLE_INPUTS,
        mask=mascara_triangular(contexto),
    )

    # A primeira linha ve so a si mesma, a ultima ve tudo -- todas somam 1.
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(contexto))
    assert pesos[0, 0] == 1.0


def test_posicoes_bloqueadas_recebem_peso_zero() -> None:
    contexto = len(EXAMPLE_WORDS)
    mascara = mascara_triangular(contexto)

    _, pesos = scaled_dot_product_attention(
        EXAMPLE_INPUTS, EXAMPLE_INPUTS, EXAMPLE_INPUTS, mask=mascara
    )

    assert torch.equal(pesos[mascara], torch.zeros(int(mascara.sum())))


def test_reproduz_o_vetor_de_contexto_do_livro() -> None:
    contexto, _ = simplified_attention(EXAMPLE_INPUTS)

    posicao = 1
    assert EXAMPLE_WORDS[posicao] == "journey"
    assert torch.allclose(contexto[posicao], Z2_ESPERADO, atol=1e-4), (
        f"contexto de {EXAMPLE_WORDS[posicao]!r} deu {contexto[posicao]}, "
        f"esperado {Z2_ESPERADO}"
    )


def test_scores_sao_simetricos_mas_os_pesos_nao() -> None:
    scores = EXAMPLE_INPUTS @ EXAMPLE_INPUTS.T
    _, pesos = simplified_attention(EXAMPLE_INPUTS)

    # Sem W_q e W_k separados, os dois sentidos sao o mesmo produto escalar.
    assert torch.allclose(scores, scores.T)

    # Mas o softmax normaliza linha a linha, e a simetria nao sobrevive.
    assert not torch.allclose(pesos, pesos.T)


def test_bate_com_o_pytorch_sem_mascara() -> None:
    # Baseline: o unico uso que o README permite para o pronto do PyTorch.
    torch.manual_seed(SEED)
    queries, keys, values = (torch.rand(2, 6, 8) for _ in range(3))

    nosso, _ = scaled_dot_product_attention(queries, keys, values)
    pytorch = F.scaled_dot_product_attention(queries, keys, values)

    assert torch.allclose(nosso, pytorch, atol=1e-6)


def test_bate_com_o_pytorch_com_mascara_causal() -> None:
    torch.manual_seed(SEED)
    queries, keys, values = (torch.rand(2, 6, 8) for _ in range(3))

    nosso, _ = scaled_dot_product_attention(
        queries, keys, values, mask=mascara_triangular(6)
    )
    # is_causal=True em vez da nossa mascara: no attn_mask bool do PyTorch
    # True e "pode ver", o inverso daqui. Passar a nossa direto nao daria
    # erro -- daria a atencao anticausal.
    pytorch = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)

    assert torch.allclose(nosso, pytorch, atol=1e-6)


def test_a_mascara_do_pytorch_e_a_inversa_da_nossa() -> None:
    torch.manual_seed(SEED)
    queries, keys, values = (torch.rand(2, 6, 8) for _ in range(3))
    nossa_mascara = mascara_triangular(6)

    nosso, _ = scaled_dot_product_attention(queries, keys, values, mask=nossa_mascara)
    pytorch = F.scaled_dot_product_attention(
        queries, keys, values, attn_mask=~nossa_mascara
    )

    # Negar a mascara e o que iguala as duas chamadas. Em teste para que a
    # inversao nao seja redescoberta depurando.
    assert torch.allclose(nosso, pytorch, atol=1e-6)


def test_funciona_com_eixo_de_cabecas() -> None:
    torch.manual_seed(SEED)
    queries, keys, values = (torch.rand(2, 4, 6, 8) for _ in range(3))

    # Mesma funcao, um eixo a mais: o que a multi-head vai usar.
    contexto, pesos = scaled_dot_product_attention(
        queries, keys, values, mask=mascara_triangular(6)
    )

    assert contexto.shape == (2, 4, 6, 8)
    assert pesos.shape == (2, 4, 6, 6)
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(2, 4, 6))


def test_self_attention_devolve_d_out() -> None:
    d_in, d_out = 3, 2
    torch.manual_seed(SEED)
    camada = SelfAttention(d_in=d_in, d_out=d_out)
    lote = torch.rand(2, 6, d_in)

    contexto, pesos = camada(lote)

    assert contexto.shape == (2, 6, d_out)
    # Os pesos sao T x T qualquer que seja d_out: a dimensao projetada some
    # no produto escalar, sobra token contra token.
    assert pesos.shape == (2, 6, 6)
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(2, 6))


def test_projecao_equivale_a_multiplicar_pela_matriz_transposta() -> None:
    # d_in != d_out de proposito: iguais, esquecer o .T daria numero errado
    # em silencio em vez de erro de shape.
    d_in, d_out = 3, 2
    torch.manual_seed(SEED)
    camada = SelfAttention(d_in=d_in, d_out=d_out)
    entradas = EXAMPLE_INPUTS

    # Exercicio 3.1: nn.Linear guarda o peso como (d_out, d_in).
    assert camada.W_query.weight.shape == (d_out, d_in)

    # Refeita do zero: comparar a funcao com ela mesma nao provaria nada.
    queries = entradas @ camada.W_query.weight.T
    keys = entradas @ camada.W_key.weight.T
    values = entradas @ camada.W_value.weight.T
    scores = queries @ keys.T
    esperado = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1) @ values

    contexto, _ = camada(entradas)

    assert torch.allclose(contexto, esperado, atol=1e-6)


def test_permutar_a_entrada_permuta_a_saida() -> None:
    torch.manual_seed(SEED)
    camada = SelfAttention(d_in=EXAMPLE_INPUTS.shape[1], d_out=2)

    # Fixa, nao randperm: assim o teste falha sempre do mesmo jeito.
    perm = torch.tensor([3, 0, 5, 1, 4, 2])

    contexto, pesos = camada(EXAMPLE_INPUTS)
    contexto_perm, pesos_perm = camada(EXAMPLE_INPUTS[perm])

    # A atencao e EQUIVARIANTE a permutacao: embaralhar os tokens embaralha a
    # saida na mesma ordem, sem mudar valor nenhum. Ela nao enxerga ordem. O
    # que separa "The bank of the river" de "river the of bank The" e o vetor
    # de posicao somado em src/embeddings/positional.py, antes da atencao.
    assert torch.allclose(contexto[perm], contexto_perm, atol=1e-6)

    # Nos pesos a permutacao vale nos DOIS eixos: queries e keys.
    assert torch.allclose(pesos[perm][:, perm], pesos_perm, atol=1e-6)


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
