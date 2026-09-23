"""Testes da mascara causal e da CausalAttention.

O que fica protegido aqui:

- A mascara bloqueia o futuro e SO o futuro: diagonal livre, T(T-1)/2 posicoes
  bloqueadas.
- As duas abordagens da secao 3.5.1 coincidem -- zerar depois do softmax e
  renormalizar, contra -inf antes. E o que justifica o pacote so usar a segunda.
- A camada nao vaza o futuro: mexer nos tokens depois de t nao muda a saida ate
  t. Os zeros nos pesos sao o mecanismo; este teste e a consequencia.
- A janela e recortada: sequencia menor usa o canto superior esquerdo, maior
  levanta ValueError.
- O dropout so age em train(); em eval() a saida e deterministica.
- A mascara e buffer: entra no state_dict e fica fora do parameters().

Cuidado ao mexer: o que confere valor de peso roda com dropout=0.0 e em eval().
Em train() o nn.Dropout reescala por 1/(1-p) e as linhas NAO somam 1.

Uso:
    python tests/test_causal_attention.py
    pytest tests/
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import (  # noqa: E402
    EXAMPLE_INPUTS,
    CausalAttention,
    causal_mask,
    scaled_dot_product_attention,
)
from tests.test_self_attention import mascara_triangular  # noqa: E402

SEED = 42


def camada_em_eval(**kwargs) -> CausalAttention:
    """CausalAttention com seed fixa, sem dropout e em modo de inferencia."""
    torch.manual_seed(SEED)
    return CausalAttention(**kwargs).eval()


def test_mascara_bloqueia_so_o_futuro() -> None:
    contexto = 6
    mascara = causal_mask(contexto)

    assert mascara.shape == (contexto, contexto)
    # Bool e nao float: masked_fill exige bool, e ocupa 1 byte em vez de 4.
    assert mascara.dtype == torch.bool

    # A diagonal livre e o que deixa a query em i ler a chave em i.
    assert not mascara.diagonal().any()

    # Triangulo estrito: mais comeria a diagonal.
    assert int(mascara.sum()) == contexto * (contexto - 1) // 2

    # Bate com a mascara refeita do zero nos testes da funcao nucleo.
    assert torch.equal(mascara, mascara_triangular(contexto))


def test_as_duas_abordagens_da_secao_351_coincidem() -> None:
    torch.manual_seed(SEED)
    queries, keys, values = (torch.rand(2, 6, 8) for _ in range(3))
    mascara = causal_mask(6)

    # Abordagem 1: softmax normal, zera acima da diagonal, renormaliza a mao.
    scores = queries @ keys.transpose(-2, -1) / keys.shape[-1] ** 0.5
    pesos_crus = torch.softmax(scores, dim=-1)
    pesos_zerados = pesos_crus.masked_fill(mascara, 0.0)
    pesos_renormalizados = pesos_zerados / pesos_zerados.sum(dim=-1, keepdim=True)

    # Abordagem 2: -inf antes do softmax, que e a do pacote.
    contexto_2, pesos_2 = scaled_dot_product_attention(
        queries, keys, values, mask=mascara
    )

    assert torch.allclose(pesos_renormalizados, pesos_2, atol=1e-6)
    assert torch.allclose(pesos_renormalizados @ values, contexto_2, atol=1e-6)


def test_pesos_acima_da_diagonal_sao_zero_e_linhas_somam_um() -> None:
    contexto = 6
    camada = camada_em_eval(d_in=3, d_out=2, context_length=contexto)
    torch.manual_seed(SEED)
    lote = torch.rand(2, contexto, 3)

    _, pesos = camada(lote)

    mascara = causal_mask(contexto)
    # Igualdade exata, nao allclose: exp(-inf) e zero, nao "quase zero".
    assert torch.equal(
        pesos[:, mascara], torch.zeros(2, int(mascara.sum()))
    )
    # Somar 1 apesar dos zeros e o que dispensa renormalizar.
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(2, contexto))
    # O primeiro token so ve a si mesmo, entao gasta toda a atencao nele.
    assert torch.equal(pesos[:, 0, 0], torch.ones(2))


def test_mudar_o_futuro_nao_muda_o_passado() -> None:
    contexto = 6
    camada = camada_em_eval(d_in=3, d_out=2, context_length=contexto)
    corte = 3

    original = EXAMPLE_INPUTS
    alterada = original.clone()
    alterada[corte + 1 :] = torch.rand(contexto - corte - 1, 3)

    saida_original, _ = camada(original)
    saida_alterada, _ = camada(alterada)

    # Ate o corte, saida identica: prova de que o futuro nao entra na conta.
    assert torch.allclose(
        saida_original[: corte + 1], saida_alterada[: corte + 1], atol=1e-6
    )
    # Depois do corte tem que mudar, senao uma camada que ignora a entrada
    # inteira passaria neste teste.
    assert not torch.allclose(
        saida_original[corte + 1 :], saida_alterada[corte + 1 :], atol=1e-6
    )


def test_recorta_a_mascara_para_sequencias_curtas() -> None:
    camada = camada_em_eval(d_in=3, d_out=2, context_length=8)
    torch.manual_seed(SEED)
    lote = torch.rand(2, 4, 3)

    contexto, pesos = camada(lote)

    # A mascara continua 8x8; quem encolhe e o recorte dentro do forward.
    assert camada.mask.shape == (8, 8)
    assert contexto.shape == (2, 4, 2)
    assert pesos.shape == (2, 4, 4)
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(2, 4))


def test_sequencia_maior_que_a_janela_levanta_erro() -> None:
    camada = camada_em_eval(d_in=3, d_out=2, context_length=8)
    torch.manual_seed(SEED)
    lote = torch.rand(2, 9, 3)

    try:
        camada(lote)
    except ValueError as erro:
        assert "context_length" in str(erro)
    else:
        raise AssertionError("9 tokens numa janela de 8 deveria levantar ValueError")


def test_dropout_so_age_em_train() -> None:
    torch.manual_seed(SEED)
    camada = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.5)

    camada.eval()
    saida_1, _ = camada(EXAMPLE_INPUTS)
    saida_2, _ = camada(EXAMPLE_INPUTS)
    # Inferencia nao pode sortear nada: mesma entrada, mesma saida.
    assert torch.equal(saida_1, saida_2)

    camada.train()
    torch.manual_seed(SEED)
    saida_treino, pesos_treino = camada(EXAMPLE_INPUTS)
    assert not torch.allclose(saida_treino, saida_1, atol=1e-6)

    # Em train() o dropout zera pesos e escala o resto por 1/(1-p): as linhas
    # deixam de somar 1. E a pegadinha de inspecionar pesos sem eval().
    assert not torch.allclose(pesos_treino.sum(dim=-1), torch.ones(6))


def test_mascara_e_buffer_nao_parametro() -> None:
    camada = camada_em_eval(d_in=3, d_out=2, context_length=6)

    # No state_dict: a janela viaja com o checkpoint e com .to(device).
    assert "mask" in camada.state_dict()

    # Fora do parameters(): nao ha gradiente num triangulo de booleanos.
    parametros = list(camada.parameters())
    assert all(parametro.shape != camada.mask.shape for parametro in parametros)

    # So os tres pesos de projecao, sem bias (qkv_bias=False).
    assert len(parametros) == 3
    assert sum(parametro.numel() for parametro in parametros) == 3 * 3 * 2


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
