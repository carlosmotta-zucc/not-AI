"""Testes das peças do bloco: LayerNorm, GELU e FeedForward.

O que fica protegido aqui:

- LayerNorm: cada vetor sai com média 0 e variância 1; o resultado coincide com
  o nn.LayerNorm (baseline); scale e shift são treináveis e fazem efeito; cada
  token é normalizado sozinho -- nada atravessa posições nem exemplos do lote.
- GELU: a forma por tanh coincide com nn.GELU(approximate="tanh") e a exata com
  nn.GELU(); os negativos não são zerados, como na ReLU.
- FeedForward: o shape volta ao de entrada (é a condição da soma residual), a
  camada escondida é 4x por padrão, e a saída de uma posição não depende das
  outras.

Uso:
    python tests/test_transformer_components.py
    pytest tests/
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transformer import FeedForward, GELU, LayerNorm  # noqa: E402

SEED = 42


def test_layer_norm_zera_media_e_unitariza_variancia() -> None:
    torch.manual_seed(SEED)
    # Escala e deslocamento grandes de propósito: a normalização tem que
    # desfazer os dois.
    x = 5.0 * torch.randn(2, 4, 16) + 3.0

    saida = LayerNorm(emb_dim=16)(x)

    assert saida.shape == x.shape
    assert torch.allclose(saida.mean(dim=-1), torch.zeros(2, 4), atol=1e-5)
    # Variância enviesada, a mesma que a camada usa por dentro.
    assert torch.allclose(
        saida.var(dim=-1, unbiased=False), torch.ones(2, 4), atol=1e-4
    )


def test_layer_norm_coincide_com_nn_layer_norm() -> None:
    torch.manual_seed(SEED)
    x = torch.randn(3, 5, 32)

    nossa = LayerNorm(emb_dim=32)
    baseline = nn.LayerNorm(32, eps=nossa.eps)

    # Scale e shift diferentes de 1 e 0, para a comparação cobrir também a
    # parte afim, e não só a normalização.
    with torch.no_grad():
        nossa.scale.copy_(torch.rand(32) + 0.5)
        nossa.shift.copy_(torch.randn(32))
        baseline.weight.copy_(nossa.scale)
        baseline.bias.copy_(nossa.shift)

    assert torch.allclose(nossa(x), baseline(x), atol=1e-6)


def test_layer_norm_tem_scale_e_shift_treinaveis() -> None:
    camada = LayerNorm(emb_dim=8)

    parametros = dict(camada.named_parameters())
    assert set(parametros) == {"scale", "shift"}
    assert all(p.requires_grad for p in parametros.values())
    # 2 * emb_dim: 1 536 por LayerNorm no GPT-2 small.
    assert sum(p.numel() for p in camada.parameters()) == 2 * 8

    # A parte afim faz efeito: scale 2 e shift 1 levam a média a 1 e o desvio a 2.
    with torch.no_grad():
        camada.scale.fill_(2.0)
        camada.shift.fill_(1.0)

    torch.manual_seed(SEED)
    saida = camada(torch.randn(4, 8))
    assert torch.allclose(saida.mean(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(saida.std(dim=-1, unbiased=False), 2 * torch.ones(4), atol=1e-3)


def test_layer_norm_ignora_escala_e_deslocamento_da_entrada() -> None:
    torch.manual_seed(SEED)
    x = torch.randn(2, 6, 16)
    camada = LayerNorm(emb_dim=16)

    # Multiplicar por um positivo e somar uma constante muda média e desvio na
    # mesma proporção; a normalização desfaz os dois. É o que faz a escala do
    # fluxo residual, que cresce bloco a bloco, não chegar na subcamada.
    assert torch.allclose(camada(x), camada(7.0 * x - 2.0), atol=1e-4)


def test_layer_norm_normaliza_cada_token_sozinho() -> None:
    torch.manual_seed(SEED)
    x = torch.randn(2, 5, 16)
    camada = LayerNorm(emb_dim=16)
    saida = camada(x)

    # Troca o token 3 do exemplo 1 por outra coisa: só aquela linha pode mudar.
    alterado = x.clone()
    alterado[1, 3] = 100 * torch.randn(16)
    saida_alterada = camada(alterado)

    mudou = ~torch.isclose(saida, saida_alterada, atol=1e-6).all(dim=-1)
    esperado = torch.zeros(2, 5, dtype=torch.bool)
    esperado[1, 3] = True
    assert torch.equal(mudou, esperado)


def test_gelu_tanh_coincide_com_nn_gelu() -> None:
    x = torch.linspace(-6, 6, 1001)

    assert torch.allclose(GELU()(x), nn.GELU(approximate="tanh")(x), atol=1e-6)
    assert torch.allclose(GELU(approximate="none")(x), nn.GELU()(x), atol=1e-6)


def test_gelu_nao_zera_os_negativos() -> None:
    gelu = GELU()
    x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 3.0])
    saida = gelu(x)

    assert saida[3] == 0.0
    # Negativos: saída pequena, mas diferente de zero -- a ReLU daria 0.
    assert bool((saida[:3] < 0).all())
    assert abs(saida[1].item() - (-0.1588)) < 1e-3
    # Para x grande, GELU(x) ~ x.
    assert abs(saida[-1].item() - 3.0) < 0.01


def test_gelu_tem_gradiente_nos_negativos() -> None:
    x = torch.tensor([-2.0, -1.0, -0.5], requires_grad=True)
    GELU()(x).sum().backward()

    # A ReLU teria gradiente zero nos três. Perto de -0,75 o gradiente da GELU
    # troca de sinal (é o mínimo da curva), então ele não é zero aqui.
    assert x.grad is not None
    assert bool((x.grad.abs() > 1e-3).all())


def test_gelu_rejeita_aproximacao_desconhecida() -> None:
    try:
        GELU(approximate="sigmoid")
    except ValueError as erro:
        assert "approximate" in str(erro)
    else:
        raise AssertionError("GELU aceitou uma aproximação que não existe")


def test_feed_forward_mantem_o_shape() -> None:
    torch.manual_seed(SEED)
    ffn = FeedForward(emb_dim=64)
    x = torch.rand(2, 3, 64)

    assert ffn(x).shape == (2, 3, 64)
    # O padrão é o do GPT-2: 4x de expansão.
    assert ffn.hidden_dim == 4 * 64

    # Por dentro o shape passa pela camada expandida.
    expandido = ffn.layers[1](ffn.layers[0](x))
    assert expandido.shape == (2, 3, 256)


def test_feed_forward_conta_parametros_pela_formula() -> None:
    for emb_dim, hidden_dim in ((64, None), (64, 128), (768, None)):
        ffn = FeedForward(emb_dim, hidden_dim)
        oculta = ffn.hidden_dim

        # Duas Linear com bias: D·H + H na ida, H·D + D na volta.
        esperado = 2 * emb_dim * oculta + oculta + emb_dim
        assert sum(p.numel() for p in ffn.parameters()) == esperado

    # Exercício 4.1: a FeedForward do GPT-2 small.
    assert sum(p.numel() for p in FeedForward(768).parameters()) == 4_722_432


def test_feed_forward_age_posicao_a_posicao() -> None:
    torch.manual_seed(SEED)
    ffn = FeedForward(emb_dim=16)
    x = torch.randn(1, 6, 16)

    alterado = x.clone()
    alterado[0, 4] = torch.randn(16)

    saida, saida_alterada = ffn(x), ffn(alterado)

    # Só a posição alterada muda: a FeedForward não mistura tokens. Quem
    # mistura é a atenção.
    for posicao in range(6):
        igual = torch.allclose(saida[0, posicao], saida_alterada[0, posicao])
        assert igual == (posicao != 4)


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
