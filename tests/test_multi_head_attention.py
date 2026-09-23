"""Testes do split em cabecas, do MultiHeadAttentionWrapper e da MultiHeadAttention.

Comecou pelo teste de ida e volta do reshape, escrito antes de multi_head.py
existir: view + transpose na dimensao errada tem o mesmo numel, entao passa sem
erro e devolve numero errado em silencio.

O que fica protegido aqui:

- O ida e volta (B, T, d_out) -> (B, H, T, head_dim) -> (B, T, d_out) devolve o
  tensor original, e a cabeca h e o bloco de colunas
  [h*head_dim : (h+1)*head_dim] -- a correspondencia que autoriza o teste de
  equivalencia a concatenar os pesos das cabecas em dim=0.
- O .contiguous() nao e decoracao e o view direto nao substitui o transpose.
- As duas implementacoes dao o MESMO resultado com os mesmos pesos: o weight
  split e o calculo do Wrapper reorganizado, nao outro modelo.
- O numero de parametros nao depende de num_heads.
- d_out indivisivel por num_heads falha na construcao, nao no forward.
- A configuracao do GPT-2 small (exercicio 3.3) instancia e roda.
- A mascara causal sobrevive ao split: nenhuma cabeca enxerga o futuro.

Uso:
    python tests/test_multi_head_attention.py
    pytest tests/
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import (  # noqa: E402
    EXAMPLE_INPUTS,
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
)

SEED = 42

# Configuracao do exercicio 3.3: a Multi-Head do menor GPT-2.
GPT2_SMALL = {
    "d_in": 768,
    "d_out": 768,
    "context_length": 1024,
    "num_heads": 12,
    "dropout": 0.1,
    "qkv_bias": False,
}


def camadas_em_eval(**kwargs) -> tuple[MultiHeadAttentionWrapper, MultiHeadAttention]:
    """As duas implementacoes com a MESMA seed, sem dropout e em inferencia.

    Seed igual nao faz os pesos coincidirem (a ordem de criacao dos tensores e
    outra em cada classe); quem iguala e o teste de equivalencia.
    """
    torch.manual_seed(SEED)
    wrapper = MultiHeadAttentionWrapper(**kwargs).eval()
    torch.manual_seed(SEED)
    multi_head = MultiHeadAttention(**kwargs).eval()
    return wrapper, multi_head


def test_ida_e_volta_do_split_em_heads() -> None:
    lote, tokens, cabecas, head_dim = 2, 6, 4, 3
    d_out = cabecas * head_dim

    torch.manual_seed(SEED)
    original = torch.rand(lote, tokens, d_out)

    # Ida: separa as colunas em cabecas e troca T com H, para as duas ultimas
    # dimensoes voltarem a ser (tokens, features).
    dividido = original.view(lote, tokens, cabecas, head_dim).transpose(1, 2)
    assert dividido.shape == (lote, cabecas, tokens, head_dim)

    # Volta: desfaz o transpose e junta as cabecas de novo.
    voltou = dividido.transpose(1, 2).contiguous().view(lote, tokens, d_out)

    # Exata e nao allclose: nao houve conta, so remanejo de indices.
    assert torch.equal(original, voltou)


def test_cada_cabeca_e_um_bloco_de_colunas() -> None:
    lote, tokens, cabecas, head_dim = 2, 6, 4, 3
    d_out = cabecas * head_dim

    torch.manual_seed(SEED)
    original = torch.rand(lote, tokens, d_out)
    dividido = original.view(lote, tokens, cabecas, head_dim).transpose(1, 2)

    for cabeca in range(cabecas):
        fatia = slice(cabeca * head_dim, (cabeca + 1) * head_dim)
        # Colunas CONTIGUAS, na ordem: e por isso que empilhar os pesos das
        # cabecas em dim=0 reproduz a projecao unica do weight split.
        assert torch.equal(dividido[:, cabeca], original[..., fatia])


def test_o_contiguous_e_o_view_direto_nao_sao_opcionais() -> None:
    lote, tokens, cabecas, head_dim = 2, 6, 4, 3
    d_out = cabecas * head_dim

    torch.manual_seed(SEED)
    original = torch.rand(lote, tokens, d_out)
    dividido = original.view(lote, tokens, cabecas, head_dim).transpose(1, 2)

    # O transpose so mexe nos strides: mesmo buffer, outra ordem de leitura.
    assert not dividido.is_contiguous()

    # Pegadinha: desfazer o transpose devolve o layout original, contiguo, entao
    # aqui o .contiguous() seria no-op. No forward e outro tensor -- a SAIDA da
    # atencao, nova como (B, H, T, head_dim) -- e ai o transpose quebra mesmo.
    assert dividido.transpose(1, 2).is_contiguous()

    torch.manual_seed(SEED)
    saida_da_atencao = torch.rand(lote, cabecas, tokens, head_dim)
    assert not saida_da_atencao.transpose(1, 2).is_contiguous()
    try:
        saida_da_atencao.transpose(1, 2).view(lote, tokens, d_out)
    except RuntimeError:
        pass
    else:
        raise AssertionError("view sobre tensor nao contiguo deveria falhar")

    # O atalho tentador: mesmo numel, view legal, numeros embaralhados. Falha
    # silenciosa, nao erro de shape.
    atalho = original.view(lote, cabecas, tokens, head_dim)
    assert atalho.shape == dividido.shape
    assert not torch.equal(atalho, dividido)


def test_shapes_das_duas_classes() -> None:
    lote, tokens, d_in, d_out, cabecas = 2, 6, 4, 12, 4
    wrapper, multi_head = camadas_em_eval(
        d_in=d_in, d_out=d_out, context_length=8, num_heads=cabecas
    )
    torch.manual_seed(SEED)
    entrada = torch.rand(lote, tokens, d_in)

    for camada in (wrapper, multi_head):
        contexto, pesos = camada(entrada)

        # d_out e o TOTAL: o concat ja devolve d_out, sem multiplicar por H.
        assert contexto.shape == (lote, tokens, d_out)
        assert pesos.shape == (lote, cabecas, tokens, tokens)
        assert torch.allclose(pesos.sum(dim=-1), torch.ones(lote, cabecas, tokens))

        # Entrada sem lote, como as outras tres classes do pacote aceitam.
        contexto_2d, pesos_2d = camada(entrada[0])
        assert contexto_2d.shape == (tokens, d_out)
        assert pesos_2d.shape == (cabecas, tokens, tokens)
        assert torch.allclose(contexto_2d, contexto[0], atol=1e-6)


def test_d_out_indivisivel_levanta_erro() -> None:
    for classe in (MultiHeadAttentionWrapper, MultiHeadAttention):
        try:
            # 6 colunas em 4 cabecas dariam cabecas de tamanhos diferentes.
            classe(d_in=4, d_out=6, context_length=8, num_heads=4)
        except ValueError as erro:
            assert "num_heads" in str(erro)
        else:
            raise AssertionError(f"{classe.__name__} aceitou d_out indivisivel")


def test_weight_split_equivale_ao_wrapper() -> None:
    lote, tokens, d_in, d_out, cabecas = 2, 6, 4, 12, 4
    wrapper, multi_head = camadas_em_eval(
        d_in=d_in, d_out=d_out, context_length=8, num_heads=cabecas
    )

    with torch.no_grad():
        for projecao in ("W_query", "W_key", "W_value"):
            # dim=0 porque nn.Linear guarda o peso como (out_features, in_features):
            # empilhar ali poe cada cabeca num bloco contiguo, na ordem do view.
            empilhado = torch.cat(
                [getattr(cabeca, projecao).weight for cabeca in wrapper.heads], dim=0
            )
            getattr(multi_head, projecao).weight.copy_(empilhado)

        # out_proj como identidade: o Wrapper nao tem esse passo.
        multi_head.out_proj.weight.copy_(torch.eye(d_out))
        multi_head.out_proj.bias.zero_()

    torch.manual_seed(SEED)
    entrada = torch.rand(lote, tokens, d_in)

    contexto_wrapper, pesos_wrapper = wrapper(entrada)
    contexto_split, pesos_split = multi_head(entrada)

    # Mesmos pesos: as cabecas fazem a mesma pergunta.
    assert torch.allclose(pesos_wrapper, pesos_split, atol=1e-6)
    # E mesmo contexto: o weight split e o laco do Wrapper como um matmul so.
    assert torch.allclose(contexto_wrapper, contexto_split, atol=1e-6)


def test_numero_de_parametros_nao_depende_de_num_heads() -> None:
    d_in, d_out, context_length = 4, 12, 8

    totais = set()
    for cabecas in (1, 2, 4, 6, 12):
        torch.manual_seed(SEED)
        camada = MultiHeadAttention(
            d_in=d_in, d_out=d_out, context_length=context_length, num_heads=cabecas
        )
        totais.add(sum(parametro.numel() for parametro in camada.parameters()))

    # A projecao continua d_in -> d_out, so a leitura das colunas muda: mais
    # cabecas com o mesmo d_out da cabecas MENORES, nao rede maior.
    assert len(totais) == 1
    assert totais.pop() == 3 * d_in * d_out + d_out * d_out + d_out

    torch.manual_seed(SEED)
    wrapper = MultiHeadAttentionWrapper(
        d_in=d_in, d_out=d_out, context_length=context_length, num_heads=4
    )
    torch.manual_seed(SEED)
    multi_head = MultiHeadAttention(
        d_in=d_in, d_out=d_out, context_length=context_length, num_heads=4
    )

    parametros_wrapper = sum(p.numel() for p in wrapper.parameters())
    parametros_split = sum(p.numel() for p in multi_head.parameters())

    # A unica diferenca e a out_proj: as projecoes das 4 cabecas somadas dao
    # exatamente a projecao unica do weight split.
    assert parametros_split - parametros_wrapper == d_out * d_out + d_out


def test_configuracao_do_gpt2_small() -> None:
    torch.manual_seed(SEED)
    camada = MultiHeadAttention(**GPT2_SMALL).eval()

    # 768 / 12 = 64, o head_dim do GPT-2.
    assert camada.head_dim == 64
    assert camada.context_length == 1024

    esperado = 3 * 768 * 768 + 768 * 768 + 768  # projecoes + out_proj com bias
    assert sum(p.numel() for p in camada.parameters()) == esperado
    assert esperado == 2_360_064

    # Forward curto so para provar que a instancia roda.
    torch.manual_seed(SEED)
    contexto, pesos = camada(torch.rand(1, 4, 768))
    assert contexto.shape == (1, 4, 768)
    assert pesos.shape == (1, 12, 4, 4)


def test_mascara_causal_continua_valendo() -> None:
    tokens, d_in, d_out, cabecas = 6, 3, 8, 4
    wrapper, multi_head = camadas_em_eval(
        d_in=d_in, d_out=d_out, context_length=tokens, num_heads=cabecas
    )
    corte = 3

    alterada = EXAMPLE_INPUTS.clone()
    torch.manual_seed(SEED)
    alterada[corte + 1 :] = torch.rand(tokens - corte - 1, d_in)

    for camada in (wrapper, multi_head):
        saida, pesos = camada(EXAMPLE_INPUTS)

        # Zero exato acima da diagonal em TODAS as cabecas: o split nao pode ter
        # embaralhado a mascara junto com as colunas.
        acima_da_diagonal = torch.triu(
            torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1
        )
        assert torch.equal(
            pesos[:, acima_da_diagonal],
            torch.zeros(cabecas, int(acima_da_diagonal.sum())),
        )

        saida_alterada, _ = camada(alterada)
        assert torch.allclose(saida[: corte + 1], saida_alterada[: corte + 1], atol=1e-6)
        assert not torch.allclose(
            saida[corte + 1 :], saida_alterada[corte + 1 :], atol=1e-6
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
