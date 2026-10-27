"""Testes do TransformerBlock: a integração da atenção com LayerNorm, FFN e atalhos.

O que fica protegido aqui:

- O bloco devolve o shape que recebe -- a condição para empilhar n_layers deles.
- Os parâmetros são exatamente atenção + FeedForward + duas LayerNorms
  (exercício 4.1: no GPT-2 small, 2 360 064 + 4 722 432 + 2 x 1 536).
- Conexão residual: com as duas subcamadas zeradas, o bloco vira a identidade.
  Sem o atalho, sairia zero.
- A causalidade da Sprint 3 sobrevive ao bloco inteiro: LayerNorm e
  FeedForward agem token a token e não abrem caminho para o futuro.
- Dropout: eval() é determinístico, train() não; e as duas taxas do
  exercício 4.3 chegam aos módulos certos.
- O gradiente chega a todos os parâmetros.

Uso:
    python tests/test_transformer_block.py
    pytest tests/
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transformer import TransformerBlock  # noqa: E402

SEED = 42

# Configuração pequena para os testes rodarem rápido; o caso do GPT-2 small
# entra só onde o número do livro é o que se confere.
SMALL = {"emb_dim": 32, "context_length": 16, "n_heads": 4}


def bloco(**kwargs) -> TransformerBlock:
    """Bloco pequeno, seed fixa, em eval()."""
    torch.manual_seed(SEED)
    return TransformerBlock(**{**SMALL, **kwargs}).eval()


def test_bloco_preserva_o_shape() -> None:
    camada = bloco()
    torch.manual_seed(SEED)

    for shape in ((2, 6, 32), (1, 16, 32), (6, 32)):
        x = torch.rand(*shape)
        assert camada(x).shape == x.shape


def test_bloco_do_gpt2_small_preserva_o_shape() -> None:
    # Seção 4.5: rand(2, 4, 768) entra e sai com o mesmo shape.
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    camada = TransformerBlock(emb_dim=768, context_length=1024, n_heads=12, drop_rate=0.1)

    assert camada(x).shape == (2, 4, 768)


def test_parametros_sao_atencao_mais_ffn_mais_normas() -> None:
    torch.manual_seed(SEED)
    camada = TransformerBlock(emb_dim=768, context_length=1024, n_heads=12)

    atencao = sum(p.numel() for p in camada.att.parameters())
    feed_forward = sum(p.numel() for p in camada.ff.parameters())
    normas = sum(p.numel() for n in (camada.norm1, camada.norm2) for p in n.parameters())

    # Exercício 4.1: a FeedForward tem praticamente o dobro da atenção.
    assert atencao == 2_360_064
    assert feed_forward == 4_722_432
    assert normas == 2 * 2 * 768

    total = sum(p.numel() for p in camada.parameters())
    assert total == atencao + feed_forward + normas == 7_085_568


def test_com_subcamadas_zeradas_o_bloco_e_a_identidade() -> None:
    camada = bloco()

    # Zerar a última projeção de cada ramo faz atenção e FeedForward
    # devolverem zero. O que sobra é o atalho: x + 0 + 0.
    with torch.no_grad():
        for linear in (camada.att.out_proj, camada.ff.layers[2]):
            linear.weight.zero_()
            linear.bias.zero_()

    torch.manual_seed(SEED)
    x = torch.randn(2, 6, 32)
    assert torch.equal(camada(x), x)


def test_causalidade_sobrevive_ao_bloco() -> None:
    camada = bloco()
    torch.manual_seed(SEED)
    x = torch.randn(1, 8, 32)
    corte = 4

    alterado = x.clone()
    alterado[0, corte + 1 :] = torch.randn(8 - corte - 1, 32)

    saida, saida_alterada = camada(x), camada(alterado)

    # Até o corte, nada muda: a posição t não lê t+1 em nenhuma subcamada.
    assert torch.allclose(saida[0, : corte + 1], saida_alterada[0, : corte + 1], atol=1e-6)
    assert not torch.allclose(saida[0, corte + 1 :], saida_alterada[0, corte + 1 :], atol=1e-6)


def test_sequencia_maior_que_o_contexto_levanta_erro() -> None:
    camada = bloco()
    try:
        camada(torch.randn(1, SMALL["context_length"] + 1, 32))
    except ValueError as erro:
        assert "context_length" in str(erro)
    else:
        raise AssertionError("o bloco aceitou sequência maior que a máscara")


def test_dropout_so_age_em_train() -> None:
    camada = bloco(drop_rate=0.5)
    torch.manual_seed(SEED)
    x = torch.randn(2, 6, 32)

    # eval(): duas passadas, mesmo resultado bit a bit.
    assert torch.equal(camada(x), camada(x))

    camada.train()
    assert not torch.equal(camada(x), camada(x))


def test_taxas_de_dropout_separadas() -> None:
    # Exercício 4.3: o dropout da atenção e o do atalho são independentes.
    camada = bloco(drop_rate=0.2, attn_drop_rate=0.05)
    assert camada.drop_shortcut.p == 0.2
    assert camada.att.dropout.p == 0.05

    # Sem a taxa da atenção, ela herda a global, como no livro.
    camada = bloco(drop_rate=0.1)
    assert camada.att.dropout.p == camada.drop_shortcut.p == 0.1


def test_gradiente_chega_a_todos_os_parametros() -> None:
    camada = bloco().train()
    torch.manual_seed(SEED)
    x = torch.randn(2, 6, 32, requires_grad=True)

    camada(x).pow(2).mean().backward()

    for nome, parametro in camada.named_parameters():
        assert parametro.grad is not None, f"{nome} ficou sem gradiente"
        assert parametro.grad.abs().sum() > 0, f"{nome} recebeu gradiente nulo"

    # E a entrada também recebe: é por ela que o gradiente passa ao bloco anterior.
    assert x.grad is not None and x.grad.abs().sum() > 0
    # A máscara causal continua sendo buffer, sem gradiente.
    assert camada.att.mask.grad is None


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
