"""Testes do GPTModel e da GPTConfig.

O que fica protegido aqui:

- Contrato do forward: (B, T) long -> (B, T, vocab_size), com T até o
  context_length e erro acima dele.
- A contagem de parâmetros por componente soma o total, e o total do GPT-2
  small e do XL é o do livro (seção 4.6 e exercício 4.2), com e sem weight
  tying. Os modelos grandes são montados no device "meta": shapes sem memória,
  então contar 1,6 bilhão de parâmetros não aloca 6 GB.
- Weight tying: a cabeça de saída É a tabela de tokens, não uma cópia.
- Causalidade no modelo inteiro: o logit da posição t não depende de t+1.
- A configuração rejeita combinações inválidas na construção, e as três taxas
  de dropout do exercício 4.3 chegam aos módulos certos.
- GPT2_VOCAB_SIZE é o vocabulário do BPE de verdade.

Uso:
    python tests/test_gpt_model.py
    pytest tests/
"""

import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import (  # noqa: E402
    BASE_CONFIG,
    COMPONENTS,
    GPT2_CONFIGS,
    GPT2_VOCAB_SIZE,
    GPT_CONFIG_124M,
    GPTConfig,
    GPTModel,
)
from src.tokenizer import BPETokenizer  # noqa: E402

SEED = 42

# Vocabulário e dimensões pequenos: os testes de comportamento não dependem do
# tamanho, e o forward fica instantâneo.
TINY = GPTConfig(
    vocab_size=100,
    context_length=12,
    emb_dim=16,
    n_heads=4,
    n_layers=2,
    drop_rate=0.0,
)


def modelo(cfg: GPTConfig = TINY) -> GPTModel:
    torch.manual_seed(SEED)
    return GPTModel(cfg).eval()


def ids_aleatorios(batch: int, tokens: int, vocab_size: int = TINY.vocab_size) -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randint(0, vocab_size, (batch, tokens))


def no_meta(cfg: GPTConfig) -> GPTModel:
    """Monta o modelo sem alocar os pesos: só os shapes existem."""
    with torch.device("meta"):
        return GPTModel(cfg)


def test_forward_devolve_um_logit_por_token_do_vocabulario() -> None:
    gpt = modelo()

    for tokens in (1, 5, TINY.context_length):
        logits = gpt(ids_aleatorios(3, tokens))
        assert logits.shape == (3, tokens, TINY.vocab_size)
        assert logits.dtype == torch.float32


def test_sequencia_maior_que_o_contexto_levanta_erro() -> None:
    gpt = modelo()
    try:
        gpt(ids_aleatorios(1, TINY.context_length + 1))
    except ValueError as erro:
        assert str(TINY.context_length) in str(erro)
    else:
        raise AssertionError("o modelo aceitou sequência maior que context_length")


def test_componentes_somam_o_total() -> None:
    for cfg in (TINY, BASE_CONFIG, replace(TINY, tie_weights=True)):
        gpt = no_meta(cfg)
        partes = gpt.parameter_breakdown()

        assert tuple(partes) == COMPONENTS
        assert sum(partes.values()) == gpt.parameter_count()


def test_contagem_bate_com_as_formulas() -> None:
    cfg = BASE_CONFIG
    partes = no_meta(cfg).parameter_breakdown()
    d, v, h = cfg.emb_dim, cfg.vocab_size, cfg.feed_forward_dim

    assert partes["token_embedding"] == v * d
    assert partes["positional_embedding"] == cfg.context_length * d
    # Por bloco: Q, K, V sem bias + out_proj com bias; duas Linear com bias; 2 normas.
    assert partes["attention"] == cfg.n_layers * (3 * d * d + d * d + d)
    assert partes["feed_forward"] == cfg.n_layers * (2 * d * h + h + d)
    assert partes["block_norms"] == cfg.n_layers * 2 * 2 * d
    assert partes["final_norm"] == 2 * d
    assert partes["out_head"] == d * v


def test_gpt2_small_tem_os_numeros_do_livro() -> None:
    gpt = no_meta(GPT_CONFIG_124M)

    # Seção 4.6.
    assert gpt.parameter_count() == 163_009_536
    assert f"{gpt.size_in_megabytes():.2f}" == "621.83"
    assert gpt.embedding.token_embedding.weight.shape == (50_257, 768)
    assert gpt.out_head.weight.shape == (50_257, 768)

    # Sem contar a cabeça de saída, como o GPT-2 original faz com weight tying.
    sem_cabeca = gpt.parameter_count() - gpt.parameter_breakdown()["out_head"]
    assert sem_cabeca == 124_412_160

    # E o mesmo número sai do modelo com tying de verdade.
    assert no_meta(replace(GPT_CONFIG_124M, tie_weights=True)).parameter_count() == 124_412_160


def test_gpt2_xl_tem_os_numeros_do_livro() -> None:
    # Exercício 4.2 (apêndice C).
    gpt = no_meta(GPT2_CONFIGS["gpt2-xl"])

    assert gpt.parameter_count() == 1_637_792_000
    assert gpt.parameter_count() - gpt.parameter_breakdown()["out_head"] == 1_557_380_800
    assert f"{gpt.size_in_megabytes():.2f}" == "6247.68"


def test_weight_tying_compartilha_o_mesmo_tensor() -> None:
    cfg = replace(TINY, tie_weights=True)
    gpt = modelo(cfg).train()

    tabela = gpt.embedding.token_embedding.weight
    assert gpt.out_head.weight is tabela
    assert gpt.parameter_count() == modelo(TINY).parameter_count() - TINY.vocab_size * TINY.emb_dim
    assert gpt.parameter_breakdown()["out_head"] == 0

    # Um gradiente só, somado das duas pontas: a tabela recebe gradiente até
    # nas linhas de tokens que não estão na entrada, por causa da cabeça.
    gpt(ids_aleatorios(2, 5)).pow(2).mean().backward()
    assert tabela.grad is not None
    assert bool((tabela.grad.abs().sum(dim=1) > 0).all())


def test_logit_da_posicao_t_nao_depende_do_futuro() -> None:
    gpt = modelo()
    ids = ids_aleatorios(1, 10)
    corte = 5

    alterado = ids.clone()
    alterado[0, corte + 1 :] = (ids[0, corte + 1 :] + 1) % TINY.vocab_size

    logits, logits_alterados = gpt(ids), gpt(alterado)

    # Prefixo idêntico => logits do prefixo idênticos. É o que permite treinar
    # as T posições de uma vez sem que nenhuma leia a resposta.
    assert torch.allclose(logits[0, : corte + 1], logits_alterados[0, : corte + 1], atol=1e-6)
    assert not torch.allclose(logits[0, corte + 1 :], logits_alterados[0, corte + 1 :], atol=1e-6)


def test_logits_de_um_prefixo_sao_os_da_sequencia_inteira() -> None:
    gpt = modelo()
    ids = ids_aleatorios(1, 10)

    completos = gpt(ids)
    for tamanho in (1, 4, 9):
        # Rodar só o prefixo dá os mesmos logits que as primeiras posições da
        # passada completa: é a base da geração token a token.
        assert torch.allclose(gpt(ids[:, :tamanho]), completos[:, :tamanho], atol=1e-5)


def test_dropout_so_age_em_train() -> None:
    gpt = modelo(replace(TINY, drop_rate=0.3))
    ids = ids_aleatorios(2, 8)

    assert torch.equal(gpt(ids), gpt(ids))

    gpt.train()
    assert not torch.equal(gpt(ids), gpt(ids))


def test_taxas_de_dropout_separadas_chegam_aos_modulos() -> None:
    # Exercício 4.3.
    cfg = replace(TINY, drop_rate=0.1, drop_rate_emb=0.0, drop_rate_attn=0.2, drop_rate_shortcut=0.3)
    gpt = modelo(cfg)

    assert gpt.drop_emb.p == 0.0
    for camada in gpt.trf_blocks:
        assert camada.att.dropout.p == 0.2
        assert camada.drop_shortcut.p == 0.3

    # Sem as três, todas herdam drop_rate, como no livro.
    gpt = modelo(replace(TINY, drop_rate=0.1))
    assert gpt.drop_emb.p == 0.1
    assert all(b.att.dropout.p == b.drop_shortcut.p == 0.1 for b in gpt.trf_blocks)


def test_configuracao_invalida_falha_na_construcao() -> None:
    invalidas = {
        "emb_dim indivisível por n_heads": {"emb_dim": 18},
        "dropout igual a 1": {"drop_rate": 1.0},
        "dropout negativo": {"drop_rate_attn": -0.1},
        "contexto zero": {"context_length": 0},
        "camadas negativas": {"n_layers": -1},
    }
    for motivo, mudanca in invalidas.items():
        try:
            replace(TINY, **mudanca)
        except ValueError:
            pass
        else:
            raise AssertionError(f"GPTConfig aceitou {motivo}")


def test_zero_blocos_e_o_esqueleto_do_livro() -> None:
    # n_layers = 0: embeddings, norma e cabeça, sem bloco nenhum. É o
    # DummyGPTModel da seção 4.1 (com a norma de verdade no lugar da falsa).
    gpt = modelo(replace(TINY, n_layers=0))
    assert len(gpt.trf_blocks) == 0
    assert gpt(ids_aleatorios(2, 4)).shape == (2, 4, TINY.vocab_size)


def test_vocabulario_da_configuracao_e_o_do_bpe() -> None:
    assert GPT2_VOCAB_SIZE == len(BPETokenizer())
    assert all(cfg.vocab_size == GPT2_VOCAB_SIZE for cfg in GPT2_CONFIGS.values())
    assert BASE_CONFIG.head_dim == 64


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
