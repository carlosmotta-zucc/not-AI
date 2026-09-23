"""Teste de integracao: The Verdict -> BPE -> lotes -> embeddings -> atencao.

Os outros arquivos de tests/ cobrem um pacote cada; este liga os dois. O
experimento E5 da Sprint 2 ja percorria o caminho, mas parava no InputEmbedding
-- foi escrito antes de a atencao existir.

O que fica protegido aqui:

- Os shapes fecham de ponta a ponta, na configuracao base do projeto:
  (B, T) long -> (B, T, emb_dim) -> (B, T, emb_dim).
- A saida da atencao tem o MESMO shape da entrada quando d_in = d_out = emb_dim.
  E a condicao do `x + attn(x)` da Sprint 4: sem isso o bloco nao monta.
- O backward() atravessa o pipeline e chega com gradiente NAO NULO nas duas
  tabelas de embedding e em todas as matrizes da atencao. Um detach esquecido no
  meio congelaria parte da rede em silencio, e o treino nunca acusaria.
- Na tabela de tokens o gradiente alcanca exatamente as linhas dos IDs do lote --
  consequencia de a tabela ser indexada, nao multiplicada.

Diferente dos outros testes, este depende do corpus: na primeira execucao
load_verdict() baixa The Verdict para data/, e o tiktoken baixa a tabela do BPE.

Uso:
    python tests/test_attention_pipeline.py
    pytest tests/
"""

import sys
from functools import lru_cache
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import MultiHeadAttention  # noqa: E402
from src.corpus import load_verdict  # noqa: E402
from src.dataset import create_dataloader  # noqa: E402
from src.embeddings import InputEmbedding  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

SEED = 42

# Configuracao base: os cinco primeiros valores vem do BASE_CONFIG da Sprint 2
# (experimentos/sprint02/frente_b_experimentos.py), os tres ultimos do docstring
# de src/attention/__init__.py. emb_dim 256 em 4 cabecas da head_dim 64, o do
# GPT-2; dropout desligado porque aqui se conferem valores.
PIPELINE_CONFIG = {
    "context_length": 128,
    "stride": 128,
    "batch_size": 8,
    "emb_dim": 256,
    "num_heads": 4,
    "dropout": 0.0,
    "qkv_bias": False,
    "seed": SEED,
}


@lru_cache(maxsize=1)
def primeiro_lote() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Primeiro lote de The Verdict, mais o tamanho do vocabulario do BPE.

    Returns:
        (entrada, alvo, vocab_size): dois tensores (B, T) de dtype long e o
        numero de linhas que a tabela de tokens precisa ter.

    Em cache porque os quatro testes partem do mesmo lote. shuffle=False e o que
    torna "o primeiro lote" uma frase com sentido: com o embaralhamento padrao
    do create_dataloader, cada execucao veria outro trecho.
    """
    tokenizer = BPETokenizer()
    loader = create_dataloader(
        load_verdict(),
        tokenizer,
        context_length=PIPELINE_CONFIG["context_length"],
        stride=PIPELINE_CONFIG["stride"],
        batch_size=PIPELINE_CONFIG["batch_size"],
        shuffle=False,
        drop_last=True,
    )

    entrada, alvo = next(iter(loader))
    return entrada, alvo, len(tokenizer)


def monta_camadas(vocab_size: int) -> tuple[InputEmbedding, MultiHeadAttention]:
    """As duas camadas do pipeline, com seed fixa e em modo de inferencia.

    d_in = d_out = emb_dim: e o que deixa a saida da atencao ser somada a
    propria entrada na Sprint 4.
    """
    torch.manual_seed(PIPELINE_CONFIG["seed"])

    embedding = InputEmbedding(
        vocab_size=vocab_size,
        emb_dim=PIPELINE_CONFIG["emb_dim"],
        context_length=PIPELINE_CONFIG["context_length"],
    )
    atencao = MultiHeadAttention(
        d_in=PIPELINE_CONFIG["emb_dim"],
        d_out=PIPELINE_CONFIG["emb_dim"],
        context_length=PIPELINE_CONFIG["context_length"],
        num_heads=PIPELINE_CONFIG["num_heads"],
        dropout=PIPELINE_CONFIG["dropout"],
        qkv_bias=PIPELINE_CONFIG["qkv_bias"],
    )

    return embedding.eval(), atencao.eval()


def test_shapes_do_pipeline_ponta_a_ponta() -> None:
    lote, alvo, vocab_size = primeiro_lote()
    embedding, atencao = monta_camadas(vocab_size)

    batch_size = PIPELINE_CONFIG["batch_size"]
    tokens = PIPELINE_CONFIG["context_length"]
    emb_dim = PIPELINE_CONFIG["emb_dim"]
    cabecas = PIPELINE_CONFIG["num_heads"]

    # Saida do DataLoader: Token IDs, nao vetores -- long para indexar a tabela.
    assert lote.shape == (batch_size, tokens)
    assert lote.dtype == torch.long
    assert alvo.shape == lote.shape

    vetores = embedding(lote)
    assert vetores.shape == (batch_size, tokens, emb_dim)
    assert vetores.dtype == torch.float32

    contexto, pesos = atencao(vetores)
    assert contexto.shape == (batch_size, tokens, emb_dim)
    assert pesos.shape == (batch_size, cabecas, tokens, tokens)
    assert torch.allclose(
        pesos.sum(dim=-1), torch.ones(batch_size, cabecas, tokens), atol=1e-6
    )


def test_saida_encaixa_na_conexao_residual() -> None:
    lote, _, vocab_size = primeiro_lote()
    embedding, atencao = monta_camadas(vocab_size)

    # A condicao, escrita na assinatura da camada antes de qualquer tensor.
    assert atencao.d_in == atencao.d_out == PIPELINE_CONFIG["emb_dim"]

    vetores = embedding(lote)
    contexto, _ = atencao(vetores)

    # Mesmo shape na entrada e na saida: e o que permite a linha abaixo.
    assert contexto.shape == vetores.shape

    residual = vetores + contexto
    assert residual.shape == vetores.shape

    # A soma nao e identidade: sem isto, uma camada que devolvesse zeros passaria.
    assert not torch.allclose(residual, vetores, atol=1e-6)


def test_backward_alcanca_embeddings_e_atencao() -> None:
    lote, _, vocab_size = primeiro_lote()
    embedding, atencao = monta_camadas(vocab_size)

    contexto, _ = atencao(embedding(lote))

    # Escalar de mentira, so para ter o que derivar: a perda de verdade
    # (cross-entropy sobre o proximo token) precisa da cabeca de saida da
    # Sprint 5. Ao quadrado e nao .sum() porque soma admite cancelamento, e
    # gradiente nulo assim seria indistinguivel de gradiente que nao chegou.
    contexto.pow(2).mean().backward()

    esperam_gradiente = {
        "tabela de tokens": embedding.token_embedding.weight,
        "tabela de posicoes": embedding.positional_embedding.weight,
        "W_query": atencao.W_query.weight,
        "W_key": atencao.W_key.weight,
        "W_value": atencao.W_value.weight,
        "out_proj": atencao.out_proj.weight,
        "out_proj (bias)": atencao.out_proj.bias,
    }

    for nome, tensor in esperam_gradiente.items():
        assert tensor.grad is not None, f"{nome} ficou sem gradiente"
        # Nao basta existir: gradiente todo zero nao treina nada.
        assert tensor.grad.abs().sum() > 0, f"{nome} recebeu gradiente nulo"

    # T == context_length, entao nenhuma linha da tabela de posicoes fica de fora.
    linhas_de_posicao = embedding.positional_embedding.weight.grad.abs().sum(dim=1)
    assert bool((linhas_de_posicao > 0).all())

    # A mascara atravessou o forward inteiro e continua sendo so um buffer.
    assert atencao.mask.grad is None
    assert all(
        parametro.shape != atencao.mask.shape for parametro in atencao.parameters()
    )


def test_gradiente_do_token_so_alcanca_os_ids_do_lote() -> None:
    lote, _, vocab_size = primeiro_lote()
    embedding, atencao = monta_camadas(vocab_size)

    contexto, _ = atencao(embedding(lote))
    contexto.pow(2).mean().backward()

    gradiente = embedding.token_embedding.weight.grad
    assert gradiente is not None, "tabela de tokens ficou sem gradiente"

    gradiente_por_linha = gradiente.abs().sum(dim=1)
    linhas_com_gradiente = set(torch.nonzero(gradiente_por_linha).flatten().tolist())
    ids_do_lote = set(lote.flatten().tolist())

    # A tabela e indexada, nao multiplicada: um passo de treino so ajusta os
    # tokens que apareceram -- e por isso que token raro aprende devagar.
    assert linhas_com_gradiente == ids_do_lote
    assert len(ids_do_lote) < vocab_size


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
