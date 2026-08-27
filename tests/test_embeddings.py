"""Testes da Frente B da Sprint 2: do Token ID ao vetor.

Cobrem as afirmacoes que a analise faz sobre embeddings -- que a camada de
embedding e literalmente uma multiplicacao por matriz one-hot, e que o mesmo
token em posicoes diferentes so deixa de ter o mesmo vetor depois da soma
posicional. A janela deslizante (GPTDataset, create_dataloader) tem testes
proprios em tests/test_dataset.py; aqui ela so aparece no teste de pipeline,
que prova que a saida de src/dataset/ encaixa na entrada de src/embeddings/.
Se algum destes testes cair, uma frase da analise deixou de ser verdadeira.

Uso:
    python tests/test_embeddings.py
    pytest tests/
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import create_dataloader  # noqa: E402
from src.embeddings import (  # noqa: E402
    InputEmbedding,
    PositionalEmbedding,
    TokenEmbedding,
    one_hot_lookup,
)
from src.tokenizer import SimpleTokenizer, Vocabulary  # noqa: E402

SEED = 42


def test_embedding_troca_id_por_vetor() -> None:
    torch.manual_seed(SEED)
    camada = TokenEmbedding(vocab_size=50, emb_dim=16)
    lote = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    assert camada(lote).shape == (2, 4, 16)


def test_tabela_tem_uma_linha_por_entrada_do_vocabulario() -> None:
    torch.manual_seed(SEED)
    vocab = Vocabulary.from_texts(["o gato mordeu o rato."])
    camada = TokenEmbedding(vocab_size=len(vocab), emb_dim=16)

    assert camada.weight.shape == (len(vocab), 16)
    assert camada.parameter_count() == len(vocab) * 16


def test_mesmo_id_em_posicoes_diferentes_recebe_o_mesmo_vetor() -> None:
    torch.manual_seed(SEED)
    texto = "o gato mordeu o rato."
    vocab = Vocabulary.from_texts([texto])
    ids = torch.tensor([SimpleTokenizer(vocab).encode(texto)])

    vetores = TokenEmbedding(len(vocab), emb_dim=16)(ids)

    primeiro_o, segundo_o = 0, 3
    assert ids[0, primeiro_o] == ids[0, segundo_o]
    assert torch.equal(vetores[0, primeiro_o], vetores[0, segundo_o])


def test_one_hot_equivale_a_consulta_na_tabela() -> None:
    torch.manual_seed(SEED)
    camada = TokenEmbedding(vocab_size=50, emb_dim=16)
    lote = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    assert torch.allclose(camada(lote), one_hot_lookup(lote, camada.weight))


def test_tabela_de_embeddings_e_treinavel() -> None:
    torch.manual_seed(SEED)
    camada = TokenEmbedding(vocab_size=50, emb_dim=16)
    camada(torch.tensor([[1, 2, 3, 4]])).sum().backward()

    assert camada.weight.grad is not None
    assert camada.weight.grad[1].abs().sum() > 0, "linha usada deveria receber gradiente"
    assert camada.weight.grad[40].abs().sum() == 0, "linha nao usada nao recebe gradiente"


def test_posicional_sai_sem_dimensao_de_lote() -> None:
    torch.manual_seed(SEED)
    camada = PositionalEmbedding(context_length=4, emb_dim=16)

    assert camada().shape == (4, 16)
    assert camada(2).shape == (2, 16)


def test_posicao_alem_do_contexto_levanta_erro() -> None:
    torch.manual_seed(SEED)
    camada = PositionalEmbedding(context_length=4, emb_dim=16)
    try:
        camada(5)
    except ValueError:
        return
    raise AssertionError("esperava ValueError para sequencia maior que o contexto")


def test_entrada_e_a_soma_de_token_e_posicao() -> None:
    torch.manual_seed(SEED)
    camada = InputEmbedding(vocab_size=50, emb_dim=16, context_length=4)
    lote = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    entrada = camada(lote)
    tokens = camada.token_embedding(lote)
    posicoes = camada.positional_embedding(4)

    assert entrada.shape == (2, 4, 16)
    for exemplo in range(2):
        for posicao in range(4):
            assert torch.allclose(
                entrada[exemplo, posicao], tokens[exemplo, posicao] + posicoes[posicao]
            )


def test_mesma_posicao_soma_o_mesmo_vetor_em_todo_o_lote() -> None:
    torch.manual_seed(SEED)
    camada = InputEmbedding(vocab_size=50, emb_dim=16, context_length=4)
    lote = torch.tensor([[7, 7, 7, 7], [7, 7, 7, 7]])

    entrada = camada(lote)

    assert torch.allclose(entrada[0], entrada[1])


def test_posicao_desfaz_a_igualdade_entre_tokens_repetidos() -> None:
    torch.manual_seed(SEED)
    camada = InputEmbedding(vocab_size=50, emb_dim=16, context_length=4)
    lote = torch.tensor([[7, 0, 0, 7]])

    tokens = camada.token_embedding(lote)
    entrada = camada(lote)

    assert torch.equal(tokens[0, 0], tokens[0, 3])
    assert not torch.allclose(entrada[0, 0], entrada[0, 3])


def test_pipeline_completo_do_texto_ao_tensor() -> None:
    torch.manual_seed(SEED)
    texto = "o gato mordeu o rato. " * 20
    vocab = Vocabulary.from_texts([texto])

    loader = create_dataloader(
        texto, SimpleTokenizer(vocab), context_length=4, stride=4, batch_size=8
    )
    entrada, alvo = next(iter(loader))
    vetores = InputEmbedding(len(vocab), emb_dim=256, context_length=4)(entrada)

    assert entrada.shape == (8, 4)
    assert alvo.shape == (8, 4)
    assert vetores.shape == (8, 4, 256)


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
