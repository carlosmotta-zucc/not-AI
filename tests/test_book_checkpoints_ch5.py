"""Checkpoints do livro, capítulo 5 (seções 5.1 e 5.2): avaliação e treino.

A mesma ideia de tests/test_book_checkpoints.py, que cobre o capítulo 4 e a
seção 5.3: cada teste refaz um exemplo do livro com o código de src/ e compara
com o valor impresso, guardado em tests/book_reference.py com a página.

O que fica protegido aqui:

- 5.1.1: a geração gulosa do GPT de 124M (contexto 256, seed 123) antes do
  treino.
- 5.1.2: a perda passo a passo -- argmax, probabilidade de cada alvo, log,
  média -- e a cross-entropy e a perplexidade calculadas pelo PyTorch.
- 5.1.3: o tamanho do conto, os lotes de treino e de validação e as perdas
  iniciais dos dois conjuntos.
- 5.2: os hiperparâmetros do treino e as duas primeiras avaliações (passos 0 e
  5), com 3 casas. Estas perdas só batem se o laço fizer as mesmas operações na
  mesma ordem e consumir o gerador global como o train_model_simple do livro --
  embaralhamento, dropout e iteradores da avaliação.

O treino completo de 10 épocas (o log inteiro e as amostras de texto) leva
alguns minutos e fica em experimentos/sprint05/livro_exemplos.py, que publica
a comparação lado a lado.

Este arquivo é separado do outro por causa da memória: o modelo de 124M em
treino carrega, além dos pesos, os gradientes e os dois momentos do AdamW,
~2,4 GB ao todo. Cada teste monta o seu modelo, sem cache, para que só um
exista por vez. Leva cerca de meio minuto.

Uso:
    python tests/test_book_checkpoints_ch5.py
    pytest tests/
"""

import sys
from dataclasses import replace
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import load_verdict  # noqa: E402
from src.generation import generate, text_to_token_ids, token_ids_to_text  # noqa: E402
from src.model import GPT_CONFIG_124M, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from src.training import (  # noqa: E402
    BOOK_MODEL_CONFIG,
    BOOK_TRAINING_CONFIG,
    batch_loss,
    create_optimizer,
    create_train_val_loaders,
    evaluate,
    perplexity,
    split_text,
    target_log_probabilities,
    train_model,
)
from tests import book_reference as livro  # noqa: E402

# Passos de treino necessários para as duas primeiras linhas do log (0 e 5).
FIRST_STEPS = 6


class PrimeirosLotes:
    """Os primeiros n lotes de um DataLoader, consumindo o gerador global como ele.

    Cada iteração cria UM iterador do loader de verdade -- o que sorteia o
    embaralhamento, como no livro -- e para depois de n lotes. É o que deixa o
    teste treinar só o começo da primeira época sem mudar a sequência de
    sorteios das avaliações que interessam.
    """

    def __init__(self, loader, n: int) -> None:
        self.loader = loader
        self.n = n

    def __iter__(self):
        return islice(iter(self.loader), self.n)

    def __len__(self) -> int:
        return self.n


def modelo_do_capitulo() -> GPTModel:
    """O GPTModel da seção 5.1.1: seed 123, contexto 256, em eval()."""
    torch.manual_seed(livro.BOOK_SEED)
    return GPTModel(BOOK_MODEL_CONFIG).eval()


def loaders_do_capitulo():
    """Os loaders da seção 5.1.3: 90% do conto para treino, lotes de 2 x 256."""
    return create_train_val_loaders(
        load_verdict(),
        BPETokenizer(),
        context_length=livro.CH5_CONTEXT_LENGTH,
        batch_size=livro.LOADER_BATCH_SIZE,
        train_ratio=livro.TRAIN_RATIO,
    )


def confere_cientifico(obtido: torch.Tensor, esperado: list) -> None:
    """Valores impressos como 7.4541e-05: meia unidade na 4ª casa da mantissa."""
    esperado_t = torch.tensor(esperado, dtype=torch.float64)
    obtido_t = obtido.detach().to(torch.float64)
    escala = 10.0 ** torch.floor(torch.log10(esperado_t.abs()))
    folga = 5e-5 * escala * (1 + 1e-6)
    assert bool(((obtido_t - esperado_t).abs() <= folga).all()), (obtido_t, esperado_t)


def test_5_1_1_geracao_antes_do_treino() -> None:
    modelo = modelo_do_capitulo()
    tokenizer = BPETokenizer()

    ids = text_to_token_ids(livro.START_CONTEXT, tokenizer)
    assert ids.tolist() == [livro.START_CONTEXT_IDS]

    saida = generate(modelo, ids, max_new_tokens=livro.UNTRAINED_NEW_TOKENS)
    texto = token_ids_to_text(saida, tokenizer)

    assert livro.UNPRINTABLE_IN_BOOK in texto
    assert texto.replace(livro.UNPRINTABLE_IN_BOOK, "?") == livro.UNTRAINED_OUTPUT_TEXT


def test_5_1_2_perda_passo_a_passo_e_perplexidade() -> None:
    modelo = modelo_do_capitulo()
    tokenizer = BPETokenizer()
    entradas = torch.tensor(livro.LOSS_INPUTS)
    alvos = torch.tensor(livro.LOSS_TARGETS)
    tol = livro.PRINT_TOLERANCE

    with torch.no_grad():
        logits = modelo(entradas)
    probabilidades = torch.softmax(logits, dim=-1)
    escolhidos = probabilidades.argmax(dim=-1)

    assert tuple(probabilidades.shape) == livro.PROBAS_SHAPE
    assert escolhidos.tolist() == livro.LOSS_ARGMAX_IDS
    assert token_ids_to_text(alvos[0], tokenizer) == livro.TARGETS_TEXT_1
    assert token_ids_to_text(escolhidos[0], tokenizer) == livro.OUTPUTS_TEXT_1

    # Figura 5.7, passos 1 e 2: logits e probabilidades da primeira posição.
    assert torch.allclose(logits[0, 0, :3], torch.tensor(livro.FIRST_LOGITS), atol=tol)
    confere_cientifico(probabilidades[0, 0, :3], livro.FIRST_PROBAS)

    # Passos 3 a 6: probabilidade do alvo, log, média, sinal.
    log_do_alvo = target_log_probabilities(logits, alvos)
    confere_cientifico(log_do_alvo.exp(), livro.TARGET_PROBAS)
    assert torch.allclose(log_do_alvo.flatten(), torch.tensor(livro.LOG_PROBAS), atol=tol)
    assert abs(log_do_alvo.mean().item() - livro.AVG_LOG_PROBA) < tol

    # O mesmo número pelo PyTorch, com os tensores achatados.
    assert tuple(logits.flatten(0, 1).shape) == livro.FLAT_LOGITS_SHAPE
    assert tuple(alvos.flatten().shape) == livro.FLAT_TARGETS_SHAPE
    perda = F.cross_entropy(logits.flatten(0, 1), alvos.flatten())
    assert abs(perda.item() - livro.CROSS_ENTROPY) < tol
    with torch.no_grad():
        assert abs(batch_loss(modelo, entradas, alvos).item() - livro.CROSS_ENTROPY) < tol

    # O livro calcula torch.exp em float32; perplexity() faz a conta em
    # float64. As duas concordam até a 8ª casa significativa.
    assert abs(torch.exp(perda).item() - livro.PERPLEXITY) < tol
    assert abs(perplexity(perda) - livro.PERPLEXITY) / livro.PERPLEXITY < 1e-6


def test_5_1_3_dados_e_perdas_iniciais() -> None:
    texto = load_verdict()
    tokenizer = BPETokenizer()
    assert len(texto) == livro.VERDICT_CHARS
    assert len(tokenizer.encode(texto)) == livro.VERDICT_TOKENS

    treino, validacao = loaders_do_capitulo()
    assert len(treino) == livro.TRAIN_BATCHES
    assert len(validacao) == livro.VAL_BATCHES
    for entradas, alvos in [*treino, *validacao]:
        assert tuple(entradas.shape) == tuple(alvos.shape) == livro.LOADER_BATCH_SHAPE

    perda_treino, perda_validacao = evaluate(modelo_do_capitulo(), treino, validacao)

    assert abs(perda_treino - livro.INITIAL_TRAIN_LOSS) < livro.INITIAL_LOSS_TOLERANCE
    assert abs(perda_validacao - livro.INITIAL_VAL_LOSS) < livro.INITIAL_LOSS_TOLERANCE


def test_5_2_hiperparametros_do_livro() -> None:
    assert BOOK_MODEL_CONFIG.context_length == livro.CH5_CONTEXT_LENGTH
    assert replace(BOOK_MODEL_CONFIG, context_length=1_024) == GPT_CONFIG_124M

    cfg = BOOK_TRAINING_CONFIG
    assert cfg.seed == livro.BOOK_SEED
    assert cfg.learning_rate == livro.LEARNING_RATE
    assert cfg.weight_decay == livro.WEIGHT_DECAY
    assert cfg.num_epochs == livro.NUM_EPOCHS
    assert cfg.batch_size == livro.LOADER_BATCH_SIZE
    assert cfg.train_ratio == livro.TRAIN_RATIO
    assert (cfg.eval_freq, cfg.eval_iter) == (livro.EVAL_FREQ, livro.EVAL_ITER)
    assert (cfg.start_context, cfg.sample_tokens) == (livro.START_CONTEXT, livro.SAMPLE_TOKENS)


def test_5_2_primeiras_avaliacoes_do_treino() -> None:
    treino, validacao = loaders_do_capitulo()

    # A ordem do livro (p. 149): seed, modelo, otimizador, laço.
    torch.manual_seed(livro.BOOK_SEED)
    modelo = GPTModel(BOOK_MODEL_CONFIG)
    otimizador = create_optimizer(modelo, BOOK_TRAINING_CONFIG)

    historico = train_model(
        modelo,
        PrimeirosLotes(treino, FIRST_STEPS),
        validacao,
        otimizador,
        num_epochs=1,
        eval_freq=livro.EVAL_FREQ,
        eval_iter=livro.EVAL_ITER,
        sample_tokens=0,
        log=None,
    )

    assert historico.steps == [0, 5]
    for indice, passo in enumerate(historico.steps):
        epoca, perda_treino, perda_validacao = livro.TRAINING_LOG[passo]
        assert historico.epochs[indice] == epoca
        assert abs(historico.train_losses[indice] - perda_treino) < livro.LOSS_PRINT_TOLERANCE
        assert abs(historico.val_losses[indice] - perda_validacao) < livro.LOSS_PRINT_TOLERANCE


def test_5_2_trecho_memorizado_vem_do_treino() -> None:
    texto_treino, texto_validacao = split_text(load_verdict(), livro.TRAIN_RATIO)

    assert livro.MEMORIZED_SNIPPET in texto_treino
    assert livro.MEMORIZED_SNIPPET not in texto_validacao


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
