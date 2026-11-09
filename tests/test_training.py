"""Testes da avaliação e do treinamento (Sprint 5, seções 5.1 e 5.2).

O que fica protegido aqui:

- Perda: o passo a passo (log-probabilidade do alvo, média, sinal) dá a mesma
  cross-entropy do PyTorch; a perda de um DataLoader pondera cada lote pelos
  tokens, roda em eval() sem grafo e devolve o modo original; loader vazio
  dá NaN.
- Perplexidade: distribuição uniforme sobre k candidatos dá perplexidade k.
- Dados: a divisão em caracteres, os lotes do treino (embaralhados, sem lote
  incompleto) e da validação (ordem fixa, todos os lotes, stride fixo).
- Treino: a perda cai e o modelo aprende a tarefa (o alfabeto); o histórico
  tem uma entrada por avaliação, por passo e por época, com contagens
  coerentes; duas execuções com a mesma seed são idênticas; treinar 2 + 2
  épocas, retomando, dá exatamente o mesmo que 4 de uma vez.
- Configuração: valores inválidos falham na construção.

Os números do livro (perda 10,7940, perplexidade 48 725,8, perdas iniciais e
o log do treino) estão em tests/test_book_checkpoints.py.

Uso:
    python tests/test_training.py
    pytest tests/
"""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import GPTConfig, GPTModel  # noqa: E402
from src.training import (  # noqa: E402
    BOOK_TRAINING_CONFIG,
    TrainingConfig,
    TrainingHistory,
    batch_loss,
    create_optimizer,
    create_train_val_loaders,
    evaluate,
    loader_loss,
    perplexity,
    split_text,
    target_log_probabilities,
    train_gpt,
    train_model,
)

SEED = 42

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# 520 caracteres: 468 de treino (58 janelas de 8 -> 14 lotes de 4) e 52 de
# validação (6 janelas -> lotes de 4 e 2, o segundo incompleto).
TEXT = ALPHABET * 20

TINY = GPTConfig(
    vocab_size=len(ALPHABET),
    context_length=8,
    emb_dim=16,
    n_heads=2,
    n_layers=2,
    drop_rate=0.1,
)

# Taxa 25x a do livro: o modelo é minúsculo e a tarefa, determinística; com
# ela, 5 épocas (70 passos) bastam para a perda cair de ln 26 para ~0,1.
TINY_TRAINING = TrainingConfig(
    num_epochs=5,
    batch_size=4,
    learning_rate=1e-2,
    weight_decay=0.1,
    eval_freq=5,
    eval_iter=None,
    seed=SEED,
    start_context="abc",
    sample_tokens=10,
)


class CharTokenizer:
    """Uma letra, um token: a tarefa vira "a próxima letra do alfabeto",
    determinística, que o modelo minúsculo aprende em poucos passos."""

    def encode(self, text: str) -> list[int]:
        return [ALPHABET.index(letter) for letter in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(ALPHABET[i] for i in ids)


def modelo() -> GPTModel:
    torch.manual_seed(SEED)
    return GPTModel(TINY)


def lote(sequencias: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED)
    entradas = torch.randint(0, TINY.vocab_size, (sequencias, TINY.context_length))
    alvos = torch.randint(0, TINY.vocab_size, (sequencias, TINY.context_length))
    return entradas, alvos


def treinar(cfg: TrainingConfig = TINY_TRAINING):
    return train_gpt(TINY, cfg, TEXT, CharTokenizer(), log=None)


def test_cross_entropy_passo_a_passo_e_a_do_pytorch() -> None:
    torch.manual_seed(SEED)
    logits = torch.randn(2, 5, 26)
    alvos = torch.randint(0, 26, (2, 5))

    pytorch = F.cross_entropy(logits.flatten(0, 1), alvos.flatten())
    projeto = -target_log_probabilities(logits, alvos).mean()

    # O caminho do livro (figura 5.7): softmax, probabilidade do alvo, log,
    # média, sinal trocado.
    probabilidades = torch.softmax(logits, dim=-1)
    do_alvo = probabilidades.gather(-1, alvos.unsqueeze(-1)).squeeze(-1)
    livro = -torch.log(do_alvo).mean()

    assert torch.allclose(projeto, pytorch, atol=1e-6)
    assert torch.allclose(livro, pytorch, atol=1e-6)


def test_log_probabilidade_do_alvo_recusa_shapes_que_nao_casam() -> None:
    try:
        target_log_probabilities(torch.zeros(2, 5, 26), torch.zeros(2, 4, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("alvos (2, 4) para logits (2, 5, 26) deveriam falhar")


def test_perplexidade_da_uniforme_e_o_numero_de_candidatos() -> None:
    for candidatos in (2, 26, 50_257):
        logits = torch.zeros(3, candidatos)
        alvos = torch.zeros(3, dtype=torch.long)
        perda = F.cross_entropy(logits, alvos)

        assert abs(perda.item() - math.log(candidatos)) < 1e-5
        assert abs(perplexity(perda) - candidatos) / candidatos < 1e-5


def test_perplexidade_nos_extremos() -> None:
    assert perplexity(0.0) == 1.0
    assert abs(perplexity(torch.tensor(1.0)) - math.e) < 1e-6
    # exp(1000) não cabe no float64: infinito, e não exceção.
    assert perplexity(1_000.0) == math.inf


def test_batch_loss_e_a_cross_entropy_dos_logits_do_modelo() -> None:
    rede = modelo().eval()
    entradas, alvos = lote()

    with torch.no_grad():
        esperado = F.cross_entropy(rede(entradas).flatten(0, 1), alvos.flatten())
        obtido = batch_loss(rede, entradas, alvos)

    assert torch.allclose(obtido, esperado)


def test_loader_loss_pondera_cada_lote_pelos_tokens() -> None:
    rede = modelo().eval()
    entradas, alvos = lote(3)
    # Lotes de 2 e de 1 sequência: o segundo vem incompleto.
    loader = DataLoader(TensorDataset(entradas, alvos), batch_size=2)

    with torch.no_grad():
        todos_juntos = batch_loss(rede, entradas, alvos).item()
        primeiro = batch_loss(rede, entradas[:2], alvos[:2]).item()
        segundo = batch_loss(rede, entradas[2:], alvos[2:]).item()

    # A média por token é a do lote único com as três sequências...
    assert abs(loader_loss(rede, loader) - todos_juntos) < 1e-6
    # ...e a média das médias por lote, que o livro usa, daria outro número.
    assert abs((primeiro + segundo) / 2 - todos_juntos) > 1e-4

    assert abs(loader_loss(rede, loader, max_batches=1) - primeiro) < 1e-6


def test_loader_loss_vazio_da_nan_e_recusa_max_batches_invalido() -> None:
    rede = modelo()
    vazio = DataLoader(
        TensorDataset(torch.empty(0, 8, dtype=torch.long), torch.empty(0, 8, dtype=torch.long)),
        batch_size=2,
    )
    assert math.isnan(loader_loss(rede, vazio))

    entradas, alvos = lote()
    try:
        loader_loss(rede, DataLoader(TensorDataset(entradas, alvos)), max_batches=0)
    except ValueError:
        return
    raise AssertionError("max_batches=0 deveria falhar")


def test_loader_loss_avalia_em_eval_sem_grafo_e_devolve_o_modo() -> None:
    rede = modelo().train()
    entradas, alvos = lote()
    loader = DataLoader(TensorDataset(entradas, alvos), batch_size=2)

    # Com dropout de 0,1 ligado, duas medições dariam números diferentes.
    assert loader_loss(rede, loader) == loader_loss(rede, loader)
    assert rede.training
    assert all(parametro.grad is None for parametro in rede.parameters())

    rede.eval()
    loader_loss(rede, loader)
    assert not rede.training


def test_avaliacao_nao_muda_os_pesos() -> None:
    rede = modelo()
    antes = [parametro.detach().clone() for parametro in rede.parameters()]
    treino, validacao = create_train_val_loaders(TEXT, CharTokenizer(), 8, 4)

    evaluate(rede, treino, validacao)

    assert all(torch.equal(a, p) for a, p in zip(antes, rede.parameters()))


def test_split_text_corta_nos_caracteres() -> None:
    texto = "abcdefghij"
    treino, validacao = split_text(texto, 0.9)

    assert (treino, validacao) == ("abcdefghi", "j")
    assert treino + validacao == texto
    assert len(split_text(TEXT)[0]) == int(0.9 * len(TEXT))

    for fracao in (0.0, 1.0, -0.1, 1.5):
        try:
            split_text(texto, fracao)
        except ValueError:
            continue
        raise AssertionError(f"train_ratio={fracao} deveria falhar")


def test_loaders_de_treino_e_validacao_seguem_o_livro() -> None:
    treino, validacao = create_train_val_loaders(TEXT, CharTokenizer(), 8, 4)

    assert len(treino) == 14  # 58 janelas, lote incompleto descartado
    assert [x.shape[0] for x, _ in validacao] == [4, 2]  # 6 janelas, todas usadas
    assert isinstance(treino.sampler, RandomSampler)
    assert isinstance(validacao.sampler, SequentialSampler)

    for entradas, alvos in treino:
        assert entradas.shape == (4, 8)
        assert torch.equal(entradas[:, 1:], alvos[:, :-1])

    # Stride do treino muda; a régua da validação, não.
    treino_2, validacao_2 = create_train_val_loaders(TEXT, CharTokenizer(), 8, 4, stride=2)
    assert treino_2.dataset.stride == 2
    assert validacao_2.dataset.stride == 8
    assert len(validacao_2) == len(validacao)


def test_loaders_recusam_conjunto_sem_nenhum_lote() -> None:
    try:
        create_train_val_loaders(TEXT, CharTokenizer(), context_length=100, batch_size=4)
    except ValueError:
        return
    raise AssertionError("validação de 52 tokens não forma janela de 100: deveria falhar")


def test_treino_reduz_a_perda_e_aprende_o_alfabeto() -> None:
    run = treinar()
    historico = run.history

    # Ponto de partida: chute quase uniforme entre 26 letras, ln 26 = 3,26.
    assert abs(historico.val_losses[0] - math.log(26)) < 1.0
    assert historico.train_losses[-1] < 0.3
    assert historico.val_losses[-1] < 0.3
    assert perplexity(historico.val_losses[-1]) < 1.4

    # A tarefa aprendida, vista na geração: depois de "abc", o resto do alfabeto.
    assert historico.samples[-1] == "abcdefghijklm"


def test_treino_atualiza_todos_os_pesos() -> None:
    run = treinar(replace(TINY_TRAINING, num_epochs=1))
    inicial = modelo()

    for (nome, treinado), original in zip(run.model.named_parameters(), inicial.parameters()):
        assert treinado.grad is not None, f"{nome} não recebeu gradiente"
        assert not torch.equal(treinado, original), f"{nome} não mudou"


def test_historico_do_treino_e_coerente() -> None:
    run = treinar()
    historico = run.history
    passos_por_epoca = len(run.train_loader)
    total_de_passos = TINY_TRAINING.num_epochs * passos_por_epoca
    tokens_por_passo = TINY_TRAINING.batch_size * TINY.context_length

    assert historico.global_step == total_de_passos - 1
    assert len(historico.batch_losses) == total_de_passos
    assert historico.total_tokens == total_de_passos * tokens_por_passo

    assert historico.steps == list(range(0, total_de_passos, TINY_TRAINING.eval_freq))
    assert historico.tokens_seen == [(passo + 1) * tokens_por_passo for passo in historico.steps]
    assert historico.epochs == [passo // passos_por_epoca + 1 for passo in historico.steps]
    assert len(historico.train_losses) == len(historico.val_losses) == len(historico.steps)

    for por_epoca in (historico.samples, historico.epoch_seconds, historico.monitor_seconds):
        assert len(por_epoca) == TINY_TRAINING.num_epochs
    assert historico.epochs_completed == TINY_TRAINING.num_epochs

    # Termina em modo de treino, como no livro.
    assert run.model.training


def test_treino_e_reprodutivel_com_a_mesma_seed() -> None:
    primeiro, segundo = treinar(), treinar()

    assert primeiro.history.batch_losses == segundo.history.batch_losses
    assert primeiro.history.val_losses == segundo.history.val_losses
    for a, b in zip(primeiro.model.parameters(), segundo.model.parameters()):
        assert torch.equal(a, b)


def test_treino_retomado_e_identico_ao_continuo() -> None:
    cfg = replace(TINY_TRAINING, num_epochs=4)
    continuo = treinar(cfg)

    parte = treinar(replace(cfg, num_epochs=2))
    train_model(
        parte.model,
        parte.train_loader,
        parte.val_loader,
        parte.optimizer,
        num_epochs=2,
        eval_freq=cfg.eval_freq,
        eval_iter=cfg.eval_iter,
        tokenizer=CharTokenizer(),
        start_context=cfg.start_context,
        sample_tokens=cfg.sample_tokens,
        history=parte.history,
        log=None,
    )

    for campo in ("steps", "epochs", "tokens_seen", "train_losses", "val_losses", "batch_losses", "samples"):
        assert getattr(parte.history, campo) == getattr(continuo.history, campo), campo
    for a, b in zip(parte.model.parameters(), continuo.model.parameters()):
        assert torch.equal(a, b)


def test_historico_vai_e_volta_em_json() -> None:
    historico = treinar(replace(TINY_TRAINING, num_epochs=1)).history

    texto = json.dumps(historico.to_dict())
    assert TrainingHistory.from_dict(json.loads(texto)) == historico

    melhor = historico.best_evaluation()
    assert historico.val_losses[melhor] == min(historico.val_losses)

    try:
        TrainingHistory().best_evaluation()
    except ValueError:
        return
    raise AssertionError("histórico vazio não tem melhor avaliação")


def test_train_model_recusa_entradas_invalidas() -> None:
    rede = modelo()
    treino, validacao = create_train_val_loaders(TEXT, CharTokenizer(), 8, 4)
    otimizador = create_optimizer(rede, TINY_TRAINING)
    vazio = DataLoader(TensorDataset(torch.empty(0, 8, dtype=torch.long), torch.empty(0, 8, dtype=torch.long)))

    casos = [
        dict(train_loader=treino, num_epochs=-1, eval_freq=5),
        dict(train_loader=treino, num_epochs=1, eval_freq=0),
        dict(train_loader=vazio, num_epochs=1, eval_freq=5),
    ]
    for caso in casos:
        try:
            train_model(rede, val_loader=validacao, optimizer=otimizador, eval_iter=None, log=None, **caso)
        except ValueError:
            continue
        raise AssertionError(f"deveria falhar: {caso}")


def test_otimizador_e_o_adamw_da_configuracao() -> None:
    rede = modelo()
    otimizador = create_optimizer(rede, TINY_TRAINING)
    grupo = otimizador.param_groups[0]

    assert isinstance(otimizador, torch.optim.AdamW)
    assert grupo["lr"] == TINY_TRAINING.learning_rate
    assert grupo["weight_decay"] == TINY_TRAINING.weight_decay
    assert sum(p.numel() for p in grupo["params"]) == rede.parameter_count()


def test_configuracao_de_treino_valida_na_construcao() -> None:
    invalidos = {
        "num_epochs": -1,
        "batch_size": 0,
        "learning_rate": 0.0,
        "weight_decay": -0.1,
        "eval_freq": 0,
        "eval_iter": 0,
        "train_ratio": 1.0,
        "stride": 0,
        "sample_tokens": -1,
    }
    for campo, valor in invalidos.items():
        try:
            replace(BOOK_TRAINING_CONFIG, **{campo: valor})
        except ValueError:
            continue
        raise AssertionError(f"{campo}={valor} deveria falhar")


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
