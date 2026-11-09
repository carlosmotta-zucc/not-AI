"""Avaliação do modelo pela perda (seção 5.1): cross-entropy e perplexidade.

    logits (B, T, V) --log_softmax--> log-probabilidades (B, T, V)
                     --só a coluna do alvo--> log p(alvo) (B, T)
                     --média, sinal trocado--> cross-entropy (escalar)
                     --exp--> perplexidade

A cross-entropy de um texto é a média, sobre as B·T posições, de -log p(alvo):
em cada posição, o modelo distribuiu probabilidade pelo vocabulário inteiro, e
a perda olha só para o quanto ficou no token que de fato vinha a seguir. Três
leituras do mesmo número:

- p(alvo) = 1 em toda posição daria perda 0. Nenhum modelo chega lá num texto
  de verdade, porque o próximo token não é determinado pelo contexto.
- Espalhar a probabilidade por igual pelos V tokens dá perda ln V -- 10,825
  para o vocabulário do GPT-2. É o patamar do modelo sem treino, e a régua para
  saber se o treino está aprendendo alguma coisa.
- exp(perda) é a perplexidade: o inverso da média geométrica de p(alvo).
  Espalhar por igual entre k candidatos dá perplexidade k, por isso ela se lê
  como "entre quantos tokens, em média, o modelo continua em dúvida".

O livro chega à cross-entropy em seis passos (softmax, probabilidade do alvo,
log, média, sinal) e depois mostra que torch.nn.functional.cross_entropy faz o
mesmo. Aqui `target_log_probabilities` é o passo a passo, e `batch_loss` é a
versão do PyTorch, a que o treino usa. O passo a passo usa log_softmax em vez
de log(softmax): mesma conta, sem a perda de precisão de tirar o log de uma
probabilidade da ordem de 1e-5 já arredondada em float32.

As funções que devolvem float (`loader_loss`, `evaluate`) são avaliação: rodam
em eval() e sem grafo de gradiente, e devolvem o modelo no modo em que o
receberam -- o mesmo contrato do generate() da Sprint 4.
"""

import math
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def target_log_probabilities(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(alvo) em cada posição: o quanto o modelo apostou no token certo.

    Args:
        logits: (B, T, V) -- ou qualquer (..., V).
        targets: (B, T) Token IDs que deveriam sair em cada posição -- o mesmo
            shape dos logits sem a última dimensão.

    Returns:
        (B, T), valores <= 0. A cross-entropy é menos a média deste tensor; o
        exp de cada valor é a probabilidade que o modelo deu ao alvo.

    Raises:
        ValueError: se os shapes não casarem.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} e alvos {tuple(targets.shape)} não "
            "casam: os alvos precisam ter o shape dos logits sem o vocabulário"
        )

    log_probabilities = torch.log_softmax(logits, dim=-1)
    # gather escolhe, em cada posição, a coluna do Token ID alvo: (B, T, 1).
    return log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def batch_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy média de um lote -- o calc_loss_batch do livro.

    Args:
        model: recebe (B, T) e devolve logits (B, T, V). O GPTModel serve.
        inputs: (B, T) Token IDs de entrada.
        targets: (B, T) a entrada deslocada de uma posição.

    Returns:
        Escalar. Com o gradiente ligado, carrega o grafo: é sobre ele que o
        treino chama backward().

    O lote vai para o device do modelo aqui, num lugar só: nem o laço de treino
    nem a avaliação precisam saber onde o modelo está.

    flatten(0, 1) junta lote e tempo, (B, T, V) -> (B·T, V): para a perda, cada
    uma das B·T posições é uma classificação independente entre V classes, e a
    média é sobre todas elas. O modo do modelo fica como está: no treino, o
    dropout participa desta perda.
    """
    device = _device_of(model)
    logits = model(inputs.to(device))
    return F.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())


@torch.no_grad()
def loader_loss(model: nn.Module, loader: DataLoader, max_batches: int | None = None) -> float:
    """Perda média por token sobre os lotes de um DataLoader -- o calc_loss_loader do livro.

    Args:
        model: o GPTModel.
        loader: DataLoader de pares (entrada, alvo), cada um (B, T).
        max_batches: usa só os primeiros lotes (o `num_batches` do livro, que o
            treino chama de `eval_iter`); None = todos.

    Returns:
        A perda como float; NaN se não houver lote nenhum.

    Raises:
        ValueError: se max_batches < 1.

    Duas diferenças para o livro, nenhuma que mude os números dele:

    - Roda em eval() e sob no_grad(), e devolve o modo original no fim. O livro
      deixa as duas coisas para quem chama; aqui não há como medir com o
      dropout ligado por esquecimento.
    - Cada lote pesa pelo número de tokens. O livro tira a média das médias
      por lote; as duas coincidem quando todos os lotes têm o mesmo tamanho
      (sempre o caso no livro) e diferem quando o último vem incompleto
      (drop_last=False). Aí só a ponderada é a perda média por token, que é o
      que a perplexidade pressupõe.

    Cada chamada cria UM iterador do loader, como o livro: com shuffle=True,
    isso consome o gerador global, e é parte do que deixa o treino reproduzir
    os números do capítulo.
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError(f"max_batches={max_batches} precisa ser positivo ou None")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    try:
        for inputs, targets in islice(loader, max_batches):
            tokens = targets.numel()
            total_loss += batch_loss(model, inputs, targets).item() * tokens
            total_tokens += tokens
    finally:
        model.train(was_training)

    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Perdas de treino e de validação -- o evaluate_model do livro.

    Args:
        model: o GPTModel.
        train_loader: lotes de treino.
        val_loader: lotes de validação.
        max_batches: lotes por conjunto; None = os conjuntos inteiros.

    Returns:
        (perda de treino, perda de validação).

    Treino antes de validação, como no livro: a ordem decide em que ponto do
    gerador global cada iterador é criado.
    """
    train_loss = loader_loss(model, train_loader, max_batches)
    val_loss = loader_loss(model, val_loader, max_batches)
    return train_loss, val_loss


def perplexity(loss: float | torch.Tensor) -> float:
    """exp(perda): o número efetivo de candidatos entre os quais o modelo hesita.

    Args:
        loss: cross-entropy média por token, em nats (log natural).

    Returns:
        A perplexidade. 1 é certeza no token correto; V (50 257 no GPT-2) é a
        dúvida de quem sorteia uniformemente. Perda grande demais para o exp
        do float64 (> ~709) dá infinito em vez de erro.

    Só vale para perda média POR TOKEN e em nats: a média por sequência ou o
    log em base 2 dariam outro número com o mesmo nome.
    """
    try:
        return math.exp(float(loss))
    except OverflowError:
        return math.inf
