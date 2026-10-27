"""Geração autoregressiva (seção 4.7): prever, anexar, repetir.

    contexto (B, T)
      -> recorta os últimos context_length tokens
      -> GPTModel -> logits (B, T, V) -> só a última linha (B, V)
      -> select_next_token -> próximo token (B, 1)
      -> anexa ao contexto -> (B, T+1) -> volta ao início

O modelo produz uma previsão por posição, mas na geração só a última interessa:
as outras são previsões de tokens que já estão no contexto. O token escolhido
vira entrada da passada seguinte -- é isso que "autoregressivo" quer dizer: a
saída de um passo é a entrada do próximo.

Duas consequências do laço, que os experimentos medem:

- O contexto cresce um token por passo até context_length; dali em diante a
  janela desliza e o começo do texto é esquecido. Sem o recorte, a tabela de
  posições não teria linha para a posição context_length e o modelo levantaria
  ValueError.
- Cada passo refaz a passada inteira sobre o contexto, embora só um token seja
  novo. Com a máscara causal, os logits das posições antigas saem iguais aos do
  passo anterior; um cache de chaves e valores evitaria refazê-los, e fica fora
  do escopo do livro.
"""

from typing import Protocol

import torch
import torch.nn as nn

from .decoding import GREEDY_TEMPERATURE, select_next_token


class SupportsTokenizer(Protocol):
    """O que as funções de texto exigem do tokenizador: o BPETokenizer serve."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@torch.no_grad()
def generate(
    model: nn.Module,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int | None = None,
    temperature: float = GREEDY_TEMPERATURE,
    top_k: int | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Estende cada sequência do lote, um token por passo.

    Args:
        model: recebe (B, T) e devolve logits (B, T, V). O GPTModel serve.
        token_ids: contexto inicial (B, T), dtype torch.long.
        max_new_tokens: quantos passos de geração, no máximo.
        context_length: quantos tokens finais o modelo enxerga por passo. Por
            padrão, o `context_length` do próprio modelo.
        temperature: 0 é a decodificação gulosa do livro (generate_text_simple);
            acima de 0, amostragem (seção 5.3).
        top_k: restringe a amostragem aos k tokens mais prováveis.
        eos_id: token de fim de texto. A sequência que o gera para ali: o eos
            é anexado uma vez e, se o lote continuar, ela é completada com ele.
            Quando todas param, o laço termina antes de max_new_tokens.
        generator: torch.Generator para sorteio reprodutível.

    Returns:
        Tensor (B, T + n), n <= max_new_tokens: o contexto original seguido
        dos tokens gerados.

    Raises:
        ValueError: se token_ids não for 2D ou max_new_tokens for negativo.

    A função coloca o modelo em eval() e devolve o modo original no fim, mesmo
    se algo falhar no meio: com dropout ligado, a decodificação gulosa deixaria
    de ser determinística e a amostragem ganharia um ruído que ninguém pediu.
    O decorador no_grad desliga o grafo de gradiente -- nada aqui vai ter
    backward, e guardar ativações para ele só gastaria memória.
    """
    if token_ids.dim() != 2:
        raise ValueError(
            f"token_ids precisa ser (B, T); veio {tuple(token_ids.shape)}. "
            "Para uma sequência só, use .unsqueeze(0)"
        )
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens={max_new_tokens} não pode ser negativo")

    if context_length is None:
        context_length = model.context_length

    was_training = model.training
    model.eval()

    finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)

    try:
        for _ in range(max_new_tokens):
            # Janela deslizante: só os últimos context_length tokens entram.
            context = token_ids[:, -context_length:]

            # (B, T, V) -> (B, V): a última posição prevê o token seguinte.
            last_logits = model(context)[:, -1, :]
            next_ids = select_next_token(last_logits, temperature, top_k, generator)

            if eos_id is not None:
                # Quem já terminou recebe eos de novo, para o lote continuar
                # retangular; quem gerou eos agora passa a constar como pronto.
                next_ids = next_ids.masked_fill(finished.unsqueeze(-1), eos_id)
                finished |= next_ids.squeeze(-1) == eos_id

            token_ids = torch.cat([token_ids, next_ids], dim=1)

            if bool(finished.all()):
                break
    finally:
        model.train(was_training)

    return token_ids


def text_to_token_ids(text: str, tokenizer: SupportsTokenizer) -> torch.Tensor:
    """Texto -> (1, T) de Token IDs, já com a dimensão de lote que o modelo exige."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: SupportsTokenizer) -> str:
    """(1, T) ou (T,) de Token IDs -> texto.

    Raises:
        ValueError: se vier um lote com mais de uma sequência -- decodificar
            tudo junto colaria textos diferentes num só.
    """
    if token_ids.dim() == 2:
        if token_ids.shape[0] != 1:
            raise ValueError(
                f"esperado lote de 1 sequência; veio {token_ids.shape[0]}. "
                "Decodifique cada linha separadamente"
            )
        token_ids = token_ids.squeeze(0)

    return tokenizer.decode(token_ids.tolist())


def generate_text(
    model: nn.Module,
    tokenizer: SupportsTokenizer,
    prompt: str,
    max_new_tokens: int,
    **generation_options,
) -> str:
    """Atalho texto -> texto: codifica o prompt, gera e decodifica tudo.

    Args:
        model: o GPTModel.
        tokenizer: o BPETokenizer.
        prompt: contexto inicial.
        max_new_tokens: tokens a gerar.
        **generation_options: repassadas a generate() (temperature, top_k,
            eos_id, generator, context_length).

    Returns:
        O prompt seguido do texto gerado.
    """
    token_ids = generate(
        model,
        text_to_token_ids(prompt, tokenizer),
        max_new_tokens,
        **generation_options,
    )
    return token_ids_to_text(token_ids, tokenizer)
