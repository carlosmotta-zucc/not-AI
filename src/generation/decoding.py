"""Do vetor de logits ao próximo token: as estratégias de seleção.

    logits (B, V) --top-k--> logits filtrados --÷ temperatura--> --softmax--> probabilidades (B, V)
                                                                                   |
                                    temperatura = 0: argmax          temperatura > 0: sorteio
                                                                                   |
                                                                      próximo token (B, 1)

Logit, probabilidade e escolha são três coisas diferentes:

- O LOGIT é a saída crua da cabeça de saída, um número real por token do
  vocabulário. Só a ordem e as diferenças entre eles importam: somar uma
  constante a todos não muda nada depois do softmax.
- A PROBABILIDADE é o softmax dos logits: p_i = exp(z_i) / Σ_j exp(z_j). É o
  que a temperatura e o top-k alteram.
- A ESCOLHA transforma a distribuição em um token. Gulosa (argmax): sempre o
  mais provável. Amostragem (multinomial): cada token com chance p_i.

O softmax é monotônico, então o argmax dos logits é o argmax das
probabilidades: na decodificação gulosa ele é dispensável, e por isso a
temperatura 0 nem o calcula.

As três funções públicas recebem (B, V) e são independentes do modelo: os
testes e o exemplo do vocabulário de 9 palavras da seção 5.3 do livro as usam
com logits escritos à mão.
"""

import torch

# Temperatura 0 é o código para "gulosa". O livro usa o mesmo corte
# (`if temperature > 0.0`), e o exercício 5.3 depende dele.
GREEDY_TEMPERATURE = 0.0


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mantém os `top_k` maiores logits de cada linha e manda o resto para -inf.

    Args:
        logits: (..., V).
        top_k: quantos candidatos sobrevivem; acima de V, vale V.

    Returns:
        Tensor do mesmo shape. Depois do softmax, os tokens fora do top-k têm
        probabilidade zero exato -- o mesmo truque da máscara causal da Sprint 3.

    Raises:
        ValueError: se top_k < 1.

    Mantém EXATAMENTE k por linha, escrevendo os k valores de volta nas suas
    posições. O livro compara com o menor dos k (`logits < min_val`), o que
    mantém mais de k quando há empate na fronteira; sem empate, os dois dão o
    mesmo resultado.
    """
    if top_k < 1:
        raise ValueError(f"top_k={top_k} precisa ser pelo menos 1")

    top_k = min(top_k, logits.shape[-1])
    values, positions = torch.topk(logits, top_k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, positions, values)


def next_token_probabilities(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Transforma logits na distribuição da qual o próximo token é sorteado.

    Args:
        logits: (..., V).
        temperature: divide os logits antes do softmax. < 1 concentra a
            distribuição, > 1 achata, 1 não muda nada. Precisa ser > 0.
        top_k: se dado, só os k maiores ficam com probabilidade não nula.

    Returns:
        Probabilidades (..., V), cada linha somando 1.

    Raises:
        ValueError: se temperature <= 0 ou top_k < 1.

    O top-k vem antes da temperatura, como no livro. A ordem não muda quais
    tokens sobrevivem: dividir por um número positivo não altera a ordem dos
    logits.
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature={temperature}: a distribuição só existe para "
            "temperatura positiva (0 é a escolha gulosa, sem distribuição)"
        )

    if top_k is not None:
        logits = apply_top_k(logits, top_k)

    return torch.softmax(logits / temperature, dim=-1)


def select_next_token(
    logits: torch.Tensor,
    temperature: float = GREEDY_TEMPERATURE,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Escolhe o próximo token de cada sequência do lote.

    Args:
        logits: (B, V), os logits da ÚLTIMA posição de cada sequência.
        temperature: 0 é a escolha gulosa (argmax); acima de 0, amostragem
            da distribuição ajustada por temperatura e top-k.
        top_k: restringe a amostragem aos k mais prováveis. Na gulosa não
            muda nada -- o argmax já está no top-k.
        generator: fonte de aleatoriedade do sorteio. Com um
            torch.Generator semeado, a mesma chamada sorteia os mesmos tokens
            sem mexer na seed global.

    Returns:
        Token IDs (B, 1), dtype torch.long -- o shape que o laço de geração
        concatena na sequência.

    Raises:
        ValueError: se temperature < 0 ou top_k < 1.
    """
    if temperature < 0:
        raise ValueError(f"temperature={temperature} não pode ser negativa")

    if temperature == GREEDY_TEMPERATURE:
        # Valida o top_k mesmo sem usá-lo: um valor inválido não pode passar
        # só porque a temperatura escondeu.
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k={top_k} precisa ser pelo menos 1")
        return logits.argmax(dim=-1, keepdim=True)

    probabilities = next_token_probabilities(logits, temperature, top_k)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)
