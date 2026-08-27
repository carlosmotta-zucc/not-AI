"""Etapa 4.1a da Sprint 2: da sequencia plana de Token IDs aos pares (entrada, alvo).

Python puro, sem torch e sem tokenizador: recebe a lista de Token IDs ja pronta
e devolve os pares de janela deslizante. `GPTDataset` (loader.py) empacota o
mesmo recorte em tensores; esta camada fica separada para poder testar e medir
o numero de amostras sem depender de PyTorch nem de tokenizar nada de novo.

Decisao de projeto: o que fazer com o pedaco final incompleto.

Uma janela de entrada de tamanho `context_length` so e valida se existe TAMBEM
um alvo completo -- o proximo Token ID apos o ultimo da entrada. Isso exige
`context_length + 1` Token IDs a partir do inicio da janela. Quando a sequencia
termina antes disso, a janela final e DESCARTADA, nunca preenchida com
padding: um token de padding do lado do alvo ensinaria o modelo a prever algo
que nao e linguagem, contaminando a propria definicao da tarefa (prever o
proximo token real). O custo e no maximo `context_length` Token IDs perdidos
por chamada -- irrelevante perto do corpus inteiro.

No limite, se a sequencia nao tiver Token IDs suficientes nem para UMA janela,
o resultado e zero amostras, nao uma excecao. "Sem amostras" e um estado
valido (lista vazia, dataset de tamanho 0) e nao um erro de configuracao; quem
decide se isso e aceitavel e quem consome o resultado (um DataLoader sobre um
dataset vazio simplesmente nao itera), nao esta funcao.
"""


def expected_sample_count(id_count: int, context_length: int, stride: int) -> int:
    """Numero de janelas que cabem em uma sequencia de `id_count` Token IDs.

    Calcula sem gerar nenhuma amostra. Usado por `sliding_window_pairs` e por
    `GPTDataset` para nao duplicar a formula, pelos testes para validar as
    duas, e pelos experimentos para medir o efeito de stride/context_length
    sem pagar o custo de recortar a sequencia inteira so para contar o
    resultado.

    Args:
        id_count: quantos Token IDs a sequencia tem.
        context_length: tokens por amostra.
        stride: quantos tokens a janela avanca entre amostras.

    Returns:
        Quantas janelas completas (entrada + alvo) cabem na sequencia. Zero se
        nao houver Token IDs suficientes para nenhuma janela -- ver a decisao
        de projeto no topo do modulo.

    Raises:
        ValueError: se context_length ou stride nao forem positivos.
    """
    if context_length < 1:
        raise ValueError(f"context_length precisa ser >= 1, recebido {context_length}")
    if stride < 1:
        raise ValueError(f"stride precisa ser >= 1, recebido {stride}")

    ids_utilizaveis = id_count - context_length - 1
    if ids_utilizaveis < 0:
        return 0

    return ids_utilizaveis // stride + 1


def sliding_window_pairs(
    token_ids: list[int], context_length: int, stride: int
) -> list[tuple[list[int], list[int]]]:
    """Recorta `token_ids` em pares (entrada, alvo) por janela deslizante.

    Args:
        token_ids: sequencia completa e continua de Token IDs.
        context_length: tokens por amostra.
        stride: quantos tokens a janela avanca entre amostras.

    Returns:
        Lista de pares (entrada, alvo), cada um com `context_length` Token
        IDs, em que o alvo e a entrada deslocada de uma posicao:
        `alvo[i] == entrada[i + 1]` para `i < context_length - 1`, e
        `alvo[-1]` e o Token ID logo apos a janela de entrada. Lista vazia se
        `token_ids` for curto demais para uma janela.

    Raises:
        ValueError: se context_length ou stride nao forem positivos.
    """
    total_amostras = expected_sample_count(len(token_ids), context_length, stride)

    pares = []
    for indice in range(total_amostras):
        inicio = indice * stride
        fim = inicio + context_length
        pares.append((token_ids[inicio:fim], token_ids[inicio + 1 : fim + 1]))

    return pares
