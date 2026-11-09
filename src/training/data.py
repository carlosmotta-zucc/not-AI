"""Os dados do treino (seção 5.1.3): divisão em treino e validação, e os dois DataLoaders.

    texto --split_text--> (texto de treino, texto de validação)
          --create_dataloader (Sprint 2)--> train_loader: embaralha, descarta o lote incompleto
                                            val_loader:   ordem fixa, usa todos os lotes

A divisão é feita nos CARACTERES, antes de tokenizar, como no livro: os
primeiros 90% do texto treinam, os 10% finais validam. Um corte só, no fim do
texto, e não trechos sorteados: assim nenhum token da validação aparece em
alguma janela de treino, e a perda de validação mede o que o modelo faz com
texto que nunca viu -- não o quanto ele decorou. Com trechos sorteados e
janelas sobrepostas, vizinhos de treino compartilhariam tokens com a validação.

O custo do corte único é que treino e validação não vêm da mesma "região" do
conto. Com um texto de 20 mil caracteres, os 10% finais são o desfecho da
história, e parte da diferença entre as duas perdas pode vir disso, não só de
sobreajuste. Os experimentos registram essa ressalva.

A validação usa sempre janelas sem sobreposição (stride = context_length),
mesmo quando o treino usa outro stride: a perda de validação tem que ser a
mesma régua em todos os experimentos, e janelas sobrepostas contariam os
mesmos tokens várias vezes, com contextos diferentes.
"""

from torch.utils.data import DataLoader

from src.dataset import SupportsEncode, create_dataloader

# Fração do livro (p. 142): 90% treino, 10% validação.
TRAIN_RATIO = 0.9


def split_text(text: str, train_ratio: float = TRAIN_RATIO) -> tuple[str, str]:
    """Corta o texto em dois, na posição train_ratio · len(text).

    Args:
        text: texto bruto.
        train_ratio: fração dos caracteres que vai para o treino.

    Returns:
        (texto de treino, texto de validação). Concatenados, dão o original.

    Raises:
        ValueError: se train_ratio não estiver em (0, 1).

    O corte pode cair no meio de uma palavra. Não há problema: os dois
    pedaços são tokenizados separadamente, e o BPE fatia qualquer fragmento.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio={train_ratio} fora de (0, 1)")

    split_index = int(train_ratio * len(text))
    return text[:split_index], text[split_index:]


def create_train_val_loaders(
    text: str,
    tokenizer: SupportsEncode,
    context_length: int,
    batch_size: int,
    train_ratio: float = TRAIN_RATIO,
    stride: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Monta os DataLoaders de treino e de validação a partir de um texto só.

    Args:
        text: texto bruto do corpus.
        tokenizer: objeto com encode(str) -> list[int]; o BPETokenizer serve.
        context_length: tokens por amostra (o T dos shapes); normalmente o
            context_length do modelo.
        batch_size: amostras por lote, nos dois loaders.
        train_ratio: fração dos caracteres para o treino.
        stride: avanço da janela no treino; None = context_length.

    Returns:
        (train_loader, val_loader), ambos de pares (B, T).

    Raises:
        ValueError: se algum dos dois conjuntos não formar nem uma janela --
            treinar ou validar sobre zero lotes não daria erro nenhum depois,
            só uma perda NaN ou um treino que não faz nada.

    Os padrões são os do livro (p. 143), e cada um tem motivo:

    - treino com shuffle=True: janelas vizinhas são trechos seguidos do mesmo
      texto, e um lote em ordem natural cairia inteiro no mesmo parágrafo;
    - treino com drop_last=True: o lote final incompleto seria um passo com
      outra variância no fim de cada época;
    - validação com shuffle=False e drop_last=False: a ordem não muda a média,
      e com 10% de um texto curto não há lote sobrando para descartar.
    """
    if stride is None:
        stride = context_length

    train_text, val_text = split_text(text, train_ratio)

    train_loader = create_dataloader(
        train_text,
        tokenizer,
        context_length=context_length,
        stride=stride,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_text,
        tokenizer,
        context_length=context_length,
        stride=context_length,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    for nome, loader in (("treino", train_loader), ("validação", val_loader)):
        if len(loader) == 0:
            tokens = len(loader.dataset.token_ids)
            raise ValueError(
                f"o conjunto de {nome} não forma nenhum lote: {tokens} tokens, "
                f"context_length={context_length}, batch_size={batch_size}. "
                "Reduza o contexto ou o lote, ou ajuste train_ratio"
            )

    return train_loader, val_loader
