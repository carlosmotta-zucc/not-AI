"""Etapa 4.1b da Sprint 2: os pares (entrada, alvo) como tensores e lotes.

    Token IDs --sliding_window_pairs--> pares (x, y) --GPTDataset/DataLoader--> (B, T)

`sequences.py` resolve a janela deslizante em Python puro. Este modulo
empacota o mesmo recorte em tensores do PyTorch e monta o DataLoader que
alimenta o treino.

Duas escolhas de projeto merecem registro:

1. GPTDataset fatia sob demanda, em vez de pre-computar. O corpus inteiro vira
   UM tensor e o `__getitem__` calcula o recorte na hora como uma view, nao
   uma copia -- diferente de chamar `sliding_window_pairs` (que materializa
   uma lista por amostra) e converter cada par num tensor separado. Com
   stride=1 isso evita materializar milhares de tensores pequenos para
   representar poucos milhares de Token IDs: o custo de memoria fica
   independente do stride. `expected_sample_count` e reaproveitado so para o
   numero de janelas, nao a formula inteira de recorte.
2. O tokenizador entra por injecao. GPTDataset nunca instancia um: recebe
   qualquer objeto com `encode(str) -> list[int]`. E o que permite rodar o
   mesmo dataset sobre o BPE e sobre o tokenizador proprio, e comparar os
   dois sem duplicar a classe.
"""

from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset

from .sequences import expected_sample_count


class SupportsEncode(Protocol):
    """Interface minima que GPTDataset exige de um tokenizador.

    SimpleTokenizer e BPETokenizer satisfazem os dois, sem heranca nem
    registro: basta ter o metodo.
    """

    def encode(self, text: str) -> list[int]: ...


class GPTDataset(Dataset):
    """Pares (entrada, alvo) extraidos de um texto por janela deslizante.

    Cada amostra e um par de sequencias de `context_length` Token IDs, em que
    o alvo e a entrada deslocada de uma posicao. A janela anda `stride`
    tokens a cada amostra.

    O stride controla a sobreposicao entre amostras vizinhas:
    `stride == context_length` nao sobrepoe nada e cada token aparece uma vez
    por epoca; `stride == 1` gera o maximo de amostras possivel, ao custo de
    cada token reaparecer `context_length` vezes em amostras quase
    identicas.

    Texto curto demais para formar uma janela completa nao e erro: o dataset
    fica com tamanho zero (ver a decisao de projeto em sequences.py).
    """

    def __init__(
        self,
        text: str,
        tokenizer: SupportsEncode,
        context_length: int,
        stride: int,
    ) -> None:
        """Tokeniza `text` e calcula quantas janelas cabem nele.

        Args:
            text: texto bruto, tokenizado uma unica vez aqui.
            tokenizer: qualquer objeto com encode(str) -> list[int].
            context_length: tokens por amostra (o T dos shapes).
            stride: quantos tokens a janela avanca entre amostras.

        Raises:
            ValueError: se context_length ou stride nao forem positivos.
        """
        token_ids = tokenizer.encode(text)

        self._token_ids = torch.tensor(token_ids, dtype=torch.long)
        self._context_length = context_length
        self._stride = stride
        self._window_count = expected_sample_count(len(token_ids), context_length, stride)

    def __len__(self) -> int:
        """Numero de janelas -- o numero de amostras de uma epoca."""
        return self._window_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Devolve o par (entrada, alvo) da janela `index`.

        Args:
            index: posicao da janela; aceita indice negativo como uma lista.

        Returns:
            Par de tensores de shape (context_length,) e dtype torch.long, em
            que o segundo e o primeiro deslocado de uma posicao. Ambos sao
            views do tensor do corpus, nao copias.

        Raises:
            IndexError: se `index` cair fora do numero de janelas.
        """
        if index < 0:
            index += self._window_count
        if not 0 <= index < self._window_count:
            raise IndexError(f"janela {index} fora do intervalo de {self._window_count} janelas")

        start = index * self._stride
        end = start + self._context_length

        return self._token_ids[start:end], self._token_ids[start + 1 : end + 1]

    @property
    def token_ids(self) -> torch.Tensor:
        """Corpus tokenizado inteiro, shape (N,) -- usado pelos experimentos."""
        return self._token_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def stride(self) -> int:
        return self._stride


def create_dataloader(
    text: str,
    tokenizer: SupportsEncode,
    context_length: int,
    stride: int,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Monta o DataLoader que alimenta o treino com lotes de shape (B, T).

    Args:
        text: texto bruto.
        tokenizer: objeto com encode(str) -> list[int].
        context_length: tokens por amostra.
        stride: avanco da janela deslizante.
        batch_size: amostras por lote (o B dos shapes).
        shuffle: embaralha a ordem das janelas a cada epoca.
        drop_last: descarta o ultimo lote quando ele sai incompleto.
        num_workers: processos de carregamento em paralelo.

    Returns:
        DataLoader cujos itens sao pares de tensores de shape
        (batch_size, context_length) e dtype torch.long.

    Os tres ultimos padroes tem motivo, e nao sao os do PyTorch:

    - shuffle=True porque as janelas vizinhas sao trechos consecutivos do
      mesmo texto. Na ordem natural, um lote inteiro cairia dentro do mesmo
      paragrafo e o gradiente refletiria aquele assunto, nao o corpus.
    - drop_last=True porque o ultimo lote de uma epoca costuma vir menor que
      os demais, e a perda media sobre menos amostras tem variancia maior.
      Manter o lote parcial injeta um passo de escala diferente no fim de
      cada epoca.
    - num_workers=0 porque a maquina alvo e CPU sem GPU e o corpus e pequeno:
      o carregamento nao e gargalo, e subir processos extras para fatiar um
      tensor que ja esta em memoria custa mais do que economiza.
    """
    dataset = GPTDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
