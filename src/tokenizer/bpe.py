"""Adaptador do BPE do GPT-2 (tiktoken) para o comparativo da Sprint 2.

O tokenizador proprio (splitter + Vocabulary + SimpleTokenizer) e didatico:
mostra cada peca do caminho texto -> Token ID. O BPE e o que o GPT-2 usa de
verdade. Comparar os dois so e honesto se ambos expuserem a mesma interface,
e e isso que este modulo faz -- nao reimplementa o BPE, embrulha o tiktoken
com os mesmos metodos encode/decode/tokenize do SimpleTokenizer.

Diferenca estrutural que o comparativo evidencia: o BPE nao tem vocabulario
fechado no sentido do outro. Palavra que ele nunca viu e fatiada em subwords
ate, no pior caso, bytes soltos -- e byte sempre existe. Por isso ele nao
precisa de <|unk|> e o decode devolve o texto original exatamente, espacos
inclusive, ja que o espaco viaja dentro do proprio token.
"""

from functools import lru_cache

import tiktoken

from .vocabulary import END_OF_TEXT_TOKEN

GPT2_ENCODING_NAME = "gpt2"


@lru_cache(maxsize=None)
def load_gpt2_encoding() -> tiktoken.Encoding:
    """Carrega o BPE do GPT-2.

    Em cache porque a primeira chamada baixa e monta a tabela de fusoes; os
    experimentos instanciam o tokenizador varias vezes e nao ha motivo para
    pagar esse custo mais de uma vez.
    """
    return tiktoken.get_encoding(GPT2_ENCODING_NAME)


class BPETokenizer:
    """Fachada sobre o tiktoken com a interface do SimpleTokenizer."""

    def __init__(self) -> None:
        self._encoding = load_gpt2_encoding()

    def __len__(self) -> int:
        """Tamanho do vocabulario do GPT-2: 50 257 entradas."""
        return self._encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        """Converte texto em Token IDs do GPT-2.

        allowed_special libera o <|endoftext|> literal. Sem isso o tiktoken
        levanta excecao ao encontrar o marcador no texto -- protecao contra
        entrada nao confiavel injetar tokens de controle.
        """
        return self._encoding.encode(text, allowed_special={END_OF_TEXT_TOKEN})

    def decode(self, ids: list[int]) -> str:
        """Converte Token IDs de volta em texto, sem perda."""
        return self._encoding.decode(ids)

    def tokenize(self, text: str) -> list[str]:
        """Mostra os tokens como strings, com o espaco a esquerda preservado.

        E aqui que a diferenca para o tokenizador proprio fica visivel: " the"
        e um token, diferente de "the". O espaco nao foi descartado, virou
        parte do token.
        """
        return [self._encoding.decode([token_id]) for token_id in self.encode(text)]
