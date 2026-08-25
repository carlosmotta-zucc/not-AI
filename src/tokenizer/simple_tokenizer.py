"""Etapa 3.2 da Sprint 2: encode e decode sobre o vocabulario proprio.

Junta as duas pecas anteriores: o splitter quebra o texto em tokens, o
vocabulario troca cada token pelo seu ID.

    texto  --split_text-->  tokens  --vocabulario-->  Token IDs

O caminho de volta nao e simetrico. A quebra descarta o espacamento original,
entao o decode precisa remontar os espacos por heuristica -- colar a pontuacao
na palavra anterior, e assim por diante. O texto reconstruido e legivel, mas
nao e byte a byte o original. Essa perda e uma propriedade do tokenizador por
palavras, nao um bug: o BPE, que trabalha sobre bytes e carrega o espaco
dentro do proprio token, nao tem esse problema.
"""

import re

from .splitter import split_text
from .vocabulary import Vocabulary

# Heuristicas de remontagem do espacamento no decode. Depois do join por
# espaco simples, o texto fica "Ola , mundo ." e estas regras o aproximam de
# "Ola, mundo.".
_SPACE_BEFORE_CLOSING = re.compile(r"\s+([,.:;?!%)\]}])")
_SPACE_AFTER_OPENING = re.compile(r"([(\[{$])\s+")
_SPACED_APOSTROPHE = re.compile(r"(\w)\s+'\s+(\w)")


class SimpleTokenizer:
    """Tokenizador por palavras com vocabulario fechado.

    Fechado quer dizer: so representa o que estava no corpus de construcao.
    Palavra nova vira <|unk|> (se o vocabulario tiver esse token) ou levanta
    KeyError (se nao tiver).
    """

    def __init__(self, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary

        # Tokens especiais precisam ser reconhecidos ANTES da regex de quebra.
        # "<|endoftext|>" passado pelo splitter viraria ['<', '|', 'endoftext',
        # '|', '>'] -- cinco tokens comuns, e o marcador se perderia. Por isso
        # o texto e primeiro fatiado nas ocorrencias dos especiais, e so os
        # pedacos entre eles vao para a tokenizacao normal. E o mesmo motivo do
        # parametro allowed_special do tiktoken.
        specials = vocabulary.special_tokens
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(token) for token in specials) + ")")
            if specials
            else None
        )

    def encode(self, text: str) -> list[int]:
        """Converte texto bruto em lista de Token IDs.

        Args:
            text: texto de entrada; pode conter tokens especiais literais.

        Returns:
            Lista de inteiros, um por token reconhecido.

        Raises:
            KeyError: se aparecer token fora do vocabulario e o vocabulario nao
                tiver <|unk|>.
        """
        ids: list[int] = []

        for chunk, is_special in self._segments(text):
            if is_special:
                ids.append(self.vocabulary.token_to_id(chunk))
            else:
                ids.extend(self.vocabulary.token_to_id(token) for token in split_text(chunk))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Converte Token IDs de volta em texto legivel.

        A saida nao e identica a entrada original do encode: o espacamento e
        reconstruido por heuristica. Ver docstring do modulo.
        """
        tokens = [self.vocabulary.id_to_token(token_id) for token_id in ids]
        text = " ".join(tokens)

        text = _SPACE_BEFORE_CLOSING.sub(r"\1", text)
        text = _SPACE_AFTER_OPENING.sub(r"\1", text)
        text = _SPACED_APOSTROPHE.sub(r"\1'\2", text)

        return text.strip()

    def tokenize(self, text: str) -> list[str]:
        """Mostra os tokens antes da conversao em IDs.

        Nao faz parte do pipeline do modelo; existe para inspecionar o que o
        encode enxergou, sobretudo quando o resultado tem <|unk|>.
        """
        tokens: list[str] = []
        for chunk, is_special in self._segments(text):
            if is_special:
                tokens.append(chunk)
            else:
                tokens.extend(split_text(chunk))
        return tokens

    def _segments(self, text: str) -> list[tuple[str, bool]]:
        """Fatia `text` em pedacos, marcando quais sao tokens especiais.

        Returns:
            Lista de pares (trecho, e_especial), na ordem do texto.
        """
        if self._special_pattern is None:
            return [(text, False)]

        specials = set(self.vocabulary.special_tokens)
        return [
            (part, part in specials)
            for part in self._special_pattern.split(text)
            if part  # split com grupo capturado devolve strings vazias nas bordas
        ]
