"""Etapa 3.2 da Sprint 2: vocabulario e Token IDs.

O vocabulario e a ponte entre token (string) e Token ID (inteiro). Sao dois
dicionarios espelhados: um para ir, outro para voltar.

O ID e um rotulo nominal. Ele sai da posicao do token na lista ordenada, entao
a magnitude nao significa nada: se "rei" e 902 e "rainha" e 1503, a diferenca
601 nao mede distancia semantica nenhuma. O ID existe para uma unica coisa,
que so acontece na Sprint seguinte deste modulo: indexar a linha certa da
tabela de embeddings.

Consequencia pratica do vocabulario ser fechado: todo token que nao estava no
corpus de construcao nao tem ID. Ha duas saidas, e este modulo suporta as
duas -- falhar alto (KeyError) ou colapsar tudo em <|unk|>.
"""

from collections.abc import Iterable, Iterator

from .splitter import split_text

# Tokens que nao vem do texto: descrevem a sequencia em vez de fazer parte
# dela. Ambos aparecem no capitulo 2; o GPT-2 real so usa <|endoftext|>,
# porque o BPE dispensa o <|unk|>.
END_OF_TEXT_TOKEN = "<|endoftext|>"
UNKNOWN_TOKEN = "<|unk|>"

DEFAULT_SPECIAL_TOKENS = (END_OF_TEXT_TOKEN, UNKNOWN_TOKEN)


class Vocabulary:
    """Mapeamento bidirecional entre tokens e Token IDs.

    Os tokens do corpus recebem IDs em ordem alfabetica e os tokens especiais
    sao acrescentados no fim. A ordem e arbitraria, mas fixa-la torna o
    vocabulario reproduzivel: o mesmo corpus sempre gera os mesmos IDs, o que e
    condicao para comparar experimentos entre execucoes.

    Especiais no fim, e nao no comeco, por um motivo pratico: acrescentar um
    token especial novo nao desloca o ID de nenhum token ja existente.
    """

    def __init__(
        self,
        tokens: Iterable[str],
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        """Monta o vocabulario a partir de tokens ja extraidos.

        Args:
            tokens: tokens do corpus; repeticoes sao ignoradas.
            special_tokens: tokens acrescentados ao fim, na ordem dada.
        """
        corpus_tokens = sorted(set(tokens))
        specials = [token for token in special_tokens if token not in corpus_tokens]

        ordered = corpus_tokens + specials

        self._token_to_id: dict[str, int] = {token: index for index, token in enumerate(ordered)}
        self._id_to_token: list[str] = ordered
        self._special_tokens: tuple[str, ...] = tuple(specials)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> "Vocabulary":
        """Constroi o vocabulario direto de textos brutos, tokenizando antes."""
        tokens: list[str] = []
        for text in texts:
            tokens.extend(split_text(text))
        return cls(tokens, special_tokens=special_tokens)

    def __len__(self) -> int:
        """Tamanho do vocabulario -- numero de linhas da futura tabela de embeddings."""
        return len(self._id_to_token)

    def __contains__(self, token: object) -> bool:
        return token in self._token_to_id

    def __iter__(self) -> Iterator[tuple[str, int]]:
        """Percorre os pares (token, id) em ordem crescente de ID."""
        return iter((token, index) for index, token in enumerate(self._id_to_token))

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return self._special_tokens

    @property
    def has_unknown_token(self) -> bool:
        """Indica se o vocabulario sabe representar palavra fora dele."""
        return UNKNOWN_TOKEN in self._token_to_id

    def token_to_id(self, token: str) -> int:
        """Converte um token no seu Token ID.

        Se o token nao existe e <|unk|> esta no vocabulario, devolve o ID de
        <|unk|>. Se nao esta, propaga KeyError -- o comportamento do
        tokenizador do livro antes da V2, mantido de proposito para que o
        experimento 2 possa mostrar a falha em vez de descrever ela.
        """
        try:
            return self._token_to_id[token]
        except KeyError:
            if self.has_unknown_token:
                return self._token_to_id[UNKNOWN_TOKEN]
            raise

    def id_to_token(self, token_id: int) -> str:
        """Converte um Token ID de volta no token correspondente."""
        try:
            return self._id_to_token[token_id]
        except IndexError as error:
            raise KeyError(f"Token ID {token_id} fora do vocabulario de {len(self)} entradas") from error

    def preview(self, count: int = 10, from_end: bool = False) -> list[tuple[str, int]]:
        """Devolve os primeiros (ou ultimos) pares (token, id).

        So para inspecao manual nos experimentos e no relatorio.
        """
        pairs = list(self)
        return pairs[-count:] if from_end else pairs[:count]
