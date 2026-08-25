"""Etapa 3.1 da Sprint 2: quebra de texto bruto em tokens.

Primeira operacao do pipeline. Recebe uma string e devolve a lista de unidades
que o restante do modelo passa a enxergar no lugar do texto.

A regra aqui e explicita e escrita a mao, sem tabela de fusoes aprendida do
corpus (isso e o BPE, em bpe.py). Duas decisoes de projeto importam:

1. Reconhecer, em vez de separar. O livro usa re.split com o delimitador
   capturado e depois joga fora as strings vazias que a operacao produz. Aqui a
   busca e por re.findall: descrevemos o que E token, e nao o que separa
   tokens. O resultado e equivalente e nao ha lista intermediaria para limpar.

2. Pontuacao e token, espaco nao e. Pontuacao carrega informacao sintatica
   ("vamos comer, vovo" e "vamos comer vovo" nao sao a mesma frase), entao
   vira token proprio. Espaco em branco so marca a fronteira entre tokens e e
   descartado por padrao, o que encurta a sequencia. A escolha e reversivel
   pelo parametro keep_whitespace: texto onde a indentacao significa algo
   (codigo Python, por exemplo) precisaria dela preservada.
"""

import re

# A ordem das alternativas e o que faz o padrao funcionar: o regex do Python
# testa da esquerda para a direita e para na primeira que casar. Por isso os
# casos compostos ("--", "...") vem antes dos sinais isolados, senao seriam
# lidos como dois ou tres tokens de um caractere.
_TOKEN_PATTERN = re.compile(
    r"""
      \d+(?:[.,]\d+)+   # numero com separador interno: 3.14, 1.500,00
    | \w+               # palavra (\w e unicode: cobre acentuacao e cedilha)
    | --                # travessao duplo, usado como pontuacao no corpus
    | \.\.\.            # reticencias
    | [^\w\s]           # qualquer outro sinal isolado
    | \s+               # espacos, descartados salvo pedido explicito
    """,
    re.VERBOSE,
)


def split_text(text: str, keep_whitespace: bool = False) -> list[str]:
    """Quebra `text` na lista de tokens.

    Args:
        text: texto bruto, de qualquer tamanho.
        keep_whitespace: se True, sequencias de espaco viram tokens tambem.

    Returns:
        Lista de tokens na ordem em que aparecem no texto. Concatenar o
        resultado com keep_whitespace=True reconstroi o texto original
        exatamente; com keep_whitespace=False, o espacamento se perde.
    """
    tokens = _TOKEN_PATTERN.findall(text)

    if keep_whitespace:
        return tokens
    return [token for token in tokens if not token.isspace()]


def count_tokens(text: str) -> int:
    """Conta tokens sem materializar a lista inteira.

    Util nos experimentos de contagem, onde o corpus e percorrido varias vezes
    e so o total interessa.
    """
    return sum(1 for match in _TOKEN_PATTERN.finditer(text) if not match.group().isspace())


def unique_tokens(text: str) -> set[str]:
    """Devolve o conjunto de tokens distintos de `text`.

    E a base da construcao do vocabulario: o que define o tamanho do
    vocabulario nao e quantos tokens o corpus tem, e sim quantos sao
    diferentes entre si.
    """
    return set(split_text(text))
