"""Testes da Frente A da Sprint 2: do texto ao Token ID.

Cobrem as decisoes de projeto que o relatorio afirma -- que pontuacao vira
token, que espaco e descartavel sem perder tokens, que ID sem vocabulario nao
existe, que <|unk|> colapsa palavras distintas, que o BPE fecha o ciclo sem
perda. Se algum desses testes cair, uma afirmacao da analise deixou de ser
verdadeira.

Uso:
    python tests/test_tokenizer.py     # sem dependencia extra
    pytest tests/                      # se pytest estiver instalado
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import (  # noqa: E402
    END_OF_TEXT_TOKEN,
    UNKNOWN_TOKEN,
    BPETokenizer,
    SimpleTokenizer,
    Vocabulary,
    count_tokens,
    split_text,
)


# --------------------------------------------------------------------------
# 3.1 Tokenizacao
# --------------------------------------------------------------------------


def test_pontuacao_vira_token_separado() -> None:
    assert split_text("Ola, mundo.") == ["Ola", ",", "mundo", "."]


def test_espaco_nao_vira_token() -> None:
    # Espacamento irregular nao pode mudar a sequencia de tokens: a quebra
    # depende do conteudo, nao da formatacao.
    espacado = split_text("Ola,   mundo.\n\n Tudo bem ?")
    compacto = split_text("Ola, mundo. Tudo bem?")
    assert espacado == compacto


def test_keep_whitespace_reconstroi_o_texto_original() -> None:
    original = "Ola,   mundo.\nTudo bem?"
    assert "".join(split_text(original, keep_whitespace=True)) == original


def test_acentuacao_nao_quebra_a_palavra() -> None:
    # \w e unicode: se nao fosse, "coracao" acentuado viraria tres tokens.
    assert split_text("coração e limão") == ["coração", "e", "limão"]


def test_travessao_duplo_e_um_token_so() -> None:
    # Ordem das alternativas do regex: "--" precisa casar antes de "-".
    assert split_text("genius--though") == ["genius", "--", "though"]


def test_reticencias_e_um_token_so() -> None:
    assert split_text("Espere...") == ["Espere", "..."]


def test_numero_com_separador_fica_inteiro() -> None:
    assert split_text("custou 1.500,00 reais") == ["custou", "1.500,00", "reais"]


def test_numero_no_fim_da_frase_nao_engole_o_ponto() -> None:
    assert split_text("são 10.") == ["são", "10", "."]


def test_count_tokens_bate_com_split_text() -> None:
    frase = "Uma frase com pontuação, números (42) e -- travessão."
    assert count_tokens(frase) == len(split_text(frase))


def test_frases_diferentes_produzem_contagens_diferentes() -> None:
    curta = "Isto é um teste."
    longa = "Isto é um teste um pouco mais longo, com mais tokens."
    assert count_tokens(curta) < count_tokens(longa)


# --------------------------------------------------------------------------
# 3.2 Vocabulario e Token IDs
# --------------------------------------------------------------------------


def test_vocabulario_ignora_repeticoes() -> None:
    vocab = Vocabulary(["a", "b", "a", "b", "a"], special_tokens=())
    assert len(vocab) == 2


def test_ids_seguem_a_ordem_alfabetica() -> None:
    vocab = Vocabulary(["zebra", "abelha", "macaco"], special_tokens=())
    assert vocab.token_to_id("abelha") == 0
    assert vocab.token_to_id("macaco") == 1
    assert vocab.token_to_id("zebra") == 2


def test_especiais_ficam_no_fim() -> None:
    # Garante que acrescentar especial nao desloca ID de token do corpus.
    tokens = ["a", "b", "c"]
    sem_especiais = Vocabulary(tokens, special_tokens=())
    com_especiais = Vocabulary(tokens)

    for token in tokens:
        assert sem_especiais.token_to_id(token) == com_especiais.token_to_id(token)

    assert com_especiais.token_to_id(END_OF_TEXT_TOKEN) == 3
    assert com_especiais.token_to_id(UNKNOWN_TOKEN) == 4


def test_ida_e_volta_entre_token_e_id() -> None:
    vocab = Vocabulary.from_texts(["o gato mordeu o rato."])
    for token, token_id in vocab:
        assert vocab.id_to_token(token_id) == token
        assert vocab.token_to_id(token) == token_id


def test_sem_unk_palavra_desconhecida_levanta_erro() -> None:
    vocab = Vocabulary(["a", "b"], special_tokens=())
    try:
        vocab.token_to_id("c")
    except KeyError:
        return
    raise AssertionError("esperava KeyError para token fora do vocabulario")


def test_com_unk_palavras_desconhecidas_colapsam_no_mesmo_id() -> None:
    # A perda de informacao do <|unk|>: duas palavras distintas, um ID so.
    vocab = Vocabulary(["a", "b"])
    assert vocab.token_to_id("c") == vocab.token_to_id("d") == vocab.token_to_id(UNKNOWN_TOKEN)


def test_id_fora_da_faixa_levanta_erro() -> None:
    vocab = Vocabulary(["a"], special_tokens=())
    try:
        vocab.id_to_token(999)
    except KeyError:
        return
    raise AssertionError("esperava KeyError para ID inexistente")


# --------------------------------------------------------------------------
# encode / decode
# --------------------------------------------------------------------------


def test_encode_devolve_um_id_por_token() -> None:
    texto = "o gato mordeu o rato."
    tokenizer = SimpleTokenizer(Vocabulary.from_texts([texto]))
    assert len(tokenizer.encode(texto)) == len(split_text(texto))


def test_mesmo_token_em_posicoes_diferentes_tem_o_mesmo_id() -> None:
    # Motivo pelo qual os positional embeddings existem (Frente B): o ID nao
    # carrega nenhuma informacao de posicao.
    texto = "o gato mordeu o rato."
    tokenizer = SimpleTokenizer(Vocabulary.from_texts([texto]))
    ids = tokenizer.encode(texto)
    assert ids[0] == ids[3]


def test_decode_recola_a_pontuacao() -> None:
    texto = "o gato mordeu o rato."
    tokenizer = SimpleTokenizer(Vocabulary.from_texts([texto]))
    assert tokenizer.decode(tokenizer.encode(texto)) == texto


def test_decode_perde_o_espacamento_original() -> None:
    # Perda esperada, nao bug: a quebra descartou os espacos.
    texto = "o    gato\nmordeu."
    tokenizer = SimpleTokenizer(Vocabulary.from_texts([texto]))
    reconstruido = tokenizer.decode(tokenizer.encode(texto))
    assert reconstruido == "o gato mordeu."
    assert reconstruido != texto


def test_endoftext_sobrevive_a_tokenizacao() -> None:
    # Sem tratamento explicito, o regex quebraria o marcador em <, |, ... .
    vocab = Vocabulary.from_texts(["primeiro texto", "segundo texto"])
    tokenizer = SimpleTokenizer(vocab)

    texto = f"primeiro texto {END_OF_TEXT_TOKEN} segundo texto"
    assert END_OF_TEXT_TOKEN in tokenizer.tokenize(texto)
    assert tokenizer.encode(texto).count(vocab.token_to_id(END_OF_TEXT_TOKEN)) == 1


def test_palavra_desconhecida_aparece_como_unk_no_decode() -> None:
    vocab = Vocabulary.from_texts(["o gato dorme."])
    tokenizer = SimpleTokenizer(vocab)
    assert UNKNOWN_TOKEN in tokenizer.decode(tokenizer.encode("o jabuti dorme."))


# --------------------------------------------------------------------------
# BPE (tiktoken)
# --------------------------------------------------------------------------


def test_bpe_tem_o_vocabulario_do_gpt2() -> None:
    assert len(BPETokenizer()) == 50_257


def test_bpe_nao_perde_nada_na_ida_e_volta() -> None:
    # Contraste direto com test_decode_perde_o_espacamento_original: o BPE
    # carrega o espaco dentro do token, entao reconstroi tudo.
    texto = "o    gato\nmordeu o rato, e depois dormiu."
    tokenizer = BPETokenizer()
    assert tokenizer.decode(tokenizer.encode(texto)) == texto


def test_bpe_dispensa_unk_em_palavra_inventada() -> None:
    # Exercicio 2.1 do livro: palavra que nunca existiu, decodificada de volta
    # sem perda, porque foi fatiada em subwords.
    tokenizer = BPETokenizer()
    inventada = "Akwirw ier"
    assert tokenizer.decode(tokenizer.encode(inventada)) == inventada
    assert len(tokenizer.encode(inventada)) > 2  # quebrou em varios pedacos


def test_bpe_carrega_o_espaco_dentro_do_token() -> None:
    tokenizer = BPETokenizer()
    assert tokenizer.tokenize("the cat")[1] == " cat"


def test_endoftext_e_o_ultimo_id_do_gpt2() -> None:
    assert BPETokenizer().encode(END_OF_TEXT_TOKEN) == [50_256]


def main() -> int:
    """Roda todos os testes deste arquivo sem depender de pytest."""
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]

    failures = 0
    for name, function in tests:
        try:
            function()
        except Exception as error:  # noqa: BLE001 - relatorio, nao tratamento
            failures += 1
            print(f"FALHOU  {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok      {name}")

    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
