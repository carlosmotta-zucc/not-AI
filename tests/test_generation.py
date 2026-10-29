"""Testes da inferência: seleção do próximo token e laço autoregressivo.

O que fica protegido aqui:

- Top-k: sobram exatamente k candidatos, os k maiores, e o resto vai a
  probabilidade zero exato.
- Temperatura: 1 não muda nada, menor que 1 concentra, maior que 1 achata.
- Seleção: temperatura 0 é argmax com ou sem top-k; top_k = 1 também é
  determinístico (exercício 5.3); o sorteio é reprodutível com um
  torch.Generator semeado.
- generate(): devolve (B, T + n) com o contexto intacto na frente; com
  decodificação gulosa, cada token gerado é o argmax do logit que a passada
  COMPLETA daria naquela posição; passa do context_length sem erro, recortando
  a janela; respeita o eos; devolve o modelo ao modo em que o recebeu.

Os números do livro (vocabulário de 9 palavras da seção 5.3 e a geração da
seção 4.7) estão em tests/test_book_checkpoints.py.

Uso:
    python tests/test_generation.py
    pytest tests/
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import (  # noqa: E402
    apply_top_k,
    generate,
    generate_text,
    next_token_probabilities,
    select_next_token,
    text_to_token_ids,
    token_ids_to_text,
)
from src.model import GPTConfig, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

SEED = 42

TINY = GPTConfig(
    vocab_size=100,
    context_length=8,
    emb_dim=16,
    n_heads=2,
    n_layers=2,
    drop_rate=0.1,
)


def modelo() -> GPTModel:
    torch.manual_seed(SEED)
    return GPTModel(TINY).eval()


def contexto(batch: int = 1, tokens: int = 4) -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randint(0, TINY.vocab_size, (batch, tokens))


def entropia(probabilidades: torch.Tensor) -> float:
    seguras = probabilidades.clamp_min(torch.finfo(probabilidades.dtype).tiny)
    return float(-(probabilidades * seguras.log()).sum())


def test_top_k_mantem_exatamente_os_k_maiores() -> None:
    torch.manual_seed(SEED)
    logits = torch.randn(3, 50)

    for k in (1, 5, 50, 500):
        filtrados = apply_top_k(logits, k)
        mantidos = min(k, 50)

        assert bool((torch.isfinite(filtrados).sum(dim=-1) == mantidos).all())
        # Os sobreviventes são os k maiores, com o valor original.
        maiores = torch.topk(logits, mantidos, dim=-1).indices
        assert torch.equal(filtrados.gather(-1, maiores), logits.gather(-1, maiores))


def test_top_k_com_empate_mantem_so_k() -> None:
    # A comparação com o menor dos k (a do livro) manteria os quatro 1.0 aqui.
    logits = torch.tensor([[3.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    assert int(torch.isfinite(apply_top_k(logits, 2)).sum()) == 2


def test_top_k_zera_a_probabilidade_dos_outros() -> None:
    torch.manual_seed(SEED)
    logits = torch.randn(2, 30)
    probabilidades = next_token_probabilities(logits, top_k=4)

    assert torch.allclose(probabilidades.sum(dim=-1), torch.ones(2))
    assert bool(((probabilidades > 0).sum(dim=-1) == 4).all())


def test_temperatura_concentra_ou_achata() -> None:
    torch.manual_seed(SEED)
    logits = torch.randn(40)

    original = torch.softmax(logits, dim=-1)
    assert torch.allclose(next_token_probabilities(logits, temperature=1.0), original)

    entropias = [
        entropia(next_token_probabilities(logits, temperature=t))
        for t in (0.1, 0.5, 1.0, 2.0, 10.0)
    ]
    # Mais temperatura, mais espalhada a distribuição.
    assert entropias == sorted(entropias)
    # Nos extremos: concentrada em poucos tokens e quase uniforme (ln 40 =
    # 3,689). Com T = 0,1 sobra ~0,2 nat porque dois logits deste sorteio são
    # próximos e dividem a massa -- one-hot exato só no limite T -> 0.
    assert entropias[0] < 0.5
    assert entropias[-1] > 3.6


def test_temperatura_nao_muda_o_argmax() -> None:
    torch.manual_seed(SEED)
    logits = torch.randn(5, 40)
    for t in (0.1, 1.0, 7.0):
        probabilidades = next_token_probabilities(logits, temperature=t)
        assert torch.equal(probabilidades.argmax(dim=-1), logits.argmax(dim=-1))


def test_temperatura_invalida_levanta_erro() -> None:
    logits = torch.randn(1, 10)
    for chamada in (
        lambda: next_token_probabilities(logits, temperature=0.0),
        lambda: select_next_token(logits, temperature=-1.0),
        lambda: select_next_token(logits, top_k=0),
    ):
        try:
            chamada()
        except ValueError:
            pass
        else:
            raise AssertionError("parâmetro de decodificação inválido foi aceito")


def test_selecao_gulosa_e_o_argmax() -> None:
    torch.manual_seed(SEED)
    logits = torch.randn(4, 30)
    esperado = logits.argmax(dim=-1, keepdim=True)

    assert torch.equal(select_next_token(logits), esperado)
    assert torch.equal(select_next_token(logits, top_k=3), esperado)
    assert select_next_token(logits).shape == (4, 1)

    # Exercício 5.3: top_k = 1 deixa a amostragem determinística.
    gerador = torch.Generator().manual_seed(SEED)
    for _ in range(20):
        sorteado = select_next_token(logits, temperature=1.5, top_k=1, generator=gerador)
        assert torch.equal(sorteado, esperado)


def test_amostragem_e_reprodutivel_com_gerador() -> None:
    torch.manual_seed(SEED)
    logits = torch.randn(2, 30)

    def sorteia(seed: int) -> list[int]:
        gerador = torch.Generator().manual_seed(seed)
        return [
            select_next_token(logits, temperature=1.0, generator=gerador)[0, 0].item()
            for _ in range(50)
        ]

    assert sorteia(1) == sorteia(1)
    assert sorteia(1) != sorteia(2)


def test_generate_preserva_o_contexto_e_anexa_n_tokens() -> None:
    gpt = modelo()
    ids = contexto(batch=3, tokens=4)

    saida = generate(gpt, ids, max_new_tokens=5)

    assert saida.shape == (3, 9)
    assert torch.equal(saida[:, :4], ids)
    assert saida.dtype == torch.long
    assert bool(((saida >= 0) & (saida < TINY.vocab_size)).all())


def test_generate_guloso_coincide_com_a_passada_completa() -> None:
    gpt = modelo()
    ids = contexto(tokens=3)

    saida = generate(gpt, ids, max_new_tokens=5)
    assert saida.shape[1] <= TINY.context_length

    # Uma passada sobre a sequência final: o argmax da posição t precisa ser o
    # token que o laço escolheu para t+1. É a geração autoregressiva e a
    # máscara causal dizendo a mesma coisa de dois jeitos.
    with torch.no_grad():
        previsoes = gpt(saida[:, :-1]).argmax(dim=-1)
    assert torch.equal(previsoes[:, 2:], saida[:, 3:])


def test_generate_passa_do_contexto_recortando_a_janela() -> None:
    gpt = modelo()
    ids = contexto(tokens=6)

    # 6 + 10 = 16 tokens, o dobro do context_length: sem o recorte, a tabela
    # de posições levantaria ValueError no terceiro passo.
    saida = generate(gpt, ids, max_new_tokens=10)
    assert saida.shape == (1, 16)

    # O último token só pode ter vindo dos 8 anteriores.
    with torch.no_grad():
        janela = saida[:, -1 - TINY.context_length : -1]
        assert gpt(janela)[0, -1].argmax().item() == saida[0, -1].item()


def test_generate_e_deterministico_e_restaura_o_modo() -> None:
    gpt = modelo().train()  # dropout ligado de propósito
    ids = contexto(batch=2)

    primeira = generate(gpt, ids, max_new_tokens=6)
    segunda = generate(gpt, ids, max_new_tokens=6)

    # generate() roda em eval(): com dropout ligado, a gulosa variaria.
    assert torch.equal(primeira, segunda)
    assert gpt.training


def test_generate_com_amostragem_reprodutivel() -> None:
    gpt = modelo()
    ids = contexto()

    def gera(seed: int) -> torch.Tensor:
        gerador = torch.Generator().manual_seed(seed)
        return generate(gpt, ids, 8, temperature=1.5, top_k=20, generator=gerador)

    assert torch.equal(gera(7), gera(7))
    assert not torch.equal(gera(7), gera(8))


def test_generate_para_no_eos() -> None:
    gpt = modelo()
    ids = contexto()

    # O primeiro token guloso vira o "eos": a geração tem que parar logo
    # depois dele, com o eos anexado uma vez, antes de max_new_tokens.
    eos = int(generate(gpt, ids, max_new_tokens=1)[0, -1])
    saida = generate(gpt, ids, max_new_tokens=10, eos_id=eos)

    assert saida.shape == (1, ids.shape[1] + 1)
    assert saida[0, -1].item() == eos


def test_eos_completa_quem_ja_terminou() -> None:
    gpt = modelo()
    ids = contexto(batch=2)

    # O eos é o primeiro token da sequência 0: ela termina no primeiro passo
    # e, enquanto a outra continuar, é completada com eos.
    eos = int(generate(gpt, ids[:1], max_new_tokens=1)[0, -1])
    saida = generate(gpt, ids, max_new_tokens=10, eos_id=eos)

    assert bool((saida[0, ids.shape[1] :] == eos).all())


def test_generate_rejeita_entrada_sem_lote() -> None:
    try:
        generate(modelo(), torch.tensor([1, 2, 3]), max_new_tokens=2)
    except ValueError as erro:
        assert "unsqueeze" in str(erro)
    else:
        raise AssertionError("generate aceitou entrada 1D")


def test_conversao_texto_ids_ida_e_volta() -> None:
    tokenizer = BPETokenizer()
    texto = "Every effort moves you"

    ids = text_to_token_ids(texto, tokenizer)
    assert ids.shape == (1, 4)
    assert token_ids_to_text(ids, tokenizer) == texto
    assert token_ids_to_text(ids[0], tokenizer) == texto

    try:
        token_ids_to_text(ids.repeat(2, 1), tokenizer)
    except ValueError:
        pass
    else:
        raise AssertionError("decodificou um lote de 2 como um texto só")


def test_generate_text_comeca_pelo_prompt() -> None:
    tokenizer = BPETokenizer()
    cfg = GPTConfig(vocab_size=len(tokenizer), context_length=16, emb_dim=16, n_heads=2, n_layers=1)
    torch.manual_seed(SEED)
    gpt = GPTModel(cfg)

    texto = generate_text(gpt, tokenizer, "Every effort moves you", max_new_tokens=3)
    assert texto.startswith("Every effort moves you")
    assert len(tokenizer.encode(texto)) >= 4


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
