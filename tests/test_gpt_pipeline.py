"""Teste de integração: The Verdict -> BPE -> lotes -> GPTModel -> logits -> texto.

O tests/test_attention_pipeline.py da Sprint 3 parava na MultiHeadAttention.
Este percorre o pipeline inteiro do projeto até aqui, na configuração base
(BASE_CONFIG), e fixa o que a Sprint 5 vai precisar:

- Os shapes fecham: (B, T) de Token IDs vira (B, T, vocab_size) de logits.
- Os alvos do DataLoader casam com os logits: a cross-entropy entre os dois já
  pode ser calculada, e no modelo sem treino ela fica perto de ln(vocab_size)
  -- o valor de uma distribuição uniforme. É o ponto de partida do treino.
- O backward da perda de verdade alcança todos os parâmetros do modelo.
- A geração parte de um trecho do corpus e devolve texto que começa por ele.

Depende do corpus: na primeira execução load_verdict() baixa The Verdict para
data/, e o tiktoken baixa a tabela do BPE.

Uso:
    python tests/test_gpt_pipeline.py
    pytest tests/
"""

import math
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import load_verdict  # noqa: E402
from src.dataset import create_dataloader  # noqa: E402
from src.generation import generate, text_to_token_ids, token_ids_to_text  # noqa: E402
from src.model import BASE_CONFIG, GPTModel  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

SEED = 42
BATCH_SIZE = 8


@lru_cache(maxsize=1)
def primeiro_lote() -> tuple[torch.Tensor, torch.Tensor]:
    """Primeiro lote de The Verdict, na ordem do texto (shuffle=False)."""
    loader = create_dataloader(
        load_verdict(),
        BPETokenizer(),
        context_length=BASE_CONFIG.context_length,
        stride=BASE_CONFIG.context_length,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )
    return next(iter(loader))


def modelo() -> GPTModel:
    torch.manual_seed(SEED)
    return GPTModel(BASE_CONFIG)


def test_shapes_do_token_id_ao_logit() -> None:
    entrada, alvo = primeiro_lote()
    gpt = modelo().eval()

    assert entrada.shape == alvo.shape == (BATCH_SIZE, BASE_CONFIG.context_length)

    with torch.no_grad():
        logits = gpt(entrada)
    assert logits.shape == (BATCH_SIZE, BASE_CONFIG.context_length, BASE_CONFIG.vocab_size)


def test_perda_inicial_e_a_de_um_chute_uniforme() -> None:
    entrada, alvo = primeiro_lote()
    gpt = modelo().eval()

    with torch.no_grad():
        logits = gpt(entrada)

    # flatten(0, 1): cada uma das B·T posições é um problema de classificação
    # entre vocab_size classes, com o alvo deslocado de uma posição.
    perda = F.cross_entropy(logits.flatten(0, 1), alvo.flatten()).item()
    uniforme = math.log(BASE_CONFIG.vocab_size)

    # Sem treino o modelo não sabe nada: a perda é a de quem distribui a
    # probabilidade igualmente, ~10,8 nats. Muito acima disso indicaria logits
    # mal escalados (sem a final_norm, por exemplo); muito abaixo, vazamento
    # do alvo pela máscara.
    assert abs(perda - uniforme) < 1.0, f"perda {perda:.3f} longe de ln(V) = {uniforme:.3f}"


def test_backward_da_cross_entropy_alcanca_todo_o_modelo() -> None:
    entrada, alvo = primeiro_lote()
    gpt = modelo().train()

    logits = gpt(entrada)
    F.cross_entropy(logits.flatten(0, 1), alvo.flatten()).backward()

    sem_gradiente = [
        nome
        for nome, parametro in gpt.named_parameters()
        if parametro.grad is None or parametro.grad.abs().sum() == 0
    ]
    assert not sem_gradiente, f"sem gradiente: {sem_gradiente}"


def test_geracao_continua_um_trecho_do_corpus() -> None:
    tokenizer = BPETokenizer()
    gpt = modelo()

    prompt = tokenizer.decode(tokenizer.encode(load_verdict())[:10])
    ids = text_to_token_ids(prompt, tokenizer)

    saida = generate(gpt, ids, max_new_tokens=12)
    assert saida.shape == (1, ids.shape[1] + 12)
    assert torch.equal(saida[:, : ids.shape[1]], ids)
    assert token_ids_to_text(saida, tokenizer).startswith(prompt)


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
