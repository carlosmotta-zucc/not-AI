"""Testes do salvamento e carregamento do treino (Sprint 5, seção 5.4).

O que fica protegido aqui:

- Ida e volta: o arquivo devolve os mesmos pesos, as duas configurações e o
  histórico; o modelo carregado produz os mesmos logits e o mesmo texto, na
  decodificação gulosa e na amostragem com gerador semeado.
- Otimizador: os momentos do AdamW voltam iguais, e um arquivo salvo sem eles
  devolve optimizer=None e é menor.
- Retomada: treinar 2 épocas, salvar, carregar num modelo novo e treinar mais
  2 dá exatamente o mesmo que 4 de uma vez -- com o estado do otimizador e o
  do gerador. Sem o estado do otimizador, o treino retomado é outro.
- Gerador global: carregar um arquivo não muda os sorteios de quem chamou.
- Recusas: arquivo sem as chaves do formato, versão desconhecida, restore_rng
  sem estado salvo, e arquivo com objeto que não é tensor nem tipo simples.

O modelo é o minúsculo de tests/test_training.py, que aprende o alfabeto.

Uso:
    python tests/test_checkpoint.py
    pytest tests/
"""

import pickle
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import generate, text_to_token_ids  # noqa: E402
from src.model import GPTModel  # noqa: E402
from src.training import (  # noqa: E402
    CHECKPOINT_FORMAT,
    Checkpoint,
    TrainingRun,
    create_optimizer,
    create_train_val_loaders,
    load_checkpoint,
    save_checkpoint,
    train_model,
)
from tests.test_training import SEED, TEXT, TINY, TINY_TRAINING, CharTokenizer, treinar  # noqa: E402

SHORT_TRAINING = replace(TINY_TRAINING, num_epochs=2)


class NotATensor:
    """Um objeto qualquer, para o arquivo que weights_only tem que recusar."""


def salvar(pasta: str, run: TrainingRun, completo: bool = True) -> Path:
    """Salva o treino de `run` em pasta/treino.pth, com ou sem otimizador e gerador."""
    return save_checkpoint(
        Path(pasta) / "treino.pth",
        run.model,
        SHORT_TRAINING,
        run.history,
        optimizer=run.optimizer if completo else None,
        rng_state=torch.get_rng_state() if completo else None,
    )


def continuar(checkpoint: Checkpoint, optimizer: torch.optim.Optimizer, epocas: int = 2) -> None:
    """Mais `epocas` de treino sobre o que veio do arquivo, com loaders refeitos da configuração."""
    cfg = checkpoint.train_cfg
    train_loader, val_loader = create_train_val_loaders(
        TEXT,
        CharTokenizer(),
        context_length=checkpoint.model_cfg.context_length,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        stride=cfg.stride,
    )
    train_model(
        checkpoint.model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=epocas,
        eval_freq=cfg.eval_freq,
        eval_iter=cfg.eval_iter,
        tokenizer=CharTokenizer(),
        start_context=cfg.start_context,
        sample_tokens=cfg.sample_tokens,
        history=checkpoint.history,
        log=None,
    )


def test_ida_e_volta_preserva_pesos_configuracoes_e_historico() -> None:
    run = treinar(SHORT_TRAINING)

    with tempfile.TemporaryDirectory() as pasta:
        carregado = load_checkpoint(salvar(pasta, run))

    salvos = run.model.state_dict()
    lidos = carregado.model.state_dict()
    assert salvos.keys() == lidos.keys()
    for nome, tensor in salvos.items():
        assert torch.equal(tensor, lidos[nome]), nome

    assert carregado.model_cfg == TINY
    assert carregado.train_cfg == SHORT_TRAINING
    assert carregado.history == run.history
    # Outro objeto, com os mesmos números: mexer num não mexe no outro.
    assert carregado.history is not run.history
    assert carregado.model is not run.model


def test_modelo_carregado_da_os_mesmos_logits_e_o_mesmo_texto() -> None:
    run = treinar(SHORT_TRAINING)
    with tempfile.TemporaryDirectory() as pasta:
        carregado = load_checkpoint(salvar(pasta, run)).model

    ids = text_to_token_ids("abc", CharTokenizer())
    run.model.eval()
    with torch.no_grad():
        assert torch.equal(run.model(ids), carregado(ids))

    assert torch.equal(generate(run.model, ids, 10), generate(carregado, ids, 10))

    def sorteado(model: GPTModel) -> torch.Tensor:
        gerador = torch.Generator().manual_seed(SEED)
        return generate(model, ids, 10, temperature=1.5, top_k=5, generator=gerador)

    assert torch.equal(sorteado(run.model), sorteado(carregado))


def test_modelo_carregado_vem_em_eval_e_salvar_nao_muda_o_modo() -> None:
    run = treinar(SHORT_TRAINING)
    assert run.model.training

    with tempfile.TemporaryDirectory() as pasta:
        caminho = salvar(pasta, run)
        assert run.model.training
        assert not load_checkpoint(caminho).model.training


def test_otimizador_volta_com_os_momentos_do_adamw() -> None:
    run = treinar(SHORT_TRAINING)
    with tempfile.TemporaryDirectory() as pasta:
        carregado = load_checkpoint(salvar(pasta, run))

    assert isinstance(carregado.optimizer, torch.optim.AdamW)
    grupo = carregado.optimizer.param_groups[0]
    assert grupo["lr"] == SHORT_TRAINING.learning_rate
    assert grupo["weight_decay"] == SHORT_TRAINING.weight_decay

    # O estado é indexado pelo parâmetro: os do modelo novo, na mesma ordem.
    passos = run.history.global_step + 1
    for original, novo in zip(run.model.parameters(), carregado.model.parameters()):
        antes, depois = run.optimizer.state[original], carregado.optimizer.state[novo]
        assert int(depois["step"]) == passos
        assert torch.equal(antes["exp_avg"], depois["exp_avg"])
        assert torch.equal(antes["exp_avg_sq"], depois["exp_avg_sq"])


def test_arquivo_so_de_pesos_devolve_sem_otimizador_e_e_menor() -> None:
    run = treinar(SHORT_TRAINING)
    with tempfile.TemporaryDirectory() as pasta:
        completo = salvar(pasta, run).stat().st_size
        caminho = salvar(pasta, run, completo=False)
        carregado = load_checkpoint(caminho)

        # m e v têm, cada um, o tamanho dos pesos.
        assert caminho.stat().st_size < completo / 2

    assert carregado.optimizer is None
    assert carregado.rng_state is None
    for a, b in zip(run.model.parameters(), carregado.model.parameters()):
        assert torch.equal(a, b)


def test_retomada_pelo_arquivo_e_identica_ao_treino_continuo() -> None:
    continuo = treinar(replace(TINY_TRAINING, num_epochs=4))

    parte = treinar(SHORT_TRAINING)
    with tempfile.TemporaryDirectory() as pasta:
        caminho = salvar(pasta, parte)
        # Outro ponto do gerador, como num processo novo: o que vale é o do arquivo.
        torch.manual_seed(SEED + 1)
        retomado = load_checkpoint(caminho, restore_rng=True)

    continuar(retomado, retomado.optimizer)

    for campo in ("steps", "epochs", "tokens_seen", "train_losses", "val_losses", "batch_losses", "samples"):
        assert getattr(retomado.history, campo) == getattr(continuo.history, campo), campo
    for a, b in zip(retomado.model.parameters(), continuo.model.parameters()):
        assert torch.equal(a, b)


def test_sem_o_estado_do_otimizador_a_retomada_e_outro_treino() -> None:
    continuo = treinar(replace(TINY_TRAINING, num_epochs=4))

    parte = treinar(SHORT_TRAINING)
    with tempfile.TemporaryDirectory() as pasta:
        retomado = load_checkpoint(salvar(pasta, parte), restore_rng=True)

    # AdamW novo: m e v recomeçam do zero. Lotes e dropout são os mesmos.
    continuar(retomado, create_optimizer(retomado.model, retomado.train_cfg))

    pausa = len(parte.history.batch_losses)
    assert retomado.history.batch_losses[:pausa] == continuo.history.batch_losses[:pausa]
    # O primeiro passo depois da pausa ainda usa os mesmos pesos; do segundo em diante, não.
    assert retomado.history.batch_losses[pausa] == continuo.history.batch_losses[pausa]
    assert retomado.history.batch_losses[pausa + 1 :] != continuo.history.batch_losses[pausa + 1 :]
    assert any(
        not torch.equal(a, b)
        for a, b in zip(retomado.model.parameters(), continuo.model.parameters())
    )


def test_carregar_nao_consome_o_gerador_global() -> None:
    run = treinar(SHORT_TRAINING)
    with tempfile.TemporaryDirectory() as pasta:
        caminho = salvar(pasta, run)

        torch.manual_seed(SEED)
        esperado = torch.rand(4)

        torch.manual_seed(SEED)
        load_checkpoint(caminho)
        assert torch.equal(torch.rand(4), esperado)

        # Com restore_rng, o gerador fica no ponto em que o treino parou.
        torch.manual_seed(SEED)
        carregado = load_checkpoint(caminho, restore_rng=True)
        assert torch.equal(torch.get_rng_state(), carregado.rng_state)


def test_pesos_compartilhados_continuam_compartilhados() -> None:
    torch.manual_seed(SEED)
    model = GPTModel(replace(TINY, tie_weights=True))
    run = TrainingRun(model, None, treinar(replace(SHORT_TRAINING, num_epochs=0)).history, None, None)

    with tempfile.TemporaryDirectory() as pasta:
        carregado = load_checkpoint(salvar(pasta, run, completo=False)).model

    assert carregado.out_head.weight is carregado.embedding.token_embedding.weight
    assert torch.equal(carregado.out_head.weight, model.out_head.weight)
    assert carregado.parameter_count() == model.parameter_count()


def test_arquivo_fora_do_formato_levanta_erro() -> None:
    run = treinar(replace(SHORT_TRAINING, num_epochs=0))

    with tempfile.TemporaryDirectory() as pasta:
        valido = torch.load(salvar(pasta, run), weights_only=True)
        sem_gerador = salvar(pasta, run, completo=False)

        casos = {
            "só o state_dict, como no livro": run.model.state_dict(),
            "sem a configuração do modelo": {k: v for k, v in valido.items() if k != "model_config"},
            "versão do futuro": {**valido, "format_version": CHECKPOINT_FORMAT + 1},
            "uma lista": [1, 2, 3],
        }
        for nome, conteudo in casos.items():
            caminho = Path(pasta) / "invalido.pth"
            torch.save(conteudo, caminho)
            try:
                load_checkpoint(caminho)
            except ValueError:
                continue
            raise AssertionError(f"{nome}: deveria falhar")

        try:
            load_checkpoint(sem_gerador, restore_rng=True)
        except ValueError:
            return
        raise AssertionError("restore_rng sem estado salvo deveria falhar")


def test_arquivo_com_objeto_arbitrario_e_recusado() -> None:
    run = treinar(replace(SHORT_TRAINING, num_epochs=0))

    with tempfile.TemporaryDirectory() as pasta:
        conteudo = torch.load(salvar(pasta, run), weights_only=True)
        caminho = Path(pasta) / "com-objeto.pth"
        torch.save({**conteudo, "extra": NotATensor()}, caminho)

        try:
            load_checkpoint(caminho)
        except pickle.UnpicklingError:
            return
    raise AssertionError("weights_only deveria recusar um objeto que não é tensor")


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
