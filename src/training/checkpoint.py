"""Salvar e carregar um treino (seção 5.4): pesos, otimizador e o que identifica a execução.

    save_checkpoint:  modelo + otimizador + configurações + histórico --torch.save--> arquivo .pth
    load_checkpoint:  arquivo .pth --torch.load--> GPTModel + AdamW + TrainingHistory, prontos

O livro salva dois dicionários de tensores, `model.state_dict()` e
`optimizer.state_dict()`, e remonta o modelo a partir do GPT_CONFIG_124M que
está no notebook. Aqui o arquivo carrega também o que o notebook guardava na
memória de quem o rodou:

- `model_config` e `training_config` (dataclasses.asdict): sem a configuração,
  um state_dict é uma lista de tensores que só serve para quem souber de que
  arquitetura eles saíram. Com ela, o arquivo se remonta sozinho.
- `history` (TrainingHistory.to_dict): as perdas, as amostras e a contagem de
  passos, tokens e épocas, para a retomada seguir a numeração.
- `optimizer_state_dict`, opcional: os momentos m e v do AdamW, dois números
  por peso. Sem eles, o treino retomado recomeça as médias móveis do zero e os
  primeiros passos saem diferentes dos de um treino sem pausa. Para só gerar
  texto, não fazem falta, e o arquivo fica com um terço do tamanho.
- `rng_state`, opcional: o estado do gerador global do PyTorch. É dele que
  saem a ordem dos lotes e as máscaras de dropout; sem ele, a retomada é um
  treino válido, mas não o MESMO treino.

Decisões:

- state_dict, e não `torch.save(model)`. O modelo inteiro em pickle guarda o
  caminho das classes, e deixa de abrir se um módulo for renomeado. O
  state_dict só tem nomes de parâmetros e tensores.
- `weights_only=True` na leitura. O formato .pth é pickle, e abrir um pickle
  executa o que estiver nele; com weights_only, o PyTorch só aceita tensores e
  tipos simples, e um arquivo com qualquer outro objeto é recusado. Por isso o
  arquivo guarda dicionários, e não as dataclasses.
- `map_location`: os tensores vão para o device pedido na leitura, qualquer
  que seja o device em que foram salvos.
- Carregar não consome o gerador global. Criar o GPTModel sorteia os pesos
  iniciais (que o state_dict sobrescreve em seguida); o estado do gerador é
  guardado antes e devolvido depois, e quem chama continua sorteando como se
  não tivesse carregado nada. Com `restore_rng=True`, o que entra no lugar é o
  estado salvo no arquivo -- por último, depois de o modelo existir.
"""

from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import torch

from src.model import GPTConfig, GPTModel

from .config import TrainingConfig
from .trainer import TrainingHistory, create_optimizer

# Versão do dicionário salvo. Muda quando uma chave obrigatória muda de nome
# ou de significado; a leitura recusa versões que não conhece.
CHECKPOINT_FORMAT = 1

REQUIRED_KEYS = (
    "format_version",
    "model_state_dict",
    "model_config",
    "training_config",
    "history",
)


class Checkpoint(NamedTuple):
    """O que load_checkpoint devolve: o treino remontado.

    optimizer e rng_state são None quando o arquivo foi salvo sem eles.
    """

    model: GPTModel
    optimizer: torch.optim.Optimizer | None
    history: TrainingHistory
    model_cfg: GPTConfig
    train_cfg: TrainingConfig
    rng_state: torch.Tensor | None


def save_checkpoint(
    path: str | Path,
    model: GPTModel,
    train_cfg: TrainingConfig,
    history: TrainingHistory,
    optimizer: torch.optim.Optimizer | None = None,
    rng_state: torch.Tensor | None = None,
) -> Path:
    """Grava o treino num arquivo -- o torch.save da seção 5.4, com a configuração junto.

    Args:
        path: destino; a convenção do PyTorch é a extensão .pth. As pastas do
            caminho são criadas.
        model: o GPTModel; a arquitetura sai de model.cfg.
        train_cfg: os hiperparâmetros com que ele foi treinado.
        history: o histórico do treino até aqui.
        optimizer: se dado, o estado dele vai junto (para retomar o treino).
        rng_state: torch.get_rng_state() no momento em que o treino parou, para
            a retomada ser idêntica a um treino sem pausa.

    Returns:
        O caminho gravado.

    Os tensores são gravados no device em que estão. Só o state_dict é lido: o
    modo do modelo (train ou eval) não muda e não é salvo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": CHECKPOINT_FORMAT,
        # str(): torch.__version__ é um TorchVersion, e weights_only recusa
        # qualquer classe que não seja tensor ou tipo simples -- inclusive essa.
        "torch_version": str(torch.__version__),
        "model_config": asdict(model.cfg),
        "training_config": asdict(train_cfg),
        "history": history.to_dict(),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if rng_state is not None:
        payload["rng_state"] = rng_state

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> Checkpoint:
    """Remonta o treino salvo por save_checkpoint.

    Args:
        path: o arquivo .pth.
        device: para onde vão o modelo e o estado do otimizador.
        restore_rng: aplica ao gerador global o estado salvo no arquivo. É o
            que falta para a retomada ser idêntica ao treino sem pausa.

    Returns:
        Checkpoint com o modelo em eval(), pronto para gerar texto; train_model
        o devolve ao modo de treino sozinho.

    Raises:
        ValueError: o arquivo não é um checkpoint deste formato (chave
            obrigatória ausente, versão desconhecida), ou restore_rng foi
            pedido e o arquivo não tem o estado do gerador.
        pickle.UnpicklingError: o arquivo contém algo além de tensores e tipos
            simples (weights_only=True).
    """
    payload = torch.load(path, map_location=device, weights_only=True)

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: esperado um dicionário, veio {type(payload).__name__}")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path}: não é um checkpoint do projeto; faltam as chaves {missing}")
    if payload["format_version"] != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: formato {payload['format_version']}, e este código lê o {CHECKPOINT_FORMAT}"
        )

    rng_state = payload.get("rng_state")
    if rng_state is not None:
        # set_rng_state só aceita o tensor na CPU; map_location pode tê-lo movido.
        rng_state = rng_state.cpu()
    if restore_rng and rng_state is None:
        raise ValueError(f"{path}: salvo sem rng_state; não há o que restaurar")

    model_cfg = GPTConfig(**payload["model_config"])
    train_cfg = TrainingConfig(**payload["training_config"])

    # O construtor sorteia pesos que a linha seguinte sobrescreve. O estado do
    # gerador é devolvido no fim, para a carga não mudar os sorteios de quem chama.
    caller_rng_state = torch.get_rng_state()
    model = GPTModel(model_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    optimizer = None
    if payload.get("optimizer_state_dict") is not None:
        # O otimizador nasce sobre os parâmetros já no device; load_state_dict
        # leva m e v para o device de cada parâmetro.
        optimizer = create_optimizer(model, train_cfg)
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    history = TrainingHistory.from_dict(payload["history"])

    torch.set_rng_state(rng_state if restore_rng else caller_rng_state)
    return Checkpoint(model, optimizer, history, model_cfg, train_cfg, rng_state)
