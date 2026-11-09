"""O laço de treino (seção 5.2): épocas, lotes, perda, gradiente e passo do otimizador.

    para cada época:
        model.train()
        para cada lote (entrada, alvo) do train_loader:
            1. optimizer.zero_grad()     apaga o gradiente do passo anterior
            2. perda = batch_loss(...)   forward + cross-entropy, com dropout
            3. perda.backward()          gradiente da perda em cada peso
            4. optimizer.step()          pesos novos, a partir do gradiente
            5. a cada eval_freq passos:  perdas de treino e de validação
        no fim da época: um texto gerado a partir do mesmo contexto

Os passos 1 a 4 são o treino; o 5 e a amostra só observam. A avaliação roda em
eval() e sem grafo (loader_loss), e a amostra também (generate): nenhuma das
duas toca nos pesos nem no gradiente.

Por que zero_grad: backward() SOMA o gradiente novo ao que já estiver em
.grad. Sem apagar, cada passo usaria a soma de todos os gradientes desde o
começo do treino.

O otimizador é o AdamW do livro (create_optimizer). Na descida de gradiente
simples, cada peso anda lr · gradiente. No Adam, anda lr · m / (√v + ε), com m
e v médias móveis do gradiente e do quadrado dele: o passo de cada peso é
normalizado pela escala do próprio gradiente, e pesos de gradiente pequeno
andam tanto quanto os de gradiente grande. O "W" é o weight decay desacoplado:
a cada passo, todo peso encolhe pelo fator (1 - lr · wd), fora da conta do
gradiente -- é o que separa o AdamW do Adam com penalidade L2 na perda.

Reprodutibilidade. O laço consome o gerador global na mesma ordem do
train_model_simple do livro: depois dos sorteios dos pesos (feitos antes, na
criação do modelo), cada época sorteia o embaralhamento do DataLoader, cada
passo sorteia as máscaras de dropout, e cada avaliação cria iteradores novos,
que também sorteiam. Trocar a ordem -- avaliar antes do passo, por exemplo --
não muda o que o treino faz, mas muda os números, e o projeto deixaria de
reproduzir o capítulo.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, NamedTuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.generation import SupportsTokenizer, generate, text_to_token_ids, token_ids_to_text
from src.model import GPTConfig, GPTModel

from .config import TrainingConfig
from .data import create_train_val_loaders
from .loss import batch_loss, evaluate

Logger = Callable[[str], None]


@dataclass
class TrainingHistory:
    """Tudo o que um treino registra, em listas paralelas.

    Uma entrada por avaliação (a cada eval_freq passos), na mesma posição das
    cinco listas:
        steps: passo global da avaliação; o primeiro passo é o 0.
        epochs: época em que ela aconteceu, contando de 1.
        tokens_seen: tokens de entrada processados até ali.
        train_losses, val_losses: as perdas avaliadas.

    Uma entrada por passo:
        batch_losses: perda do lote de cada passo, com dropout -- a que o
            backward usou. Oscila de lote para lote; a avaliada é a medida
            estável.

    Uma entrada por época:
        samples: texto gerado no fim da época (contexto + continuação), quando
            a amostra está ligada.
        epoch_seconds: duração da época inteira (passos, avaliações, amostra).
        monitor_seconds: a parte da época gasta em avaliação e amostra.

    Estado para continuar o treino numa chamada seguinte:
        global_step: último passo concluído (-1 antes do primeiro).
        total_tokens: tokens de entrada processados no total.
    """

    steps: list[int] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    tokens_seen: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    batch_losses: list[float] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    monitor_seconds: list[float] = field(default_factory=list)
    global_step: int = -1
    total_tokens: int = 0

    @property
    def epochs_completed(self) -> int:
        return len(self.epoch_seconds)

    def best_evaluation(self) -> int:
        """Índice, nas listas por avaliação, da menor perda de validação.

        Raises:
            ValueError: se nenhuma avaliação foi registrada.
        """
        if not self.val_losses:
            raise ValueError("nenhuma avaliação registrada")
        return min(range(len(self.val_losses)), key=self.val_losses.__getitem__)

    def to_dict(self) -> dict:
        """Dicionário só com tipos de JSON: o que se salva junto com os pesos."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingHistory":
        """Reconstrói o histórico salvo por to_dict, para continuar o treino."""
        return cls(**data)


def create_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.AdamW:
    """O AdamW do livro (p. 149), com a taxa e o weight decay da configuração.

    Args:
        model: o modelo cujos parâmetros serão otimizados.
        cfg: de onde saem learning_rate e weight_decay.

    Returns:
        AdamW sobre model.parameters(), com betas e eps no padrão do PyTorch
        (0,9; 0,999; 1e-8), que é o que o livro usa.

    O weight decay vale para todos os parâmetros, inclusive bias e os ganhos
    das LayerNorms, como no livro. Treinos maiores costumam isentar esses dois
    grupos; aqui fica igual ao livro para os números baterem.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def generate_sample(
    model: nn.Module,
    tokenizer: SupportsTokenizer,
    start_context: str,
    max_new_tokens: int,
) -> str:
    """Continuação gulosa de start_context -- o generate_and_print_sample do livro, sem o print.

    Args:
        model: o GPTModel, em qualquer device.
        tokenizer: o BPETokenizer.
        start_context: texto de partida.
        max_new_tokens: tokens a gerar.

    Returns:
        O contexto seguido da continuação, com as quebras de linha originais.

    Gulosa (temperatura 0), como no livro: sem sorteio, a amostra não consome
    o gerador global e não muda o resto do treino. generate() cuida do eval()
    e do no_grad(), e devolve o modelo ao modo de treino.
    """
    device = next(model.parameters()).device
    ids = text_to_token_ids(start_context, tokenizer).to(device)
    generated = generate(model, ids, max_new_tokens)
    return token_ids_to_text(generated.cpu(), tokenizer)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int | None,
    tokenizer: SupportsTokenizer | None = None,
    start_context: str | None = None,
    sample_tokens: int = 50,
    history: TrainingHistory | None = None,
    log: Logger | None = print,
) -> TrainingHistory:
    """Treina o modelo por num_epochs épocas -- o train_model_simple do livro (listing 5.3).

    Args:
        model: o GPTModel, já no device em que vai treinar.
        train_loader: lotes de treino (B, T).
        val_loader: lotes de validação (B, T).
        optimizer: criado sobre model.parameters(); create_optimizer serve.
        num_epochs: épocas a treinar NESTA chamada.
        eval_freq: avalia quando passo_global % eval_freq == 0. O primeiro
            passo é o 0, então a primeira avaliação vem logo depois da primeira
            atualização dos pesos, como no livro.
        eval_iter: lotes por avaliação; None = os conjuntos inteiros.
        tokenizer: com start_context, liga a amostra do fim de cada época.
        start_context: texto de partida da amostra.
        sample_tokens: tamanho da amostra, em tokens; 0 desliga.
        history: histórico de um treino anterior, para continuar dele. Passos,
            tokens e épocas seguem a contagem; o otimizador tem que ser o mesmo
            objeto, ou um com o estado restaurado, para o AdamW não recomeçar
            as médias móveis do zero.
        log: recebe uma linha por avaliação e uma por amostra; None silencia.

    Returns:
        O histórico -- o mesmo objeto de `history`, quando ele vier.

    Raises:
        ValueError: num_epochs negativo, eval_freq < 1 ou train_loader sem
            lotes.

    O modelo termina em modo de treino, como no livro. Para avaliar ou gerar
    depois, loader_loss e generate já colocam em eval() sozinhos.
    """
    if num_epochs < 0:
        raise ValueError(f"num_epochs={num_epochs} não pode ser negativo")
    if eval_freq < 1:
        raise ValueError(f"eval_freq={eval_freq} precisa ser positivo")
    if len(train_loader) == 0:
        raise ValueError("train_loader sem lotes: não há o que treinar")

    if history is None:
        history = TrainingHistory()

    def emit(line: str) -> None:
        if log is not None:
            log(line)

    sample_enabled = tokenizer is not None and start_context is not None and sample_tokens > 0
    first_epoch = history.epochs_completed + 1

    for epoch in range(first_epoch, first_epoch + num_epochs):
        epoch_start = time.perf_counter()
        monitor = 0.0
        model.train()

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(model, inputs, targets)
            loss.backward()
            optimizer.step()

            history.batch_losses.append(loss.item())
            history.total_tokens += inputs.numel()
            history.global_step += 1

            if history.global_step % eval_freq == 0:
                monitor_start = time.perf_counter()
                train_loss, val_loss = evaluate(model, train_loader, val_loader, eval_iter)
                monitor += time.perf_counter() - monitor_start

                history.steps.append(history.global_step)
                history.epochs.append(epoch)
                history.tokens_seen.append(history.total_tokens)
                history.train_losses.append(train_loss)
                history.val_losses.append(val_loss)
                emit(
                    f"Época {epoch} (passo {history.global_step:06d}): "
                    f"perda de treino {train_loss:.3f}, perda de validação {val_loss:.3f}"
                )

        if sample_enabled:
            monitor_start = time.perf_counter()
            sample = generate_sample(model, tokenizer, start_context, sample_tokens)
            monitor += time.perf_counter() - monitor_start

            history.samples.append(sample)
            # Uma linha só no terminal, como o livro imprime.
            emit(sample.replace("\n", " "))

        history.epoch_seconds.append(time.perf_counter() - epoch_start)
        history.monitor_seconds.append(monitor)

    return history


class TrainingRun(NamedTuple):
    """O que train_gpt devolve: o modelo treinado e tudo o que é preciso para
    continuar o treino ou descrever a execução."""

    model: GPTModel
    optimizer: torch.optim.Optimizer
    history: TrainingHistory
    train_loader: DataLoader
    val_loader: DataLoader


def train_gpt(
    model_cfg: GPTConfig,
    train_cfg: TrainingConfig,
    text: str,
    tokenizer: SupportsTokenizer,
    device: str | torch.device = "cpu",
    log: Logger | None = print,
) -> TrainingRun:
    """Do texto ao modelo treinado, na ordem do livro: dados, seed, modelo, otimizador, laço.

    Args:
        model_cfg: arquitetura (BASE_CONFIG, BOOK_MODEL_CONFIG, ...).
        train_cfg: hiperparâmetros do treino.
        text: corpus inteiro; a divisão treino/validação é feita aqui.
        tokenizer: o BPETokenizer.
        device: onde o modelo treina. O padrão é a CPU, a máquina de referência
            do projeto; em GPU o treino funciona, mas os números deixam de ser
            os da tabela publicada (outras rotinas numéricas, outro gerador
            para o dropout).
        log: repassado a train_model.

    Returns:
        TrainingRun com modelo, otimizador, histórico e os dois loaders.

    A seed vem imediatamente antes do modelo, como no livro (p. 149): ela
    decide os pesos iniciais e, como o laço segue consumindo o mesmo gerador,
    também a ordem dos lotes e as máscaras de dropout. Os DataLoaders não
    sorteiam nada ao serem criados, só ao serem iterados, então podem vir antes.
    """
    train_loader, val_loader = create_train_val_loaders(
        text,
        tokenizer,
        context_length=model_cfg.context_length,
        batch_size=train_cfg.batch_size,
        train_ratio=train_cfg.train_ratio,
        stride=train_cfg.stride,
    )

    torch.manual_seed(train_cfg.seed)
    model = GPTModel(model_cfg).to(device)
    optimizer = create_optimizer(model, train_cfg)

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=train_cfg.num_epochs,
        eval_freq=train_cfg.eval_freq,
        eval_iter=train_cfg.eval_iter,
        tokenizer=tokenizer,
        start_context=train_cfg.start_context,
        sample_tokens=train_cfg.sample_tokens,
        log=log,
    )
    return TrainingRun(model, optimizer, history, train_loader, val_loader)
