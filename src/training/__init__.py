"""Avaliação e treinamento do GPT (Sprint 5, capítulo 5, seções 5.1 e 5.2).

    texto -> DataLoaders de treino e de validação                      (data.py)
          -> lotes (B, T) -> GPTModel -> logits (B, T, V)
          -> cross-entropy -> backward -> passo do AdamW -> pesos novos  (loss.py, trainer.py)
                           +-> perdas de treino e validação, perplexidade

- config.py -- TrainingConfig e as configurações prontas: BOOK_MODEL_CONFIG e
  BOOK_TRAINING_CONFIG (o treino do capítulo 5) e BASE_TRAINING_CONFIG (o
  treino de referência do projeto, sobre o BASE_CONFIG da Sprint 4).
- loss.py -- a perda de um lote, a de um DataLoader inteiro, a perplexidade.
- data.py -- divisão treino/validação e os dois DataLoaders.
- trainer.py -- o laço de treino, o histórico das perdas e train_gpt, que
  faz tudo na ordem do livro.

Correspondência com os nomes do livro:

    livro (seções 5.1 e 5.2)    projeto
    -------------------------   ------------------------------------
    calc_loss_batch             batch_loss
    calc_loss_loader            loader_loss
    evaluate_model              evaluate
    train_model_simple          train_model
    generate_and_print_sample   generate_sample (o print fica no log)
    torch.exp(loss)             perplexity

A decodificação (gulosa, temperatura, top-k) continua em src/generation/: o
treino só usa a gulosa, para a amostra de cada época.
"""

from .config import (
    BASE_TRAINING_CONFIG,
    BOOK_MODEL_CONFIG,
    BOOK_START_CONTEXT,
    BOOK_TRAINING_CONFIG,
    TrainingConfig,
)
from .data import TRAIN_RATIO, create_train_val_loaders, split_text
from .loss import (
    batch_loss,
    evaluate,
    loader_loss,
    perplexity,
    target_log_probabilities,
)
from .trainer import (
    TrainingHistory,
    TrainingRun,
    create_optimizer,
    generate_sample,
    train_gpt,
    train_model,
)

__all__ = [
    "BASE_TRAINING_CONFIG",
    "BOOK_MODEL_CONFIG",
    "BOOK_START_CONTEXT",
    "BOOK_TRAINING_CONFIG",
    "TRAIN_RATIO",
    "TrainingConfig",
    "TrainingHistory",
    "TrainingRun",
    "batch_loss",
    "create_optimizer",
    "create_train_val_loaders",
    "evaluate",
    "generate_sample",
    "loader_loss",
    "perplexity",
    "split_text",
    "target_log_probabilities",
    "train_gpt",
    "train_model",
]
