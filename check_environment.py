"""Verificacao do ambiente do Projeto LLM (Sprint 0).

Confere se o PyTorch esta instalado e funcional na maquina: criacao de tensores,
autograd, um passo de treino minimo e reprodutibilidade por seed. Tambem checa as
demais dependencias do requirements.txt.

Uso:
    source venv/bin/activate
    python check_environment.py
"""

import platform
import sys

import torch
import torch.nn as nn

# Seed fixa: todo experimento do projeto precisa ser reproduzivel, e esta
# verificacao serve justamente para provar que a seed funciona.
SEED = 42


def report_versions() -> None:
    """Imprime versoes do interpretador, do PyTorch e o dispositivo em uso."""
    print("== Ambiente ==")
    print(f"Python           : {sys.version.split()[0]} ({platform.machine()})")
    print(f"PyTorch          : {torch.__version__}")
    print(f"CUDA disponivel  : {torch.cuda.is_available()}")
    print(f"Threads de CPU   : {torch.get_num_threads()}")
    print()


def check_tensor_ops() -> None:
    """Cria tensores e valida uma multiplicacao de matrizes conhecida."""
    print("== Tensores ==")

    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # shape (2, 2)
    product = left @ right  # shape (2, 2)

    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(product, expected), "matmul devolveu resultado inesperado"

    print(f"left @ right     :\n{product}")
    print(f"shape / dtype    : {tuple(product.shape)} / {product.dtype}")
    print()


def check_autograd() -> None:
    """Deriva y = x^2 em x = 3 e confere que o gradiente vale 2x = 6."""
    print("== Autograd ==")

    x = torch.tensor(3.0, requires_grad=True)
    y = x**2
    y.backward()

    assert torch.isclose(x.grad, torch.tensor(6.0)), "gradiente incorreto"

    print(f"y = x^2 em x=3   : y={y.item():.1f}, dy/dx={x.grad.item():.1f}")
    print()


def check_training_step() -> None:
    """Treina uma regressao linear minima e verifica que a perda cai.

    Nao e um modelo util: e o menor circuito que exercita camada, funcao de perda,
    otimizador e backpropagation de uma vez so.
    """
    print("== Passo de treino ==")

    torch.manual_seed(SEED)

    features = torch.randn(64, 3)  # shape (64, 3)
    weights = torch.tensor([[2.0], [-1.0], [0.5]])  # shape (3, 1)
    targets = features @ weights + 1.0  # shape (64, 1)

    model = nn.Linear(in_features=3, out_features=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    initial_loss = None
    for step in range(100):
        predictions = model(features)  # shape (64, 1)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    assert final_loss < initial_loss, "a perda nao caiu; algo esta errado no treino"

    print(f"perda inicial    : {initial_loss:.6f}")
    print(f"perda final      : {final_loss:.6f}")
    print(f"pesos aprendidos : {model.weight.detach().flatten().tolist()}")
    print(f"pesos esperados  : {weights.flatten().tolist()}")
    print()


def check_seed_reproducibility() -> None:
    """Gera o mesmo tensor aleatorio duas vezes a partir da mesma seed."""
    print("== Reprodutibilidade ==")

    torch.manual_seed(SEED)
    first = torch.randn(5)

    torch.manual_seed(SEED)
    second = torch.randn(5)

    assert torch.equal(first, second), "mesma seed produziu tensores diferentes"

    print(f"seed {SEED} -> {first.tolist()}")
    print("duas execucoes com a mesma seed produziram o mesmo tensor")
    print()


def check_dependencies() -> None:
    """Importa as demais dependencias do requirements.txt e reporta as versoes."""
    print("== Dependencias ==")

    for module_name in ("numpy", "matplotlib", "tiktoken"):
        try:
            module = __import__(module_name)
            print(f"{module_name:<16} : {getattr(module, '__version__', 'ok')}")
        except ImportError as error:
            print(f"{module_name:<16} : FALTANDO ({error})")
    print()


def main() -> int:
    report_versions()
    check_tensor_ops()
    check_autograd()
    check_training_step()
    check_seed_reproducibility()
    check_dependencies()
    print("Ambiente OK: PyTorch instalado e funcional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
