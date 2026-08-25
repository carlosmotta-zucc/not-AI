"""Acesso aos corpora usados no projeto.

Os textos ficam em data/, que nao e versionado (ver .gitignore): arquivo de
corpus nao entra no repositorio de codigo. Em compensacao o download precisa
ser reproduzivel, entao ele mora aqui e nao em instrucao de README.

Corpus da Sprint 2: "The Verdict", conto de Edith Wharton de 1908, o mesmo do
capitulo 2 do livro. Dominio publico, 20 479 caracteres. Ser pequeno e a
vantagem -- da para conferir as contagens na mao e o vocabulario inteiro cabe
na tela.
"""

import urllib.request
from pathlib import Path

# Raiz do repositorio: este arquivo esta em src/, entao dois niveis acima.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

VERDICT_FILENAME = "the-verdict.txt"
VERDICT_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)

# Tamanho esperado do arquivo integro. Serve de verificacao barata: download
# truncado ou arquivo trocado aparece como divergencia aqui, e nao como numero
# estranho no meio de um experimento.
VERDICT_EXPECTED_CHARS = 20_479


def load_verdict(force_download: bool = False) -> str:
    """Devolve o texto de "The Verdict", baixando na primeira vez.

    Args:
        force_download: refaz o download mesmo se o arquivo ja existir.

    Returns:
        O conto completo como string.
    """
    path = DATA_DIR / VERDICT_FILENAME

    if force_download or not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(VERDICT_URL, timeout=30) as response:
            path.write_text(response.read().decode("utf-8"), encoding="utf-8")

    text = path.read_text(encoding="utf-8")

    if len(text) != VERDICT_EXPECTED_CHARS:
        raise ValueError(
            f"{VERDICT_FILENAME} tem {len(text)} caracteres, "
            f"esperado {VERDICT_EXPECTED_CHARS}. Rode com force_download=True."
        )

    return text
