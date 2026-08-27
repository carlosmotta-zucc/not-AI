"""Etapa 4.2 da Sprint 2: do Token ID ao vetor.

A Frente A terminou com uma conclusao negativa: o Token ID nao serve como
representacao semantica, porque e rotulo nominal. A ordem entre dois IDs nao
significa nada, a distancia entre eles nao mede parentesco, e a media de dois
IDs cai num terceiro token arbitrario.

A saida e trocar o escalar por um vetor de `emb_dim` dimensoes:

    (B, T) de Token IDs  --tabela de embeddings-->  (B, T, emb_dim)

A tabela e uma matriz (vocab_size, emb_dim) de pesos treinaveis. Cada linha e a
representacao de um token. Ela comeca em ruido aleatorio e e ajustada pelo
backpropagation junto com o resto da rede -- o significado nao e definido por
ninguem, e aprendido. E por isso que a camada precisa ser um nn.Module com
parametros, e nao uma tabela de consulta fixa.

Sobre usar nn.Embedding: e primitiva do PyTorch, no mesmo nivel de nn.Linear, e
nao um componente que a sprint pede para implementar (a regra do projeto mira
nn.MultiheadAttention e nn.Transformer, das Sprints 3 e 4). O envelope aqui
existe para documentar os shapes e expor a matriz de pesos aos experimentos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """Tabela de embeddings de tokens: um vetor treinavel por entrada do vocabulario."""

    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        """Cria a tabela (vocab_size, emb_dim) com pesos aleatorios.

        Args:
            vocab_size: numero de linhas -- vem de len(Vocabulary) ou de
                len(BPETokenizer), nunca de um literal.
            emb_dim: tamanho do vetor de cada token.
        """
        super().__init__()
        self.table = nn.Embedding(vocab_size, emb_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Troca cada Token ID pelo seu vetor.

        Args:
            token_ids: tensor de shape (B, T) e dtype torch.long. Aceita
                tambem (T,), caso em que a saida sai sem dimensao de lote.

        Returns:
            Tensor de shape (B, T, emb_dim), float32.

        Nao ha nenhuma conta sobre o valor do ID: ele so escolhe a linha. Dois
        IDs iguais, em posicoes diferentes, devolvem o MESMO vetor -- e o
        motivo pelo qual os positional embeddings precisam existir.
        """
        return self.table(token_ids)

    @property
    def weight(self) -> torch.Tensor:
        """Matriz de pesos, shape (vocab_size, emb_dim)."""
        return self.table.weight

    @property
    def vocab_size(self) -> int:
        return self.table.num_embeddings

    @property
    def emb_dim(self) -> int:
        return self.table.embedding_dim

    def parameter_count(self) -> int:
        """Numero de pesos treinaveis da tabela: vocab_size * emb_dim.

        Custo pago antes de qualquer camada de atencao existir, e a razao pela
        qual o tamanho do vocabulario e uma decisao de arquitetura.
        """
        return self.vocab_size * self.emb_dim


def one_hot_lookup(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Calcula o mesmo que TokenEmbedding.forward, mas via multiplicacao de matriz.

    Args:
        token_ids: tensor de shape (B, T), dtype torch.long.
        weight: matriz de embeddings, shape (vocab_size, emb_dim).

    Returns:
        Tensor de shape (B, T, emb_dim), identico ao da camada.

    Nao usar em treino: existe para o experimento provar uma equivalencia. A
    camada de embedding costuma ser apresentada como "uma tabela de consulta" e
    a codificacao one-hot como "a alternativa ingenua", mas as duas sao a mesma
    operacao -- multiplicar um vetor one-hot pela matriz de pesos seleciona
    exatamente uma linha dela.

    A diferenca e so de custo: aqui a matriz esparsa (B, T, vocab_size) e
    materializada, e com vocab_size = 50 257 isso e centenas de megabytes de
    zeros para buscar algumas linhas. A camada indexa direto e pula tudo isso.
    """
    one_hot = F.one_hot(token_ids, num_classes=weight.shape[0]).to(weight.dtype)
    return one_hot @ weight
