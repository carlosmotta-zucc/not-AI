"""Etapa 4.3 da Sprint 2: acrescentar a posicao ao vetor do token.

A tabela de tokens tem um ponto cego declarado: ela e indexada so pelo ID,
entao em "o gato mordeu o rato" os dois "o" recebem vetores identicos. A
camada de atencao da Sprint 3 processa todas as posicoes em paralelo, sem
recorrencia e sem convolucao -- nada no caminho carrega a ordem das palavras.
Sem intervencao, "o gato mordeu o rato" e "o rato mordeu o gato" chegariam na
rede como o mesmo conjunto de vetores.

A solucao do GPT e uma segunda tabela, indexada pela POSICAO em vez do token, e
somada a primeira:

    (B, T, D) do token  +  (T, D) da posicao  =  (B, T, D) de entrada

Somar, e nao concatenar, e o que mantem a dimensionalidade constante ao longo
de toda a rede. O preco e que conteudo e posicao dividem o mesmo espaco
vetorial, e cabe ao treino aprender a separar os dois.

Estas sao posicoes ABSOLUTAS e TREINAVEIS, como no GPT-2: a posicao 0 tem seu
vetor, a posicao 1 tem outro, e ambos sao ajustados por backpropagation. O
livro original do Transformer usa senos e cossenos fixos; modelos recentes usam
posicoes relativas (RoPE). A escolha aqui segue o modelo que o projeto
reimplementa.
"""

import torch
import torch.nn as nn

from .token_embeddings import TokenEmbedding


class PositionalEmbedding(nn.Module):
    """Tabela de embeddings de posicao: um vetor treinavel por posicao da janela."""

    def __init__(self, context_length: int, emb_dim: int) -> None:
        """Cria a tabela (context_length, emb_dim) com pesos aleatorios.

        Args:
            context_length: numero de posicoes -- o teto de tokens que o modelo
                consegue enxergar de uma vez. Diferente do vocabulario, este
                limite e rigido: nao ha linha para a posicao context_length.
            emb_dim: tamanho do vetor, o mesmo do TokenEmbedding para que a
                soma seja possivel.
        """
        super().__init__()
        self.table = nn.Embedding(context_length, emb_dim)

    def forward(self, sequence_length: int | None = None) -> torch.Tensor:
        """Devolve os vetores das primeiras `sequence_length` posicoes.

        Args:
            sequence_length: quantas posicoes retornar; por padrao, todas.

        Returns:
            Tensor de shape (sequence_length, emb_dim), SEM dimensao de lote.

        Returns sem dimensao de lote e proposital: a posicao 2 e a posicao 2
        para todas as sequencias do lote, entao nao ha o que replicar. Quem
        cuida disso e o broadcasting na hora da soma.

        Raises:
            ValueError: se `sequence_length` passar do context_length.
        """
        if sequence_length is None:
            sequence_length = self.context_length

        if not 1 <= sequence_length <= self.context_length:
            raise ValueError(
                f"sequence_length={sequence_length} fora do intervalo "
                f"[1, {self.context_length}] da tabela de posicoes"
            )

        positions = torch.arange(sequence_length, device=self.table.weight.device)
        return self.table(positions)

    @property
    def weight(self) -> torch.Tensor:
        """Matriz de pesos, shape (context_length, emb_dim)."""
        return self.table.weight

    @property
    def context_length(self) -> int:
        return self.table.num_embeddings

    @property
    def emb_dim(self) -> int:
        return self.table.embedding_dim


class InputEmbedding(nn.Module):
    """Pipeline completo do Token ID a entrada da primeira camada do modelo.

    Junta as duas tabelas e devolve o tensor que a Sprint 3 vai consumir.
    """

    def __init__(self, vocab_size: int, emb_dim: int, context_length: int) -> None:
        """Cria as duas tabelas, ambas com a mesma emb_dim.

        Args:
            vocab_size: linhas da tabela de tokens.
            emb_dim: dimensao compartilhada pelas duas tabelas.
            context_length: linhas da tabela de posicoes.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, emb_dim)
        self.positional_embedding = PositionalEmbedding(context_length, emb_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Converte um lote de Token IDs no lote de vetores de entrada.

        Args:
            token_ids: tensor de shape (B, T), dtype torch.long, com
                T <= context_length.

        Returns:
            Tensor de shape (B, T, emb_dim), float32.
        """
        sequence_length = token_ids.shape[-1]

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.positional_embedding(sequence_length)

        return token_vectors + position_vectors

    @property
    def emb_dim(self) -> int:
        return self.token_embedding.emb_dim

    def parameter_count(self) -> int:
        """Pesos treinaveis das duas tabelas somadas."""
        return sum(parameter.numel() for parameter in self.parameters())
