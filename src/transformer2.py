#!/usr/bin/env python3

"""
An implementation of a transformer based off of:

https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
https://github.com/SamLynnEvans/Transformer/
"""

import torch
from torch import nn, Tensor
from torch.autograd import Variable
import math


class Embedder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        vocab_size: The size of the vocabulary.
        d_model: The dimension of the embedding vector.
        """
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Embedder.

        x: Input tensor representing a sequence of integer-encoded words.

        The embedding tensor for the input sequence.
        """
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 80):
        """
        d_model: The dimension of the embedding vector
        max_seq_len: The longest sequence of embedding vectors
        """
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on pos and i
        positional_vector: Tensor = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                positional_vector[pos, i] = math.sin(
                    pos / (10000 ** ((2 * i) / d_model))
                )
                positional_vector[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / d_model))
                )

        # Flattens the tensor
        positional_vector = positional_vector.unsqueeze(0)

        self.register_buffer("positional_vector", positional_vector)

    def forward(self, x: Tensor) -> Tensor:
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # Add constant to embedding
        seq_len = x.size(1)

        # TODO - Variable is depecrated and not needed.
        x = (
            x
            + Variable(self.positional_vector[:, :seq_len], requires_grad=False).cuda()
        )
        return x


print("The transformer")
