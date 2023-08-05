#!/usr/bin/env python3

"""
https://nn.labml.ai/transformers/mha.html
"""

import math
from typing import Optional, List

import torch
from torch import nn

from labml import tracker


class PrepareForMultiHeadAttention(nn.Module):
    """
    This module does a linear transformation and splits the vector into given number of
    heads for multi-head attention. This is used to transform key, query, and value
    vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        """
        Parameters:

          d_model: The dimension of the input vector (embedding size),
                   a hyper-parameter.
          heads:   The number of attention heads, a hyper-parameter.
          d_k:     The dimension of the key and query vectors, a hyper-parameter.
          bias:    A boolean flag indicating whether to include a bias term in the
                   linear transformation or not.
        """
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True
    ):
        """
        heads: The number of attention heads.
        d_model: The number of features in the `query`, `key`, and `values` vectors.
        """
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads

        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        self.softmax = nn.Softmax(dim=1)

        self.output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self._dk)
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        Calculate scores between queries and keys
        """
        return torch.einsum("ibhd,jbhd->ijbh", query, key)
