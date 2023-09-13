"""
https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
"""

import math
import torch
from torch import nn, optim, Tensor
from typing import Optional, cast


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, device: Optional[torch.device] = None
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, device=device)
        self.W_k = nn.Linear(d_model, d_model, device=device)
        self.W_v = nn.Linear(d_model, d_model, device=device)
        self.W_o = nn.Linear(d_model, d_model, device=device)

    def scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: Optional[torch.device] = None
    ) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, device=device)
        self.fc2 = nn.Linear(d_ff, d_model, device=device)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, max_seq_length: int, device: Optional[torch.device] = None
    ) -> None:
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model, device=device)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        pe = cast(Tensor, self.pe)
        return x + pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        device: Optional[torch.device] = None,
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, device=device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, device=device)
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, enc_output: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float,
        device: Optional[torch.device] = None,
        # The token ID that represents padding.
        padding_id: int = 0,
    ) -> None:
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, device=device)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, device=device)
        self.positional_encoding = PositionalEncoding(
            d_model, max_seq_length, device=device
        )

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.padding_id = padding_id

    def generate_mask(
        self, source: Tensor, target: Tensor, device: Optional[torch.device] = None
    ) -> tuple[Tensor, Tensor]:
        # Consider `0` equal to a padding value that should be masked, the `!= 0` converts
        # the tensor into one of `int` indexes, into boolean values.
        src_mask = (source != self.padding_id).unsqueeze(1).unsqueeze(2)
        # e.g. src_mask.shape torch.Size([1, 1, 1, 100])
        tgt_mask = (target != self.padding_id).unsqueeze(1).unsqueeze(3)
        # e.g. tgt_mask.shape torch.Size([1, 1, 100, 1])

        seq_length = target.size(1)
        nopeak_mask = (
            1
            - torch.triu(
                torch.ones(1, seq_length, seq_length, device=device), diagonal=1
            )
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_mask, tgt_mask = self.generate_mask(src, tgt, device=self.device)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
