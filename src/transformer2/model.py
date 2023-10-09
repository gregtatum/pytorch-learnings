import torch
import torch.nn as nn
from torch import Tensor


# TODO - Use this tutorial instead:

"""
This model uses the built-in pytorch modules for implementing the typical Transformer
architecture.

# https://www.youtube.com/watch?v=M6adRGJe5cQ
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/558557c7989f0b10fee6e8d8f953d7269ae43d4f/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py#L75
"""


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        source_vocab_size: int,
        target_vocab_size: int,
        source_pad_idx: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        feed_forward_size: int,
        dropout: float,
        max_seq_length: int,
        device: torch.device,
    ) -> None:
        super(Transformer, self).__init__()

        self.source_word_embedding = nn.Embedding(source_vocab_size, embedding_size)
        self.target_word_embedding = nn.Embedding(target_vocab_size, embedding_size)

        # Use learned position embeddings, rather than using positional sin waves.
        self.source_position_embedding = nn.Embedding(max_seq_length, embedding_size)
        self.target_position_embedding = nn.Embedding(max_seq_length, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            feed_forward_size,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.source_pad_idx = source_pad_idx
        self.max_seq_length = max_seq_length

    def make_source_mask(self, source: Tensor) -> Tensor:
        source_mask = source.transpose(0, 1) == self.source_pad_idx
        # (N, source_len)

        return source_mask.to(self.device)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        source_seq_length, N = source.shape
        target_seq_length, N = target.shape

        assert source_seq_length <= self.max_seq_length
        assert target_seq_length <= self.max_seq_length

        # TODO - Cache the previous value here, there is no reason to re-compute it.
        # Here is an example of
        #   source_seq_length == 4
        #   N = 10
        # tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
        source_positions = (
            torch.arange(0, source_seq_length)
            .unsqueeze(1)
            .expand(source_seq_length, N)
            .to(self.device)
        )

        # TODO - Cache the previous value here, there is no reason to re-compute it.
        target_positions = (
            torch.arange(0, target_seq_length)
            .unsqueeze(1)
            .expand(target_seq_length, N)
            .to(self.device)
        )

        embed_source = self.dropout(
            self.source_word_embedding(source)
            + self.source_position_embedding(source_positions)
        )

        embed_target = self.dropout(
            self.target_word_embedding(target)
            + self.target_position_embedding(target_positions)
        )

        source_mask = self.make_source_mask(source)
        target_mask = self.transformer.generate_square_subsequent_mask(
            target_seq_length
        ).to(self.device)

        out = self.transformer(
            embed_source,
            embed_target,
            src_key_padding_mask=source_mask,
            tgt_mask=target_mask,
        )

        return self.fc_out(out)
