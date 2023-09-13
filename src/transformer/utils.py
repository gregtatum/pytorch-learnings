from os import path
import torch
from torch import Tensor
from utils.data_loader import Tokens


data_path = path.abspath(path.join(path.dirname(__file__), "../data"))


class SentenceProcessor:
    """
    Prepares source and target sentences for the Transformer model
    """

    def __init__(
        self, tokens: Tokens, max_seq_length: int, device: torch.device
    ) -> None:
        self.tokens = tokens
        self.pad = tokens.pad
        self.max_seq_length = max_seq_length
        self.device = device

    def apply_padding(self, list: list[int]) -> list[int]:
        """Ensures the sentence is zero padded to the correct tensor size."""
        if len(list) > self.max_seq_length:
            list = list[: self.max_seq_length]
        while len(list) < self.max_seq_length:
            list.append(self.pad)
        return list

    def prep_source_sentence(self, sentence: str) -> list[int]:
        return self.apply_padding(self.tokens.source.encode_as_ids(sentence))

    def prep_target_sentence(self, sentence: str) -> list[int]:
        return self.apply_padding(self.tokens.target.encode_as_ids(sentence))

    def get_batch(
        self,
        data: list[dict[str, str]],
        data_slice: slice,
    ) -> tuple[Tensor, Tensor]:
        """
        Get a batch of data to process.
        """

        data_batch = data[data_slice]

        source = torch.tensor(
            [
                self.prep_source_sentence(sentence[self.tokens.source_language])
                for sentence in data_batch
            ],
            device=self.device,
        )

        target = torch.tensor(
            [
                self.prep_target_sentence(sentence[self.tokens.target_language])
                for sentence in data_batch
            ],
            device=self.device,
        )

        return source, target
