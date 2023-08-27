# MIT License
#
# Copyright (c) 2023 Gabriel Chaperon
#
# Taken from:
# https://github.com/gchaperon/align-and-translate/blob/ba6ba54ec16e131092ef90ae091e201cc597362b/LICENSE

"""Typing stubs for the ``sentencepieces`` package.

The Python version of this package is poorly documented, so I'm trying to do
what I can using the examples
[here](https://github.com/google/sentencepiece/tree/master/python).
"""
from typing import BinaryIO, Iterator, overload

class SentencePieceProcessor:
    def __init__(self, model_file: str, num_threads: int = -1) -> None: ...
    def id_to_piece(self, id: int) -> str: ...
    def encode_as_ids(self, input: str) -> list[int]: ...
    @overload
    def encode(
        self,
        input: str,
        out_type: type[int] | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
        revers: bool = False,
        num_threads: int = -1,
    ) -> list[int]: ...
    @overload
    def encode(
        self,
        input: list[str],
        out_type: type[int] | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
        revers: bool = False,
        num_threads: int = -1,
    ) -> list[list[int]]: ...
    @overload
    def encode(
        self,
        input: str,
        out_type: type[str],
        add_bos: bool = False,
        add_eos: bool = False,
        revers: bool = False,
        num_threads: int = -1,
    ) -> list[str]: ...
    @overload
    def encode(
        self,
        input: list[str],
        out_type: type[str],
        add_bos: bool = False,
        add_eos: bool = False,
        revers: bool = False,
        num_threads: int = -1,
    ) -> list[list[str]]: ...
    def get_score(self, id: int) -> float: ...
    def get_piece_size(self) -> int: ...
    def vocab_size(self) -> int: ...

class SentencePieceTrainer:
    @staticmethod
    def train(
        sentence_iterator: Iterator[str],
        model_writer: BinaryIO,
        vocab_size: int = 8000,
        character_coverage: float = 0.9995,
        pad_id: int = -1,
    ) -> None: ...
