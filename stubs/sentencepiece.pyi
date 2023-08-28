from typing import Optional

class SentencePieceProcessor:
    def __init__(self, model_path: Optional[str] = None) -> None: ...
    def id_to_piece(self, id: int) -> str: ...
    def piece_to_id(self, piece: str) -> int: ...
    def encode_as_ids(self, input: str) -> list[int]: ...
    def encode_as_pieces(self, input: str) -> list[str]: ...
    def decode(self, input: list[int]) -> str: ...
    def load(self, model_path: str) -> None: ...
    def get_score(self, id: int) -> float: ...
    def get_piece_size(self) -> int: ...
    def vocab_size(self) -> int: ...

class SentencePieceTrainer:
    @staticmethod
    def train(
        input: str = "",
        model_prefix: str = "",
        vocab_size: int = 8000,
        input_sentence_size: int = 10000,
    ) -> None: ...
