from os import path
from typing import Dict, List, Tuple

from sentencepiece import SentencePieceProcessor
from datasets import load_dataset

data_path = path.abspath(path.join(path.dirname(__file__), "../../data"))


class Tokens:
    source: SentencePieceProcessor
    target: SentencePieceProcessor

    def __init__(self, source: SentencePieceProcessor, target: SentencePieceProcessor):
        self.source = source
        self.target = target


def load_tokenizers(source_language: str, target_language: str) -> Tokens:
    model_source = path.join(data_path, "vocab", source_language + ".model")
    model_target = path.join(data_path, "vocab", target_language + ".model")

    if not path.exists(model_source):
        raise Exception(
            f"Can't find: {model_source}\nTry running src/build_vocab.py first."
        )
    if not path.exists(model_target):
        raise Exception(
            f"Can't find: {model_target}\nTry running src/build_vocab.py first."
        )

    tokens_source = SentencePieceProcessor(model_source)
    tokens_target = SentencePieceProcessor(model_target)

    return Tokens(tokens_source, tokens_target)


def load_data(
    source_language: str, target_language: str, small: bool = False
) -> list[dict]:
    if small:
        # Use a smaller text source for faster iterations.
        source_txt = path.join(data_path, "vocab", source_language + ".small.txt")
        target_txt = path.join(data_path, "vocab", target_language + ".small.txt")

        if not path.exists(source_txt):
            raise Exception(f"Can't find: {source_txt}")
        if not path.exists(target_txt):
            raise Exception(f"Can't find: {target_txt}")

        with open(source_txt, "r") as f:
            source_list = f.read().splitlines()
        with open(target_txt, "r") as f:
            target_list = f.read().splitlines()

        if len(source_list) != len(target_list):
            source_len = len(source_list)
            target_len = len(target_list)
            raise Exception(
                f"The source and target lists must be equal length\n"
                + f"  {source_txt} ({source_len})\n"
                + f"  {target_txt} ({target_len})"
            )

        return [
            dict([(source_language, a), (target_language, b)])
            for a, b in zip(source_list, target_list)
        ]

    # Load the dataset from para_crawl.
    print("Dataset: creating")
    dataset = load_dataset(
        "para_crawl", source_language + target_language, split="train"
    )
    print("Dataset: accessing")
    data = dataset["translation"]
    print("Dataset: loaded")
    return data
