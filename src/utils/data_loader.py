from os import path
from typing import Dict, List, Tuple

from sentencepiece import SentencePieceProcessor
from datasets import load_dataset

data_path = path.abspath(path.join(path.dirname(__file__), "../../data"))


class Tokens:
    def __init__(
        self,
        source: SentencePieceProcessor,
        target: SentencePieceProcessor,
        source_language: str,
        target_language: str,
    ):
        self.source = source
        self.target = target
        self.source_language = source_language
        self.target_language = target_language

        self.pad: int = source.pad_id()
        self.bos: int = source.bos_id()
        self.eos: int = source.eos_id()
        self.unk: int = source.unk_id()

        assert source.pad_id() == target.pad_id(), "The pad id matches"
        assert source.bos_id() == target.bos_id(), "The bos id matches"
        assert source.eos_id() == target.eos_id(), "The eos id matches"
        assert source.unk_id() == target.unk_id(), "The unk id matches"


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

    tokens_source = SentencePieceProcessor(
        model_file=model_source, add_bos=True, add_eos=True
    )
    tokens_target = SentencePieceProcessor(
        model_file=model_target, add_bos=True, add_eos=True
    )

    return Tokens(tokens_source, tokens_target, source_language, target_language)


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
