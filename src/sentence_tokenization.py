#!/usr/bin/env python3

"""
This file loads the Spanish/English dataset from ParaCrawl, and generates a tokenization
through SentencePiece.

└── data
    ├── en.model
    ├── en.txt
    ├── en.vocab
    ├── es.model
    ├── es.txt
    └── es.vocab

https://github.com/google/sentencepiece
https://github.com/google/sentencepiece/tree/master/python
"""

from typing import Optional, TypedDict
from datasets import load_dataset
import sentencepiece as spm
from os import path

ParaCrawlDataset = TypedDict("ParaCrawlDataset", {"translation": list[dict[str, str]]})

dataset: Optional[ParaCrawlDataset] = None  # Lazily initialized


def get_dataset() -> ParaCrawlDataset:
    global dataset
    if not dataset:
        dataset = load_dataset("para_crawl", "enes", split="train").shuffle()
    return dataset


def write_file(output_path: str, lang: str) -> None:
    if path.exists(output_path):
        print(f"The text file '{output_path}' already exists.")
    else:
        print(f"Writing out file '{output_path}'.")
        with open(output_path, "w", encoding="utf-8") as f:
            for row in get_dataset()["translation"]:
                f.write(row[lang] + "\n")


data_path = path.abspath(path.join(path.dirname(__file__), "../data"))
text_en = path.join(data_path, "en.txt")
text_es = path.join(data_path, "es.txt")
model_en = path.join(data_path, "en.model")
model_es = path.join(data_path, "es.model")

write_file(text_en, "en")
write_file(text_es, "es")


def run_sentence_piece(lang: str) -> None:
    model_path = path.join(data_path, f"{lang}.model")
    if path.exists(model_path):
        print(f"The model file already exists: {model_path}")
    else:
        print(f"Running the training for {lang}")
        spm.SentencePieceTrainer.train(
            input=text_en,
            model_prefix=f"data/{lang}",
            vocab_size=5000,
            input_sentence_size=10000,
        )


run_sentence_piece("en")
run_sentence_piece("es")


def output_sentence_piece(lang: str) -> None:
    model_path = path.join(data_path, f"{lang}.model")
    text_path = path.join(data_path, f"{lang}.txt")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    print("=============================================================")
    print(f"Examples for {lang}")
    print("=============================================================")

    with open(text_path, "r", encoding="utf-8") as file:
        # Read the first 10 lines
        for line_number, line in enumerate(file):
            print("\n")
            print(line[:-1])
            print(sp.encode_as_pieces(line))
            print(sp.encode_as_ids(line))
            if line_number == 49:
                break


output_sentence_piece("en")
output_sentence_piece("es")


def max_length(lang: str) -> int:
    model_path = path.join(data_path, f"{lang}.model")
    text_path = path.join(data_path, f"{lang}.txt")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    max_length = 0

    with open(text_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            length = len(sp.encode_as_ids(line))
            max_length = max(max_length, length)
            if line_number == 100_000:
                break

    return max_length


print("Max length for en:", max_length("en"))
