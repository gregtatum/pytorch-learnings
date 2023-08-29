#!/usr/bin/env python3

"""
This file loads a parallel corpus dataset from ParaCrawl, and generates a tokenization
through SentencePiece.

└── data/vocab
    ├── {source_lang}.model
    ├── {source_lang}.vocab
    ├── {source_lang}.small.txt
    ├── {source_lang}.txt
    ├── {target_lang}.model
    ├── {target_lang}.vocab
    ├── {target_lang}.small.txt
    └── {target_lang}.txt

https://github.com/google/sentencepiece
https://github.com/google/sentencepiece/tree/master/python
"""

import argparse
from typing import Any, Optional, TypedDict
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from os import mkdir, path


def process_args() -> Any:
    parser = argparse.ArgumentParser(description="Build a vocab for two languages")
    parser.add_argument(
        "--source", type=str, help='The source language, e.g. "en"', required=True
    )
    parser.add_argument(
        "--target", type=str, help='The target language, e.g. "es"', required=True
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="How many words to generate for a vocab",
        default=5000,
    )
    parser.add_argument(
        "--input_sentence_size",
        type=int,
        help="How many words to generate for a vocab",
        default=10000,
    )

    return parser.parse_args()


args = process_args()

source_lang = args.source
target_lang = args.target

ParaCrawlDataset = TypedDict("ParaCrawlDataset", {"translation": list[dict[str, str]]})

dataset: Optional[ParaCrawlDataset] = None  # Lazily initialized


def get_dataset() -> ParaCrawlDataset:
    global dataset
    if not dataset:
        # Shuffle only so that the demo of the output is more interesting. In production
        # this wouldn't be needed.
        dataset = load_dataset("para_crawl", "enes", split="train").shuffle()
    return dataset


def write_language_text_file(output_path: str, lang: str) -> None:
    output_path_big = output_path + ".txt"
    output_path_small = output_path + ".small.txt"

    if path.exists(output_path_big):
        print(f"The text file '{output_path_big}' already exists.")
    else:
        print(f"Writing out file '{output_path_big}'.")
        with open(output_path_big, "w", encoding="utf-8") as f:
            for row in get_dataset()["translation"]:
                f.write(row[lang] + "\n")

    if path.exists(output_path_small):
        print(f"The text file '{output_path_small}' already exists.")
    else:
        print(f"Writing out file '{output_path_small}'.")
        with open(output_path_big, "w", encoding="utf-8") as f:
            for row in get_dataset()["translation"][:10_000]:
                f.write(row[lang] + "\n")


vocab_path = path.abspath(path.join(path.dirname(__file__), "../data/vocab"))
if not path.exists(vocab_path):
    mkdir(vocab_path)

text_en = path.join(vocab_path, source_lang)
text_es = path.join(vocab_path, target_lang)
model_en = path.join(vocab_path, f"{source_lang}.model")
model_es = path.join(vocab_path, f"{target_lang}.model")

write_language_text_file(text_en, source_lang)
write_language_text_file(text_es, target_lang)


def run_sentence_piece(lang: str) -> None:
    model_path = path.join(vocab_path, f"{lang}.model")
    if path.exists(model_path):
        print(f"The model file already exists: {model_path}")
    else:
        print(f"Running the training for {lang}")
        SentencePieceTrainer.train(
            input=text_en,
            model_prefix=f"data/vocab/{lang}",
            vocab_size=args.vocab_size,
            input_sentence_size=args.input_sentence_size,
        )


run_sentence_piece(source_lang)
run_sentence_piece(target_lang)


def output_sentence_piece(lang: str) -> None:
    model_path = path.join(vocab_path, f"{lang}.model")
    text_path = path.join(vocab_path, f"{lang}.txt")
    sp: SentencePieceProcessor = SentencePieceProcessor()
    sp.load(model_path)

    print("=============================================================")
    print(f"Examples for {lang}")
    print("=============================================================")

    with open(text_path, "r", encoding="utf-8") as file:
        # Read the first few lines
        for line_number, line in enumerate(file):
            print("\n")
            print(line[:-1])
            print(sp.encode_as_pieces(line))
            print(sp.encode_as_ids(line))
            if line_number == 49:
                break


output_sentence_piece(source_lang)
output_sentence_piece(target_lang)


def max_length(lang: str) -> int:
    model_path = path.join(vocab_path, f"{lang}.model")
    text_path = path.join(vocab_path, f"{lang}.txt")
    sp = SentencePieceProcessor()
    sp.load(model_path)

    max_length = 0

    with open(text_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            length = len(sp.encode_as_ids(line))
            max_length = max(max_length, length)
            if line_number == 100_000:
                break

    return max_length


print(f"Max length for {source_lang}:", max_length(source_lang))
