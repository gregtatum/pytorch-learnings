#!/usr/bin/env python3
from typing import cast
import torch
from torch import Tensor, nn
import sys
import argparse
from os import path
from sentencepiece import SentencePieceProcessor
import torch.nn.functional as F

from word_embedding import load_tokenizers

data_path = path.abspath(path.join(path.dirname(__file__), "../data"))
artifact_path = path.join(data_path, "embeddings")

tokens_en, tokens_es = load_tokenizers()


def process_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser(
        description="Find the nearest words based on embeddings."
    )
    parser.add_argument("--embeddings", type=str, help="Path to embeddings file")
    parser.add_argument("--word", type=str, help="Word to process")

    args = parser.parse_args()

    return args.embeddings, args.word


def load_embeddings(embeddings_path: str) -> tuple[nn.Embedding, Tensor]:
    embedding = nn.Embedding(5000, 5)
    state = torch.load(embeddings_path)
    embedding.load_state_dict(state)
    all_weights = state["weight"].to(torch.device("cpu"))
    return embedding, all_weights


def get_similarities(all_weights: Tensor, word_embedded: Tensor) -> list[str]:
    similarities = F.cosine_similarity(word_embedded, all_weights)
    results = list()
    for n in range(6):
        index = cast(int, torch.argmax(similarities).item())
        similarities[index] = 0
        results.append(tokens_en.decode([index]))
    return results


def main() -> None:
    embeddings_path, word = process_args()
    embeddings, all_weights = load_embeddings(embeddings_path)
    tokens = tokens_en.encode_as_ids(word)

    if len(tokens) != 1:
        print(f'"{word}" is multiple tokens, try another word.')
        sys.exit(0)

    word_embedded = embeddings(torch.tensor(tokens)).view((1, -1))

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    top_5 = sorted(all_weights, key=lambda x: cos(word_embedded, x))[:5]

    print(f'"{word}" - Most similar embeddings:')
    for w in get_similarities(all_weights, word_embedded):
        print(f"  • {w}")


if __name__ == "__main__":
    main()
