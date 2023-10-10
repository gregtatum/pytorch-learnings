#!/usr/bin/env python3

from typing import Any
import torch
from transformer2.model import Transformer
from transformer2.parameters import HyperParameters
from transformer2.translate import translate_sentence
from utils import get_device
import argparse
from utils.data_loader import Tokens, load_tokenizers
from os import path

torch.manual_seed(1234)


def process_args() -> Any:
    parser = argparse.ArgumentParser(description="Translate a sentence")
    parser.add_argument(
        "--source", type=str, help='The source language, e.g. "en"', required=True
    )
    parser.add_argument(
        "--target", type=str, help='The target language, e.g. "es"', required=True
    )
    parser.add_argument(
        "--param_hash",
        type=str,
        help='The hash of the model params to load, e.g. "a2037ab29"',
        required=True,
    )
    parser.add_argument(
        "--sentence", type=str, help="The sentence to translate", required=True
    )

    return parser.parse_args()


def load_model(
    data_path: str, p: HyperParameters, tokens: Tokens, device: torch.device
) -> Transformer:
    model = Transformer(
        p.d_embedding,
        p.source_vocab_size,
        p.source_vocab_size,
        tokens.pad,
        p.num_heads,
        p.num_layers,  # num_encoder_layers: int,
        p.num_layers,  # num_decoder_layers: int,
        p.feed_forward_size,  # feed_forward_size: int,
        p.dropout,
        p.max_seq_length,
        device,
    ).to(device)

    # Load in the model.
    model_path = path.join(data_path, "model.pt")
    if not path.exists(model_path):
        print("A model did not exist at the path:", model_path)

    model.load_state_dict(torch.load(model_path))
    return model


def main():
    args = process_args()

    root_data_path = path.abspath(path.join(path.dirname(__file__), "../data"))
    data_path = path.join(root_data_path, "transformer", args.param_hash)

    device = get_device()
    p = HyperParameters.load_from_file(path.join(data_path, "hyperparameters.json"))
    tokens = load_tokenizers(p.source_language, p.target_language)
    model = load_model(data_path, p, tokens, device)

    print("Source sentence", args.sentence)
    output_sentence, output_tokens = translate_sentence(
        model,
        args.sentence,
        tokens,
        device,
    )
    print("Target sentence", output_sentence)
    print("Target tokens", output_tokens)


if __name__ == "__main__":
    main()
