#!/usr/bin/env python3

from typing import Any
from torch import Tensor
import torch
from transformer.model import Transformer
from transformer.parameters import HyperParameters
from transformer.utils import SentenceProcessor
from utils import get_device
import argparse
from utils.data_loader import load_tokenizers
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


args = process_args()
device = get_device()
p = HyperParameters(args.source, args.target)
tokens = load_tokenizers(p.source_language, p.target_language)

model = Transformer(
    p.source_vocab_size,
    p.target_vocab_size,
    p.d_model,
    p.num_heads,
    p.num_layers,
    p.d_ff,
    p.max_seq_length,
    p.dropout,
    device=device,
    padding_id=tokens.pad,
).to(device)

# Load in the model.
data_path = path.abspath(path.join(path.dirname(__file__), "../data"))
model_path = path.join(data_path, "transformer", args.param_hash, "model.pt")
if not path.exists(model_path):
    print("A model did not exist at the path:", model_path)
model.load_state_dict(torch.load(model_path))


sentence_processor = SentenceProcessor(tokens, p.max_seq_length, device)

source = torch.tensor(
    [sentence_processor.prep_source_sentence(args.sentence)], device=device
)
target = torch.tensor(
    [sentence_processor.apply_padding([tokens.target.bos_id()])], device=device
)

print("source", source)
print("target", target)


torch.set_printoptions(threshold=10000)

for i in range(1, p.max_seq_length):
    translation = model(source, target)[0]  # Only 1 translation is done here.
    # Translation size: [max_seq_length, target_vocab_size]

    best_token_index = torch.argmax(translation[i]).item()
    # Add the next token:
    target[0][i] = best_token_index

    if best_token_index == tokens.eos:
        # The end of the sentence was found
        break

ids = target[0].tolist()
print("ids:", ids)
print("translation:", tokens.target.decode(ids))
