#!/usr/bin/env python3
from os import path
from typing import Any
from transformer.model import Transformer
from transformer.parameters import HyperParameters
from transformer.utils import SentenceProcessor
from utils import get_device
from transformer.trainer_manager import ArtifactPathFn, TrainerManager
import torch
from torch import nn, optim, Tensor
from utils.data_loader import load_data, load_tokenizers
from tap import Tap
from torch.nn.utils.rnn import pad_sequence

"""
Train a transformer model.
"""


class Arguments(Tap):
    """
    Train a transformer model for translations.

    - Train a language on the full corpus:
      poetry run src/transformer_trainer.py -- --source "en" --target "es"

    - Test the system on a small subset of the corpus:
      poetry run src/transformer_trainer.py -- --source "en" --target "es" --small
    """

    source: str  # The source language, e.g. "en"
    target: str  # The target language, e.g. "es"
    small: bool = False  # Use a small test data set which is faster to load


args = Arguments().parse_args()

device = get_device()
p = HyperParameters(args.source, args.target)
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
).to(device)

# Set the model to training mode.
model.train()

# Generate random sample data
torch.manual_seed(1234)

tokens = load_tokenizers(p.source_language, p.target_language)
criterion = nn.CrossEntropyLoss(ignore_index=tokens.pad)
optimizer = optim.Adam(
    model.parameters(),
    lr=p.learning_rate,
    betas=p.learning_betas,
    eps=p.learning_epsilon,
)

data = load_data(p.source_language, p.target_language, small=args.small)

# Get the tokens for the "begin of a sentence" (bos) and "end of sentence" eos


# TODO - This could be made faster if this processing wasn't done on the fly.

sentence_processor = SentenceProcessor(tokens, p.max_seq_length, device)


def save_model(artifact_path: ArtifactPathFn) -> None:
    torch.save(
        model.state_dict(),
        artifact_path("model.pt"),
    )
    torch.save(
        optimizer.state_dict(),
        artifact_path("optimizer.pt"),
    )


def load_model(artifact_path: ArtifactPathFn) -> None:
    if path.exists(artifact_path("model.pt")):
        print("Loading a saved model")
        model.load_state_dict(torch.load(artifact_path("model.pt")))

    if path.exists(artifact_path("optimizer.pt")):
        print("Loading a saved optimizer")
        optimizer.load_state_dict(torch.load(artifact_path("optimizer.pt")))


def train_step(data_slice: slice) -> float:
    source_data, target_data = sentence_processor.get_batch(data, data_slice)
    output = model(source_data, target_data[:, :-1])

    # Perform backward propagation.
    optimizer.zero_grad()
    loss = criterion(
        output.contiguous().view(-1, p.target_vocab_size),
        target_data[:, 1:].contiguous().view(-1),
    )
    loss.backward()
    optimizer.step()

    return loss.item()


manager = TrainerManager("transformer", model, p)
manager.train(
    train_step,
    name=f"Translations Model: {args.source}-{args.target}",
    batch_size=p.batch_size,
    num_epochs=100,
    data_size=len(data),
    save_model=save_model,
    load_model=load_model,
)
