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
import argparse
from mlflow import log_metric, log_param, log_params, log_artifacts


"""
Train a transformer model.
"""


def process_args() -> Any:
    parser = argparse.ArgumentParser(
        description="Train a transformer model for translations."
    )
    parser.add_argument(
        "--source", type=str, help='The source language, e.g. "en"', required=True
    )
    parser.add_argument(
        "--target", type=str, help='The target language, e.g. "es"', required=True
    )
    parser.add_argument(
        "--small",
        help="Use a small test data set which is faster to load",
        action="store_true",
    )

    return parser.parse_args()


args = process_args()

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

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(
    model.parameters(),
    lr=p.learning_rate,
    betas=p.learning_betas,
    eps=p.learning_epsilon,
)

tokens = load_tokenizers(p.source_language, p.target_language)
data = load_data(p.source_language, p.target_language, small=True)

# Get the tokens for the "begin of a sentence" (bos) and "end of sentence" eos


# TODO - This could be made faster if this processing wasn't done on the fly.

sentence_processor = SentenceProcessor(tokens, p.max_seq_length)


def process_batch_data(data_slice: slice) -> tuple[Tensor, Tensor]:
    """
    Get a batch of data to process.
    """

    data_batch = data[data_slice]

    return torch.tensor(
        [
            sentence_processor.prep_source_sentence(sentence[p.source_language])
            for sentence in data_batch
        ],
        device=device,
    ), torch.tensor(
        [
            sentence_processor.prep_training_target_sentence(
                sentence[p.target_language]
            )
            for sentence in data_batch
        ],
        device=device,
    )


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
    # Zero out the gradients for this run.
    optimizer.zero_grad()

    source_data, target_data = process_batch_data(data_slice)

    output = model(source_data, target_data[:, :-1])

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
