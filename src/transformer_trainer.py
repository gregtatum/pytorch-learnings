#!/usr/bin/env python3
from os import path
from typing import List, TypedDict
from numpy import pad
from transformer import Transformer
from utils import get_device
from utils.trainer_manager import TrainerManager
import torch
from torch import nn, optim, Tensor
from utils.data_loader import load_test_data
import torch.nn.functional as F

"""
Train a transformer model.
"""

device = get_device()


class HyperParameters:
    def __init__(self):
        self.source_language = "en"
        self.target_language = "es"
        self.source_vocab_size = 5000
        self.target_vocab_size = 5000
        self.batch_size = 256
        self.d_model = 512  # d_k := 64
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        self.max_seq_length = 100
        self.dropout = 0.1
        self.learning_rate = 0.0001
        self.learning_betas = (0.9, 0.98)
        self.learning_epsilon = 1e-9


p = HyperParameters()

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

tokens, data = load_test_data(p.source_language, p.target_language, small=True)

# TODO - Differentiate between batch and epoch.


def process_batch_data(data_slice: slice):
    """
    Get a batch of data to process.
    """

    data_batch = data[data_slice]

    # Ensures the sentence is zero padded to the correct tensor size.
    def zero_pad(list):
        if len(list) > p.max_seq_length:
            list = list[: p.max_seq_length]
        while len(list) < p.max_seq_length:
            list.append(0)
        return list

    return torch.tensor(
        [
            zero_pad(tokens.source.encode_as_ids(sentence[p.source_language]))
            for sentence in data_batch
        ],
        device=device,
    ), torch.tensor(
        [
            zero_pad(tokens.target.encode_as_ids(sentence[p.target_language]))
            for sentence in data_batch
        ],
        device=device,
    )


def save_model(artifact_path):
    torch.save(
        model.state_dict(),
        artifact_path("model.pt"),
    )
    torch.save(
        optimizer.state_dict(),
        artifact_path("optimizer.pt"),
    )


def load_model(artifact_path):
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
    #
    batch_size=p.batch_size,
    num_epochs=100,
    data_size=len(data),
    save_model=save_model,
    load_model=load_model,
)
