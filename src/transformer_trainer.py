#!/usr/bin/env python3
from transformer import Transformer
from utils import naive_hash
from utils.trainer_manager import TrainerManager
import torch
from torch import nn, optim, Tensor

"""
Train a transformer model.
"""

# The hyper parameters.
p = {
    "src_vocab_size": 5000,
    "tgt_vocab_size": 5000,
    "d_model": 512,  # d_k := 64
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 2048,
    "max_seq_length": 100,
    "dropout": 0.1,
    "learning_rate": 0.0001,
    "learning_betas": (0.9, 0.98),
    "learning_epsilon": 1e-9,
}

model = Transformer(
    p["src_vocab_size"],
    p["tgt_vocab_size"],
    p["d_model"],
    p["num_heads"],
    p["num_layers"],
    p["d_ff"],
    p["max_seq_length"],
    p["dropout"],
)
model.train()

# Generate random sample data
torch.manual_seed(1234)
source_data = torch.randint(
    # low, high
    1,
    p["src_vocab_size"],
    # (batch_size, seq_length)
    (64, p["max_seq_length"]),
)

target_data = torch.randint(
    # low, high
    1,
    p["src_vocab_size"],
    # (batch_size, seq_length)
    (64, p["max_seq_length"]),
)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(
    model.parameters(),
    lr=p["learning_rate"],
    betas=p["learning_betas"],
    eps=p["learning_epsilon"],
)


def train_step(epoch):
    optimizer.zero_grad()
    output = model(source_data, target_data[:, :-1])
    loss = criterion(
        output.contiguous().view(-1, p["tgt_vocab_size"]),
        target_data[:, 1:].contiguous().view(-1),
    )
    loss.backward()
    optimizer.step()
    return loss.item()


manager = TrainerManager("transformer", model, p)
manager.train(train_step, 100)
