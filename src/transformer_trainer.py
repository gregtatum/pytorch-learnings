#!/usr/bin/env python3
from transformer import Transformer
from utils import naive_hash
import torch
from torch import nn, optim, Tensor

"""
https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
"""

hyper_parameters = {
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
param_hash = naive_hash(hyper_parameters)

transformer = Transformer(
    hyper_parameters["src_vocab_size"],
    hyper_parameters["tgt_vocab_size"],
    hyper_parameters["d_model"],
    hyper_parameters["num_heads"],
    hyper_parameters["num_layers"],
    hyper_parameters["d_ff"],
    hyper_parameters["max_seq_length"],
    hyper_parameters["dropout"],
)

# Generate random sample data
src_data = torch.randint(
    1, hyper_parameters["src_vocab_size"], (64, hyper_parameters["max_seq_length"])
)  # (batch_size, seq_length)

tgt_data = torch.randint(
    1, hyper_parameters["tgt_vocab_size"], (64, hyper_parameters["max_seq_length"])
)  # (batch_size, seq_length)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(
    transformer.parameters(),
    lr=hyper_parameters["learning_rate"],
    betas=hyper_parameters["learning_betas"],
    eps=hyper_parameters["learning_epsilon"],
)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(
        output.contiguous().view(-1, hyper_parameters["tgt_vocab_size"]),
        tgt_data[:, 1:].contiguous().view(-1),
    )
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
