#!/usr/bin/env python3

"""
Perform a word embedding, run src/sentence_tokenization.py first.

https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentencepiece import SentencePieceProcessor
from datasets import load_dataset
from os import path

torch.manual_seed(1234)

# Hyper parameters
vocab_size = 5000
embedding_dim = 5
learning_rate = 0.001
num_epochs = 10
sentence_sample_size = 1000
learning_rate = 0.001
context_size = 2


def load_tokenizers():
    data_path = path.abspath(path.join(path.dirname(__file__), "../data"))
    model_en = path.join(data_path, "en.model")
    model_es = path.join(data_path, "es.model")

    if not path.exists(model_en):
        raise Exception('No "en" model was found, run sentence_tokenization.py first.')
    if not path.exists(model_es):
        raise Exception('No "es" model was found, run sentence_tokenization.py first.')

    tokens_en = SentencePieceProcessor()
    tokens_es = SentencePieceProcessor()

    tokens_en.load(model_en)
    tokens_es.load(model_es)

    return tokens_en, tokens_es


tokens_en, tokens_es = load_tokenizers()


print("Loading dataset")
dataset = load_dataset("para_crawl", "enes", split="train")


class LanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(LanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        hidden_size = 128
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using cuda")
else:
    device = torch.device("cpu")
    print("Using cpu")

losses = []
criterion = nn.NLLLoss()
model = LanguageModeler(vocab_size, embedding_dim, context_size).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def print_embedding(tokens, text):
    example_ids = tokens.encode_as_ids(text)

    print("Embedding:")
    print(text)
    print("Pieces:", tokens.encode_as_pieces(text))
    print("Ids:", example_ids)
    print("Embedding:", model.embeddings(torch.tensor(example_ids, device=device)))
    print()


print_embedding(tokens_en, "The quick brown fox.")

print("Accessing data")
dataset_slice = dataset["translation"]
dataset_len = dataset.num_rows
print("Data is loaded")

for epoch in range(num_epochs):
    total_loss = 0.0

    print(f"Starting epoch {epoch}.")
    for training_i in range(sentence_sample_size):
        sentence = dataset_slice[
            (epoch * sentence_sample_size + training_i) % dataset_len
        ]
        sentence_ids = tokens_en.encode_as_ids(sentence["en"])

        for i in range(context_size, len(sentence_ids)):
            context = torch.tensor(
                [sentence_ids[i - j - 1] for j in range(context_size)],
                dtype=torch.long,
                device=device,
            )
            target_word = torch.tensor(
                [sentence_ids[i]],
                dtype=torch.long,
                device=device,
            )

            # Reset the gradients on each step.
            model.zero_grad()

            log_probabilities = model(context)

            loss = criterion(log_probabilities, target_word)

            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()

    print("(Loss)", total_loss)
    losses.append(total_loss)

print("Losses", losses)
print_embedding(tokens_en, "The quick brown fox.")
