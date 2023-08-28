#!/usr/bin/env python3

"""
Perform a word embedding, run src/sentence_tokenization.py first.

https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
"""

from typing import Any, Optional, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from sentencepiece import SentencePieceProcessor
from datasets import load_dataset
from os import path, mkdir
import signal
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from imgcat import imgcat
import time
from utils import naive_hash

torch.manual_seed(1234)
data_path = path.abspath(path.join(path.dirname(__file__), "../data"))
language = "en"

# Create the artifact directory.
if not path.exists(path.join(data_path, "embeddings")):
    mkdir(path.join(data_path, "embeddings"))


def save_json(path_str: str, out: Any) -> None:
    with open(path_str, "w") as f:
        f.write(json.dumps(out, indent=2, sort_keys=True))


num_epochs = 10000


# Hyper parameters
class HyperParameters:
    def __init__(self) -> None:
        self.vocab_size = 5000
        self.embedding_dim = 5
        self.sentences_per_epoch = 1000
        self.learning_rate = 0.01
        self.context_size = 2


p = HyperParameters()
param_hash = naive_hash(p)


def load_tokenizers() -> tuple[SentencePieceProcessor, SentencePieceProcessor]:
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


dataset = load_dataset("para_crawl", "enes", split="train")


class LanguageModeler(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int) -> None:
        super(LanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        hidden_size = 128
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: Tensor) -> Tensor:
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using cuda")
else:
    device = torch.device("cpu")
    print("Using cpu")

criterion = nn.NLLLoss()
model = LanguageModeler(
    p.vocab_size,
    p.embedding_dim,
    p.context_size,
).to(device)
optimizer = optim.SGD(model.parameters(), lr=p.learning_rate)


def print_embedding(tokens: SentencePieceProcessor, text: str) -> None:
    example_ids = tokens.encode_as_ids(text)

    print("Example embedding before training:")
    print(text)
    print("Pieces:", tokens.encode_as_pieces(text))
    print("Ids:", example_ids)
    print("Embedding:", model.embeddings(torch.tensor(example_ids, device=device)))
    print()


print_embedding(tokens_en, "The quick brown fox.")


def artifact_path(postfix: str) -> str:
    return path.join(data_path, "embeddings", f"{language}-{param_hash}-{postfix}")


# TODO - Use TrainerManager
class Trainer:
    epoch = 0
    sigint_sent = False
    __data: Optional[dict[str, str]] = None

    parameters_path = artifact_path("hyperparameters.json")
    loss_path = artifact_path("loss.json")
    embedding_path = artifact_path("embedding.pt")
    model_path = artifact_path("model.pt")
    graph_path = artifact_path("graph.png")

    losses: list[float] = []

    def __init__(self) -> None:
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, *args: Any) -> None:
        self.sigint_sent = True

    def save_hyperparameters(self) -> None:
        if not path.exists(self.parameters_path):
            save_json(self.parameters_path, p)

    def load_saved_model(self) -> None:
        if path.exists(self.model_path):
            print("Loading a saved model")
            model.load_state_dict(torch.load(self.model_path))

        if path.exists(self.loss_path):
            print("Loading loss history")
            with open(self.loss_path) as f:
                self.losses = json.load(f)
                self.epoch = len(self.losses)

    def train(self) -> None:
        self.save_hyperparameters()
        self.load_saved_model()

        print("Beginning training")
        print(f" Hyperparameters: {self.parameters_path}")
        print(f" Loss plot: {self.graph_path}")
        print(f" Embedding: {self.embedding_path}")

        while self.epoch < num_epochs:
            print(f"Running epoch {self.epoch} of {num_epochs}")

            self.train_one_epoch()

            self.epoch += 1

        print("Training complete, losses:", self.losses)

    def data(self) -> list[dict[str, str]]:
        """
        Lazily loads the data. This method can take some time to run.
        """
        if not self.__data:
            print("Accessing data")
            self.__data = dataset["translation"]
            print("Data is loaded")

            # with open(path.join(data_path, "en.small.txt")) as f:
            #     self.__data = [{"en": line} for line in f.readlines()]

        return cast(Any, self.__data)

    def save_embeddings(self) -> None:
        torch.save(
            model.state_dict(),
            self.model_path,
        )
        torch.save(
            model.embeddings.state_dict(),
            self.embedding_path,
        )
        save_json(self.loss_path, self.losses)

    def graph_loss(self) -> None:
        figure, axes = plt.subplots()

        axes.set_title("Word Embedding Training")
        data: Any = torch.arange(0, len(self.losses), 1)
        axes.plot(data, self.losses)
        axes.set_xlabel("Epoch")
        axes.set_ylabel("Loss")
        plt.savefig(self.graph_path, dpi=150)
        plt.close(figure)

    def gracefully_exit(self) -> None:
        print("\nRestarting this script will pick up the training where it left off.")
        imgcat(open(self.graph_path))
        sys.exit(0)

    def train_one_epoch(self) -> None:
        total_loss = 0.0

        sentences_per_epoch = p.sentences_per_epoch
        context_size = p.context_size
        data = self.data()
        start_time = time.time()

        for training_i in range(sentences_per_epoch):
            if self.sigint_sent:
                self.gracefully_exit()

            sentence = data[(self.epoch * sentences_per_epoch + training_i) % len(data)]
            sentence_ids = tokens_en.encode_as_ids(sentence[language])

            loss_sentence = 0

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
                loss_sentence += loss.item()

            word_count_training_step = len(sentence_ids) - context_size
            total_loss += loss_sentence / word_count_training_step

        elapsed_time = time.time() - start_time
        print(f" (time) {elapsed_time:.2f} seconds")
        print(" (loss)", total_loss)

        self.losses.append(total_loss)
        self.save_embeddings()
        self.graph_loss()


print("Creating trainer")
trainer = Trainer()
trainer.train()

print_embedding(tokens_en, "The quick brown fox.")
