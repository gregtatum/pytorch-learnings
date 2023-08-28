import signal
from typing import Any, List, cast

from torch import nn
from os import path, mkdir
from utils import naive_hash, save_json
from imgcat import imgcat
import matplotlib
import matplotlib.pyplot as plt
import torch
import sys
import time
import json
import math

data_path = path.abspath(path.join(path.dirname(__file__), "../../data"))


class TrainerManager:
    """
    Manage the training, and output the model.
    """

    epoch = 0
    batch = 0
    sigint_sent = False
    name: str
    losses: List[float] = []

    def __init__(self, name: str, model: nn.Module, hyper_parameters: object):
        signal.signal(signal.SIGINT, self.handle_signal)
        self.name: str = name
        self.hyper_parameters: object = hyper_parameters
        self.param_hash: str = naive_hash(hyper_parameters)
        self.model: nn.Module = model

        self.parameters_path: str = self.artifact_path("hyperparameters.json")
        self.loss_path: str = self.artifact_path("loss.json")
        self.model_path: str = self.artifact_path("model.pt")
        self.graph_path: str = self.artifact_path("graph.png")

        if not path.exists(data_path):
            mkdir(data_path)

        output_dir = path.join(data_path, self.name)
        if not path.exists(output_dir):
            mkdir(output_dir)

    def artifact_path(self, postfix) -> str:
        return path.join(data_path, self.name, f"{self.param_hash}-{postfix}")

    def handle_signal(self, *args):
        print("\nFinishing current batch, then exiting.")
        self.sigint_sent = True

    def load_losses(self, num_batches: int):
        if path.exists(self.loss_path):
            print("Loading loss history")
            with open(self.loss_path) as f:
                self.losses = json.load(f)
                self.epoch = int(len(self.losses) / num_batches)
                self.batch = int(len(self.losses) % num_batches)

    def graph_loss(self, num_batches: int):
        figure, axes = plt.subplots()

        axes.set_title(f'"{self.name}" Training')
        losses_batch = cast(list, torch.arange(0, len(self.losses), 1))
        axes.plot(losses_batch, self.losses)

        for i in range(0, math.ceil(len(self.losses) / num_batches)):
            x = i * num_batches
            cast(Any, axes).plot(x, self.losses[x], "ro")

        axes.set_xlabel("Batch")
        axes.set_ylabel("Loss")
        plt.savefig(self.graph_path, dpi=150)
        plt.close(figure)

    def gracefully_exit(self):
        imgcat(open(self.graph_path, "r"))
        sys.exit(0)

    def save_hyperparameters(self):
        if not path.exists(self.parameters_path):
            save_json(self.parameters_path, self.hyper_parameters)

    def save_losses(self):
        save_json(self.loss_path, self.losses)

    def train(
        self,
        train_step,
        data_size=0,
        batch_size=0,
        num_epochs=0,
        save_model=None,
        load_model=None,
    ):
        if batch_size == 0:
            raise Exception("A batch_size must be provided.")
        if num_epochs == 0:
            raise Exception("A num_epochs must be provided.")
        if data_size == 0:
            raise Exception("A data_size must be provided.")
        if not save_model:
            raise Exception("A save_model must be provided.")
        if not load_model:
            raise Exception("A load_model must be provided.")

        num_batches = math.ceil(data_size / batch_size)

        self.save_hyperparameters()
        load_model(self.artifact_path)
        self.load_losses(num_batches)

        print("Beginning training")
        print(f" Hyperparameters: {self.parameters_path}")
        print(f" Loss plot: {self.graph_path}")
        print(f" Loss data: {self.loss_path}")
        print(f" Model: {self.model_path}")

        while self.epoch < num_epochs:
            # Bail out if a sigint is discovered.
            if self.sigint_sent:
                self.gracefully_exit()
            # Measure the timing.
            start_time = time.time()

            # Compute the data range of the batch. Normally this is the full range, but
            # at the end of the data, this may be a smaller batch.
            start = self.batch * batch_size
            end = start + batch_size
            if self.batch + 1 == num_batches:
                # The last batch may not be the batch size.
                end = start + data_size - start
            data_slice = slice(start, end)

            # Report where the training is in terms of batches, and epochs.
            batch_completeness = self.batch * batch_size / data_size
            epoch_completeness = (self.epoch + batch_completeness) / num_epochs
            print(
                f"Epoch {self.epoch + 1}/{num_epochs} ({epoch_completeness * 100:.2f}%). "
                + f"Batch {self.batch + 1}/{num_batches} ({batch_completeness * 100:.2f}%)"
            )

            # Run
            loss = train_step(data_slice)
            self.losses.append(loss)

            self.graph_loss(num_batches)
            self.save_losses()
            save_model(self.artifact_path)

            elapsed_time = time.time() - start_time
            per_item_time = 1000.0 * elapsed_time / batch_size
            print(f"  {elapsed_time:.2f} seconds elapsed for {batch_size} items")
            print(f"  {per_item_time:.2f} ms per item ")
            print(f"  {loss:.2f} loss")

            # Increment the values.
            self.batch += 1
            if self.batch == num_batches:
                self.epoch += 1
                self.batch = 0

        print("Training complete 🎉")
        imgcat(open(self.graph_path, "r"))
