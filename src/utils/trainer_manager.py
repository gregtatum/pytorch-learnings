import signal
from typing import List

from torch import nn
from os import path, mkdir
from utils import naive_hash, save_json
from imgcat import imgcat
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import sys
import time
import json

data_path = path.abspath(path.join(path.dirname(__file__), "../../data"))


class TrainerManager:
    """
    Manage the training, and output the model.
    """

    epoch = 0
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
        print("\nFinishing current epoch, then exiting.")
        self.sigint_sent = True

    def load_saved_model(self):
        if path.exists(self.model_path):
            print("Loading a saved model")
            self.model.load_state_dict(torch.load(self.model_path))

        if path.exists(self.loss_path):
            print("Loading loss history")
            with open(self.loss_path) as f:
                self.losses = json.load(f)
                self.epoch = len(self.losses)

    def graph_loss(self):
        figure, axes = plt.subplots()

        axes.set_title(f'"{self.name}" Training')
        axes.plot(torch.arange(0, len(self.losses), 1), self.losses)
        axes.set_xlabel("Epoch")
        axes.set_ylabel("Loss")
        plt.savefig(self.graph_path, dpi=150)

    def gracefully_exit(self):
        imgcat(open(self.graph_path, "r"))
        sys.exit(0)

    def save_hyperparameters(self):
        if not path.exists(self.parameters_path):
            save_json(self.parameters_path, self.hyper_parameters)

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            self.model_path,
        )
        save_json(self.loss_path, self.losses)

    def train(self, train_step, batch_size=0, num_epochs=0):
        if batch_size == 0:
            raise Exception("A batch_size must be provided.")
        if num_epochs == 0:
            raise Exception("A num_epochs must be provided.")

        self.save_hyperparameters()
        self.load_saved_model()

        print("Beginning training")
        print(f" Hyperparameters: {self.parameters_path}")
        print(f" Loss plot: {self.graph_path}")
        print(f" Loss data: {self.loss_path}")
        print(f" Model: {self.model_path}")

        while self.epoch < num_epochs:
            if self.sigint_sent:
                self.gracefully_exit()
            start_time = time.time()

            print(f"Running epoch {self.epoch} of {num_epochs}")

            loss = train_step(self.epoch)
            self.losses.append(loss)

            self.epoch += 1
            self.graph_loss()
            self.save_model()

            elapsed_time = time.time() - start_time
            per_item_time = 1000.0 * elapsed_time / batch_size
            print(f"  {elapsed_time:.2f} seconds elapsed for {batch_size} items")
            print(f"  {per_item_time:.0f} ms per item ")
            print(f"  {loss:.2f} loss")

        print("Training complete 🎉")
        imgcat(open(self.graph_path, "r"))
