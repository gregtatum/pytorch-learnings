#!/usr/bin/env python3

"""
This script demonstrates the training and evaluation of a neural network model
using the FashionMNIST dataset. It defines a simple neural network architecture,
trains the model on the training dataset, and evaluates its performance on the test
dataset.

It is based off of: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_dataloaders():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return (train_dataloader, test_dataloader)


class NeuralNetwork(nn.Module):
    """
    To define a neural network in PyTorch, we create a class that inherits from nn.Module.
    We define the layers of the network in the __init__ function and specify how data will
    pass through the network in the forward function. To accelerate operations in the
    neural network, we move it to the GPU or MPS if available.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(
    dataloader: DataLoader,
    model: nn.Module,
    device: str,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
):
    """
    In a single training loop, the model makes predictions on the training dataset
    (fed to it in batches), and backpropagates the prediction error to adjust the model’s
    parameters.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        prediction_error = model(X)
        loss = loss_fn(prediction_error, y)

        # Backpropagation
        loss.backward()
        optimizer.step()  # Perform a single optimization step.
        optimizer.zero_grad()  # Reset the gradients to zero.

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader: DataLoader, model: nn.Module, device: str, loss_fn: nn.Module):
    """
    Check the model performance against the
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    print(f"Test size: {size}")
    print(f"Test number of batches: {size}")

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def setup_model(device: str):
    model = NeuralNetwork().to(device)
    # Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    # Shape of y: torch.Size([64]) torch.int64

    # To train a model, we need a loss function and an optimizer.

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1 / 1000)
    return (model, loss_fn, optimizer)


def run_training():
    print('Using "mps" device, which is backed by Metal on macOS.')
    device = "mps"

    (model, loss_fn, optimizer) = setup_model(device)
    (train_dataloader, test_dataloader) = get_dataloaders()

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, device, loss_fn, optimizer)
        test(test_dataloader, model, device, loss_fn)
    print("Done!")


run_training()
