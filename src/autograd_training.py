#!/usr/bin/env python3

"""
This is an annotated version of the code in:
https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
"""

import torch

BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(100, 10)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


print("Created TinyModel")
model = TinyModel()
some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)
prediction = model(some_input)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


def output_weights():
    print(
        "   Layer 2 weights (slice):", model.layer2.weight[0][0:4]
    )  # Just a small slice
    grad = model.layer2.weight.grad
    print(
        "   Layer 2 gradient (slice):", None if grad is None else grad[0][0:4]
    )  # Expect "None"


output_weights()

loss = (ideal_output - prediction).pow(2).sum()
print(f"The computed loss is: {loss}")

print("Compute the gradient")
loss.backward()
output_weights()

print("Run the optimizer a step.")
optimizer.step()
output_weights()

print("Zero out the weights")
optimizer.zero_grad()
output_weights()
