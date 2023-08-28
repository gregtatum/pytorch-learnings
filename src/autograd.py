#!/usr/bin/env python3

"""
This is an annotated version of the code in:
https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
"""

from typing import Sequence, cast
import torch
from utils import output_plot

import matplotlib.pyplot as plt
import math

# Create a linear vector with 25 elements, ranged from [0, 2π].
# `requires_grad` ensures it tracks the gradient.
a = torch.linspace(0.0, 2.0 * math.pi, steps=25, requires_grad=True)

print("a", a)
# tensor([0.0000, 0.2618, 0.5236, 0.7854, 1.0472, 1.3090, 1.5708, 1.8326, 2.0944,
#         2.3562, 2.6180, 2.8798, 3.1416, 3.4034, 3.6652, 3.9270, 4.1888, 4.4506,
#         4.7124, 4.9742, 5.2360, 5.4978, 5.7596, 6.0214, 6.2832],
#        requires_grad=True)

b = torch.sin(a)

a_plot = cast(Sequence[float], a.detach())
b_plot = cast(Sequence[float], b.detach())
plt.plot(a_plot, b_plot)

output_plot("./plots/autograd-01.png")

print("b", b)
# The gradient is stored in the vector with the SinBackward.
# tensor([ ... ], grad_fn=<SinBackward0>)

# Do more computations, and note the gradients.

c = 2 * b
print("c", c)
# tensor([ ... ], grad_fn=<MulBackward0>)

d = c + 1
print("d", d)
# tensor([ ... ], grad_fn=<AddBackward0>)

# Compute a fake loss value, which is a single element.
out = d.sum()
print("out", out)
# tensor(25., grad_fn=<SumBackward0>)

# Run the gradient function to compute the gradients in all the functions.
out.backward()
print("a.grad", a.grad)

if a.grad:
    a_plot = cast(Sequence[float], a.detach())
    b_plot = cast(Sequence[float], a.grad.detach())
    plt.plot(a_plot, b_plot)

output_plot("./plots/autograd-02.png")
