import pytest
import torch

# fmt: off

def test_tensor_size():
    x = torch.tensor([
        [0, 1, 2],
        [3, 4, 5]
    ])
    assert x.size() == torch.Size([2, 3])


def test_tensor_view():
    x = torch.tensor([
        [0, 1, 2],
        [3, 4, 5]
    ])
    y = x.view(3, 2)
    assert y.size() == torch.Size([3, 2])
    y_expected = torch.tensor([
        [0, 1],
        [2, 3],
        [4, 5],
    ])
    assert torch.allclose(y, y_expected)

    assert x.view(torch.Size([3, 2])).size() == torch.Size([3, 2])


def test_tensor_transpose():
    x = torch.tensor([
        [0, 1, 2],
        [3, 4, 5]
    ])
    assert torch.allclose(
        x.transpose(0, 1),
        torch.tensor([
            [0, 3],
            [1, 4],
            [2, 5],
        ])
    )
    assert torch.allclose(
        x.transpose(1, 0),
        torch.tensor([
            [0, 3],
            [1, 4],
            [2, 5],
        ])
    )

def test_arange():
    assert torch.allclose(
        torch.tensor([2, 4, 6]),
        torch.arange(2, 8, 2),
    )

    assert torch.allclose(
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        torch.arange(0, 10, 1),
    )

def test_zeros():
    assert torch.allclose(
        torch.zeros(3),
        torch.tensor([0., 0., 0.])
    )

    assert torch.allclose(
        torch.zeros(2, 3),
        torch.tensor([
            [0., 0., 0.],
            [0., 0., 0.]
        ])
    )

def test_unsqueeze():
    x = torch.tensor([1, 2, 3, 4])
    assert torch.allclose(
        x.unsqueeze(0),
        torch.tensor([1, 2, 3, 4])
    )
    assert torch.allclose(
        x.unsqueeze(1),
        torch.tensor([
            [1],
            [2],
            [3],
            [4]
        ])
    )

def test_slicing():
    assert torch.allclose(
        torch.tensor([0, 1, 2, 3, 4, 5, 6])[:],
        torch.tensor([0, 1, 2, 3, 4, 5, 6])
    )
    assert torch.allclose(
        torch.tensor([0, 1, 2, 3, 4, 5, 6])[0:4],
        torch.tensor([0, 1, 2, 3])
    )
    assert torch.allclose(
        torch.tensor([0, 1, 2, 3, 4, 5, 6])[0::2],
        torch.tensor([0, 2, 4, 6])
    )
    assert torch.allclose(
        torch.tensor([0, 1, 2, 3, 4, 5, 6])[0::3],
        torch.tensor([0, 3, 6])
    )
    assert torch.allclose(
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])[:, 0:2],
        torch.tensor([[0, 1], [4, 5], [8, 9]])
    )

    x = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    x[0::2] = 0
    assert torch.allclose(
        x,
        torch.tensor([0, 1, 0, 3, 0, 5, 0])
    )

# fmt: on
