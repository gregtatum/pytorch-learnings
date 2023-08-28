import pytest
import torch


def test_einstein_summation() -> None:
    # fmt: off
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    assert torch.isclose(torch.dot(a, b), torch.tensor([32.0]))
    assert torch.isclose(torch.einsum("i,i->", a, b), torch.tensor([32.0]))

    a = torch.tensor([[1.0], [2.0], [3.0]])
    b = torch.tensor([[4.0], [5.0], [6.0]])

    result = torch.einsum("ij,ij->", a, b)
    assert torch.isclose(result, torch.tensor([32.0]))

    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[4.0, 5.0, 6.0]])

    result = torch.einsum("ij,ij->", a, b)
    assert torch.isclose(result, torch.tensor([32.0]))

    a = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    b = torch.tensor([
        [5.0, 6.0],
        [7.0, 8.0]
    ])

    assert torch.allclose(
        torch.einsum("ij,ij->i", a, b),
        torch.tensor([
            torch.dot(a[0], b[0]),
            torch.dot(a[1], b[1]),
        ]),
    )

    assert torch.allclose(
        torch.einsum("ij,ij->ij", a, b),
        torch.multiply(a, b),
    )
    assert torch.allclose(
        torch.einsum("ij,ij->ji", a, b),
        # The .T is the transpose of the matrix.
        torch.multiply(a, b).T,
    )

    # I'm not really visualizing this one, I just tried transposing things until it worked.
    assert torch.allclose(
        torch.einsum("ij,ji->ij", a, b),
        torch.multiply(
            torch.tensor([
                [1.0, 2.0],
                [3.0, 4.0]
            ]),
            torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0]
            ]).T
        ),
    )

    # I'm not really visualizing this one, I just tried transposing things until it worked.
    assert torch.allclose(
        torch.einsum("ij,ji->ji", a, b),
        torch.multiply(
            torch.tensor([
                [1.0, 2.0],
                [3.0, 4.0]
            ]).T,
            torch.tensor([
                [5.0, 6.0],
                [7.0, 8.0]
            ])
        ),
    )

    # The letters are just convention.
    assert torch.allclose(
        torch.einsum("rq,rq->r", a, b),
        torch.tensor([
            torch.dot(a[0], b[0]),
            torch.dot(a[1], b[1]),
        ]),
    )

    # fmt: on
