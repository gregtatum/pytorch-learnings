# Pytorch Learning

This repo contains my learnings for pytorch.

## Examples

The bare minimum hello world for pytorch.

```
poetry run src/pytorch-hello.py
```

Train a simple feed forward neural network for fashionMNIST.

```
poetry run src/fashion-mnist.py
```

## Transformers

[Discovering the Transformer paper](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634)

| Formula | Description |
| ------- | ----------- |
| $Q$ | The Query |
| $K$ | The Key |
| $V$ | The Value |
| $\text{Attention}(Q, K, V) = \text{softmax}({\Large QK^T \over \sqrt{d_k}})V$ | The self attention function, known as scaled dot-product attention |
| $x^{(i)}$ | An embedded input |
| $W_q, W_k, W_v$ | Weight queries, used to train the attention. |
| $T$ | The number of embedded input vectors |
| $d$ | The length of a single word embedding vector |
| $q_i = W_qx \text{ for } i \in [1, T]$ | | Query sequence |
| $k_i = W_kx \text{ for } i \in [1, T]$ | | Key sequence |
| $v_i = W_vx \text{ for } i \in [1, T]$ | | Value sequence |
| $q^{(i)}$ and $k^{(i)}$ | Vectors of dimension $d_k$ |

$$
\mathbf{QK}^T =

\begin{bmatrix}

e_{11} & e_{12} & \dots & e_{1n} \\

e_{21} & e_{22} & \dots & e_{2n} \\

\vdots & \vdots & \ddots & \vdots \\

e_{m1} & e_{m2} & \dots & e_{mn} \\

\end{bmatrix}
$$

$$
\frac{\mathbf{QK}^T}{\sqrt{d_k}} =

\begin{bmatrix}

\tfrac{e_{11}}{\sqrt{d_k}} & \tfrac{e_{12}}{\sqrt{d_k}} & \dots & \tfrac{e_{1n}}{\sqrt{d_k}} \\

\tfrac{e_{21}}{\sqrt{d_k}} & \tfrac{e_{22}}{\sqrt{d_k}} & \dots & \tfrac{e_{2n}}{\sqrt{d_k}} \\

\vdots & \vdots & \ddots & \vdots \\

\tfrac{e_{m1}}{\sqrt{d_k}} & \tfrac{e_{m2}}{\sqrt{d_k}} & \dots & \tfrac{e_{mn}}{\sqrt{d_k}} \\

\end{bmatrix}
$$

$$
\text{softmax} \left( \frac{\mathbf{QK}^T}{\sqrt{d_k}} \right) =

\begin{bmatrix}

\text{softmax} ( \tfrac{e_{11}}{\sqrt{d_k}} & \tfrac{e_{12}}{\sqrt{d_k}} & \dots & \tfrac{e_{1n}}{\sqrt{d_k}} ) \\

\text{softmax} ( \tfrac{e_{21}}{\sqrt{d_k}} & \tfrac{e_{22}}{\sqrt{d_k}} & \dots & \tfrac{e_{2n}}{\sqrt{d_k}} ) \\

\vdots & \vdots & \ddots & \vdots \\

\text{softmax} ( \tfrac{e_{m1}}{\sqrt{d_k}} & \tfrac{e_{m2}}{\sqrt{d_k}} & \dots & \tfrac{e_{mn}}{\sqrt{d_k}} ) \\

\end{bmatrix}
$$

$$

$$
