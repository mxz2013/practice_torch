"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/40_linear_regression.ipynb)

# 🟡 Medium: Linear Regression

Implement **linear regression** using three different approaches — all in pure PyTorch.

Given data `X` of shape `(N, D)` and targets `y` of shape `(N,)`, find weight `w` of shape `(D,)` and bias `b` (scalar) such that:

$$\hat{y} = Xw + b$$

### Signature
```python
class LinearRegression:
    def closed_form(self, X: Tensor, y: Tensor) -> tuple[Tensor, Tensor]: ...
    def gradient_descent(self, X: Tensor, y: Tensor, lr=0.01, steps=1000) -> tuple[Tensor, Tensor]: ...
    def nn_linear(self, X: Tensor, y: Tensor, lr=0.01, steps=1000) -> tuple[Tensor, Tensor]: ...
```

All methods return `(w, b)` where `w` has shape `(D,)` and `b` has shape `()`.

### Method 1 — Closed-Form (Normal Equation)
Theory behind:
θ = [b, w₁, w₂, ..., w_D] -> (D+1)
X_aug = [ 1 | X ]   shape: (N, D+1)

ŷ = X_aug @ θ
  = 1·b + x₁·w₁ + x₂·w₂ + ...
  = Xw + b   ✓  (same thing!)

If we use L2 loss:
L(θ) = ||y - X_aug θ||²

This is quadratic in θ. Expanding it gives a polynomial of degree 2 in each component of θ. That means:

The gradient ∂L/∂θ is linear in θ
Setting it to zero gives a linear system: X^T X θ = X^T y
A linear system has an exact algebraic solution → the Normal Equation

Augment X with a ones column, then solve:

$$\theta = (X_{aug}^T X_{aug})^{-1} X_{aug}^T y$$

Or use `torch.linalg.lstsq` / `torch.linalg.solve`.
torch.linalg.solve() computes A.inv() @ B with a numerically stable algorithm.

### Method 2 — Gradient Descent from Scratch
Initialize `w` and `b` to zeros. Repeat for `steps` iterations:
```
pred = X @ w + b
error = pred - y
grad_w = (2/N) * X^T @ error
grad_b = (2/N) * error.sum()
w -= lr * grad_w
b -= lr * grad_b
```

### Method 3 — PyTorch nn.Linear
Create `nn.Linear(D, 1)`, use `nn.MSELoss` and an optimizer (e.g., `torch.optim.SGD`).
After training, extract `w` and `b` from the layer.

### Rules
- All inputs and outputs must be **PyTorch tensors**
- Do **NOT** use numpy or sklearn
- `closed_form` must not use iterative optimization
- `gradient_descent` must manually compute gradients (no `autograd`)
- `nn_linear` should use `torch.nn.Linear` and `loss.backward()`
"""

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE


class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        """Normal equation: w = (X^T X)^{-1} X^T y"""
        # X shape N D, y (N,)
        ones_col = torch.ones(X.shape[0], 1)  # shape N, 1
        X_aug = torch.cat([ones_col, X], dim=1)  # dim=1 so x_aug N, D+1
        # use torch.transpose for >2D
        A = torch.t(X_aug) @ X_aug  # D+1,N @ N, D+1 -> D+1, D+1
        B = torch.t(X_aug)  # D+1, N
        # A^-1 @ B @ y -> D+1, N @ N = D+1 ->  D+1, D+1 @ D+1, 1 -> D+1,1
        theta = torch.linalg.solve(A, B) @ y  # shape (D+1,)
        b = theta[0]
        w = theta[1:]
        return (w, b)

        # pass  # Return (w, b)

    def gradient_descent(
        self, X: torch.Tensor, y: torch.Tensor, lr: float = 0.01, steps: int = 1000
    ):
        """Manual gradient descent loop"""
        N, D = X.shape
        # initialize w and b
        w = torch.zeros(D)
        b = 0.0
        for _ in range(steps):
            pred = X @ w + b
            error = pred - y
            grad_w = (2.0 / N) * torch.t(X) @ error
            grad_b = 2.0 / N * error.sum()
            w -= lr * grad_w
            b -= lr * grad_b
        return (w, b)
        # pass  # Return (w, b)

    def nn_linear(
        self, X: torch.Tensor, y: torch.Tensor, lr: float = 0.01, steps: int = 1000
    ):
        """Train nn.Linear with autograd"""
        N, D = X.shape
        layer = nn.Linear(D, 1)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            pred = layer(X).squeeze(-1)  # (N,)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        w = layer.weight.data.squeeze(0)  # (D,)
        b = layer.bias.data.squeeze(0)  # scalar
        return (w, b)


if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(42)
    X = torch.randn(100, 3)
    true_w = torch.tensor([2.0, -1.0, 0.5])
    y = X @ true_w + 3.0

    model = LinearRegression()

    w_cf, b_cf = model.closed_form(X, y)
    print(f"Closed-form:  w={w_cf}, b={b_cf.item():.4f}")

    w_gd, b_gd = model.gradient_descent(X, y, lr=0.05, steps=2000)
    print(f"Grad descent: w={w_gd}, b={b_gd.item():.4f}")

    w_nn, b_nn = model.nn_linear(X, y, lr=0.05, steps=2000)
    print(f"nn.Linear:    w={w_nn}, b={b_nn.item():.4f}")

    print(f"\nTrue:         w={true_w}, b=3.0")

    # ✅ SUBMIT
    from torch_judge import check

    check("linear_regression")
