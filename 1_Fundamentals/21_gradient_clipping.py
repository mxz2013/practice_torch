"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/21_gradient_clipping.ipynb)

# 🟢 Easy: Gradient Norm Clipping

Implement **gradient norm clipping** — a training stability technique.

### Signature
```python
def clip_grad_norm(parameters, max_norm: float) -> float:
    # Clip gradients in-place so total norm <= max_norm
    # Returns the original (unclipped) total norm
```

### Algorithm
1. Compute total norm: `sqrt(sum(p.grad.norm()^2 for p in parameters))`
2. If total > max_norm: scale all grads by `max_norm / total`
3. Return original total norm

### Theory behand
What is a Gradient Norm?
During backprop, each parameter p gets a gradient p.grad — a tensor of partial derivatives. The gradient norm is a single scalar measuring the "total size" of all gradients across all parameters:


total_norm = sqrt( sum of (||p.grad||²) for all p )
It's the L2 norm treating every gradient element as a component of one big vector.

Why Clip It?
In deep networks (especially RNNs/Transformers), gradients can explode — grow exponentially through layers. A single huge gradient update can destroy learned weights entirely. Clipping keeps the update bounded:

If total_norm <= max_norm: do nothing
If total_norm > max_norm: scale all gradients down uniformly by max_norm / total_norm
This preserves the direction of the gradient but caps its magnitude.
"""

import torch
# ✏️ YOUR IMPLEMENTATION HERE


def clip_grad_norm(parameters, max_norm):
    """clip the gradients according to the max norm

    Args:
        parameters (_type_): a list of all the learnable tensors in the model, one per layer's weight (and biases)
                            e.g., for a 2-layer network
                            parameters = [
    layer1.weight,   # shape [256, 512],  grad shape [256, 512]
    layer1.bias,     # shape [256],       grad shape [256]
    layer2.weight,   # shape [10, 256],   grad shape [10, 256]
    layer2.bias,     # shape [10],        grad shape [10]
     ]
        max_norm (_type_): a float representing the maximum norm

    Returns:
        _type_: _description_
    """
    total_norm = torch.sqrt(
        torch.sum(
            torch.tensor([p.grad.norm() ** 2 for p in parameters if p.grad is not None])
        )
    )

    #  p.grad.norm() computes the L2 norm (Frobenius norm) of the gradient tensor
    # treating all elements as a flat vector:
    # ||p.grad|| = sqrt( sum of g²  for every element g in p.grad )
    # Example: if p.grad = [3.0, 4.0], then p.grad.norm() = sqrt(9 + 16) = 5.0
    # For a 2D weight matrix, it flattens all rows/columns into one vector first,
    # then computes that single scalar.
    if total_norm > max_norm:
        scale_factor = max_norm / total_norm  # so it is smaller than 1
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale_factor)

    return total_norm


if __name__ == "__main__":
    # 🧪 Debug
    p = torch.randn(100, requires_grad=True)
    (p * 10).sum().backward()
    orig = clip_grad_norm([p], max_norm=1.0)
    if p.grad is not None:
        print("Before:", p.grad.norm().item())
        print("After: ", p.grad.norm().item())
    print("Original norm:", orig)

    # ✅ SUBMIT
    from torch_judge import check

    check("gradient_clipping")
