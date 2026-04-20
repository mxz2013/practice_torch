"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/31_gradient_accumulation.ipynb)

# 🟢 Easy: Gradient Accumulation

Implement a **training step with gradient accumulation** — simulating large batches with limited memory.

### Signature
```python
def accumulated_step(model, optimizer, loss_fn, micro_batches) -> float:
    # micro_batches: list of (input, target) tuples
    # Returns: average loss (float)
```

### Algorithm
1. `optimizer.zero_grad()`
2. For each `(x, y)` in micro_batches: `loss = loss_fn(model(x), y) / len(micro_batches)`, then `loss.backward()`
3. `optimizer.step()`
4. Return total accumulated loss

The key insight: dividing each loss by `n` before backward makes accumulated gradients equal to a single large-batch gradient.
WHY?
The Math
For a large batch of N samples, the loss is the mean across all samples:
L_large = (1/N) * sum(loss_i  for i in 1..N)
The gradient is:
∂L_large/∂W = (1/N) * sum(∂loss_i/∂W  for i in 1..N)

Now if we simulate this N samples using k mini-batches with N/n samples per mini-batch.
L_k, we compute the grad contributation from each mini-batch

L_k = (1/k) * sum(loss_i for i in mini-batch k)

∂L_k/∂W = (1/k) * sum(∂loss_i/∂W  for i in mini-batch) -> mini-batch mean

if we call backward() and accumulate, we end up with

accumulated grad = sum_k[ ∂L_k/∂W ]
                = sum_k[ (1/(N/n)) * sum_{i in batch k}(∂loss_i/∂W) ]
                = (n/N) * sum_k[ sum_{i in batch k}(∂loss_i/∂W) ]
                = (n/N) * sum_i(∂loss_i/∂W)          # all samples combined
                = n * (1/N) * sum_i(∂loss_i/∂W)
                = n * ∂L_large/∂W   ✓

That is why if we divide each L_k by n:
grad contribution from batch k = ∂(L_k/n)/∂W = (1/n) * ∂L_k/∂W
Summing across all n mini-batches:

accumulated grad = sum_k[ (1/n) * ∂L_k/∂W ]
                = (1/n) * sum_k[ (1/(N/n)) * sum_i(∂loss_i/∂W) ]
                = (1/N) * sum_i(∂loss_i/∂W)
                = ∂L_large/∂W  ✓
"""

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE


def accumulated_step(model, optimizer, loss_fn, micro_batches) -> float:
    """

    Args:
        model (_type_): model object
        optimizer (_type_): optimizer
        loss_fn (_type_): loss function
        micro_batches (_type_): list of (input, target) tuples
    """
    total_samples = sum(y.shape[0] for _, y in micro_batches)
    num_micro_batches = len(micro_batches)
    optimizer.zero_grad()
    total_loss = 0
    for x, y in micro_batches:
        batch_size = y.shape[0]
        mean_loss = loss_fn(model(x), y)  # by default the reduction=mean
        scaled_loss = (
            mean_loss / num_micro_batches
        )  # key here to make accumulation valid

        total_loss += batch_size * mean_loss.item()  # loss is the mean of  batch
        scaled_loss.backward()  # accumulateing all gradients

    optimizer.step()

    return total_loss / total_samples

    # pass  # zero_grad, loop (forward, scale loss, backward), step


if __name__ == "__main__":
    # 🧪 Debug
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = accumulated_step(
        model,
        opt,
        nn.MSELoss(),
        [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)],
    )
    print("Loss:", loss)

    # ✅ SUBMIT
    from torch_judge import check

    check("gradient_accumulation")
