"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/07_batchnorm.ipynb)

# 🟡 Medium: Implement BatchNorm

Implement **Batch Normalization** with both **training** and **inference** behavior.

In training mode, use **batch statistics** and update running estimates:

$$\text{BN}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

where $\mu_B$ and $\sigma_B^2$ are the mean and variance computed **across the batch** (dim=0).

In inference mode, use the provided **running mean/var** instead of current batch stats.

### Signature
```python
def my_batch_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    momentum: float = 0.1,
    training: bool = True,
) -> torch.Tensor:
    # x: (N, D) — normalize each feature across all samples in the batch
    # running_mean, running_var: updated in-place during training; used as-is during inference
```

### Rules
- Do **NOT** use `F.batch_norm`, `nn.BatchNorm1d`, etc.
- Compute batch mean and variance over `dim=0` with `unbiased=False`
- Update running stats like PyTorch: `running = (1 - momentum) * running + momentum * batch_stat`
- Use `running_mean` / `running_var` for inference when `training=False`
- Must support autograd w.r.t. `x`, `gamma`, `beta`（running statistics 应视作 buffer，而不是需要梯度的参数）

"""

import torch
# ✏️ YOUR IMPLEMENTATION HERE


def my_batch_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    momentum: float = 0.1,
    training: bool = True,
) -> torch.Tensor:

    if training:
        # batch_mean = x.mean(dim=0, keepdim=True)  # batch norm for dim=0
        batch_mean = x.mean(dim=0)  # batch norm for dim=0
        batch_var = x.var(
            dim=0, correction=0
        )  # sigma**2, unbiased=False -> correction=0
        # Why no keepdim=True here?
        # In LayerNorm you normalize over the last dim and the mean has shape (N, 1)
        # you need keepdim=True so it broadcasts back against (N, D).
        # In BatchNorm you normalize over dim=0, so batch_mean has shape (D,).
        # Broadcasting in PyTorch aligns from the right, so (D,)
        # already broadcasts correctly against (N, D) — no keepdim needed.
        # x:          (N, D)
        # batch_mean:    (D,)   ← aligns from right, broadcasts over N automatical

        # running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        # running_var = (1 - momentum) * running_var + momentum * batch_var
        # Update running statistics in-place. Detach to avoid tracking gradients.
        running_mean.mul_(1 - momentum).add_(momentum * batch_mean.detach())
        running_var.mul_(1 - momentum).add_(momentum * batch_var.detach())
        # mul_ and add_ are in-place ops on the original tensor,
        # which is already tracked in the autograd graph.
        # Modifying it in-place without detaching confuses the graph
        # (you'd be mutating a tensor that has a grad_fn).
        # running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        # creates a new tensor and rebinds the local name —
        # it doesn't touch the original buffer, so autograd isn't disturbed.
        # But crucially, this also means the caller's running_mean is not updated —
        # it's a local reassignment, not an in-place update. That's the bug with = here.
        # So mul_ / add_ is the right approach — just need .detach() to stop gradients flowing into the buffer update

        mean, var = batch_mean, batch_var
    else:
        # mean, var = running_mean.unsqueeze(dim=0), running_var.unsqueeze(dim=0)
        mean, var = running_mean, running_var

    x_bn = gamma * ((x - mean) / (torch.sqrt(var + eps))) + beta
    return x_bn
    # pass  # Replace this


if __name__ == "__main__":
    # 🧪 Debug
    x = torch.randn(8, 4)
    gamma = torch.ones(4)
    beta = torch.zeros(4)

    # Running stats typically live on the same device and shape as features
    running_mean = torch.zeros(4)
    running_var = torch.ones(4)

    # Training mode: uses batch stats and updates running_mean / running_var
    out_train = my_batch_norm(x, gamma, beta, running_mean, running_var, training=True)
    print("[Train] Output shape:", out_train.shape)
    print("[Train] Column means:", out_train.mean(dim=0))  # should be ~0
    print("[Train] Column stds: ", out_train.std(dim=0))  # should be ~1
    print("Updated running_mean:", running_mean)
    print("Updated running_var:", running_var)

    # Inference mode: uses running_mean / running_var only
    out_eval = my_batch_norm(x, gamma, beta, running_mean, running_var, training=False)
    print("[Eval] Output shape:", out_eval.shape)

    # ✅ SUBMIT
    from torch_judge import check

    check("batchnorm")
