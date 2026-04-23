"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/04_layernorm.ipynb)

# 🟡 Medium: Implement LayerNorm

Implement **Layer Normalization** from scratch.

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu$ and $\sigma^2$ are computed over the **last dimension**.

### Signature
```python
def my_layer_norm(
    x: torch.Tensor,      # input
    gamma: torch.Tensor,   # scale (same size as last dim)
    beta: torch.Tensor,    # shift (same size as last dim)
    eps: float = 1e-5
) -> torch.Tensor:
    ...
```

### Rules
- Do **NOT** use `F.layer_norm` or `torch.nn.LayerNorm`
- Normalize over the last dimension only
- Must support autograd

"""

import torch

# ✏️ YOUR IMPLEMENTATION HERE


def my_layer_norm(x, gamma, beta, eps=1e-5):
    """perform a layer norm

    Args:
        x (_type_): input Tensor
        gamma (_type_): scale, same size as last dim
        beta (_type_): shift, same size as last dim
        eps (_type_, optional): avoid dividing by 0. Defaults to 1e-5.
    """
    mu = x.mean(dim=-1, keepdim=True)  # x.shape , normalize the last dim
    var = x.var(dim=-1, keepdim=True, correction=0)  # x.shape, variance=std**2
    # check torch.var documentaiton about correction https://docs.pytorch.org/docs/stable/generated/torch.var.html

    return gamma * ((x - mu) / (torch.sqrt(var + eps))) + beta
    # pass  # Replace this


if __name__ == "__main__":
    # 🧪 Debug
    x = torch.randn(2, 8)
    gamma = torch.ones(8)
    beta = torch.zeros(8)

    out = my_layer_norm(x, gamma, beta)
    ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)

    print("Your output mean:", out.mean(dim=-1))  # should be ~0
    print("Your output std: ", out.std(dim=-1))  # should be ~1
    print("Match ref?      ", torch.allclose(out, ref, atol=1e-4))
    # ✅ SUBMIT
    from torch_judge import check

    check("layernorm")
