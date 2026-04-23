"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/08_rmsnorm.ipynb)

# 🟡 Medium: Implement RMSNorm

Implement **Root Mean Square Layer Normalization** — the normalization used in LLaMA, Gemma, etc.

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot w, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$$
where d is the last dim of x

### Signature
```python
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Normalize over the last dimension. No mean subtraction (unlike LayerNorm).
```

### Rules
- Do **NOT** use any built-in norm layers
- Normalize over `dim=-1`
- Must support autograd
"""

import torch
# ✏️ YOUR IMPLEMENTATION HERE


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """_summary_

    Args:
        x (torch.Tensor): tensor with shape N, D
        weight (torch.Tensor): learnable parameter tensor with shape D,
        eps (float, optional): _description_. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    rms_x = torch.sqrt(
        torch.mean(x.pow(2), dim=-1, keepdim=True) + eps
    )  # when we normalize dim=-1, always keepdim

    rms_norm_x = x * weight / rms_x
    return rms_norm_x


if __name__ == "__main__":
    # 🧪 Debug
    x = torch.randn(2, 8)
    w = torch.ones(8)
    out = rms_norm(x, w)
    print("Output shape:", out.shape)
    print("RMS of output:", out.pow(2).mean(dim=-1).sqrt())  # should be ~1
    from torch_judge import check

    check("rmsnorm")
