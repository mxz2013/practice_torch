"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/03_linear.ipynb)

# 🟡 Medium: Simple Linear Layer

Implement a fully-connected linear layer: **y = xW^T + b**

### Signature
```python
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

### Requirements
- `self.weight`: shape `(out_features, in_features)`, init with `randn * (1/√in_features)`
- `self.bias`: shape `(out_features,)`, init as zeros
- Both must have `requires_grad=True`
- `forward(x)` computes `x @ W^T + b`
- Do **NOT** use `torch.nn.Linear
"""

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE


class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        # pass  # Initialize weight and bias
        # self.weight = torch.randn(
        #     out_features, in_features, requires_grad=True
        # ) * math.sqrt(1.0 / in_features) # doing multiplication
        # The * scalar operation creates a new non-leaf tensor —
        # it has a grad_fn (the multiply op). PyTorch only stores .grad on leaf tensors,
        # so weight.grad stays None after backward.
        self.weight = torch.randn(out_features, in_features) * math.sqrt(
            1.0 / in_features
        )
        self.weight.requires_grad_(True)
        # Here, since no requires_grad tensor was involved in the scaling, the result has no grad_fn and is a proper leaf. Then .requires_grad_(True) marks it for gradient accumulation.
        self.bias = torch.zeros(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ torch.transpose(self.weight, dim0=0, dim1=1) + self.bias
        # pass  # Compute y = x @ W^T + b


if __name__ == "__main__":
    # 🧪 Debug
    layer = SimpleLinear(8, 4)
    print("W shape:", layer.weight.shape)  # should be (4, 8)
    print("b shape:", layer.bias.shape)  # should be (4,)

    x = torch.randn(2, 8)
    y = layer.forward(x)
    print("Output shape:", y.shape)  # should be (2, 4)
    # ✅ SUBMIT
    from torch_judge import check

    check("linear")
