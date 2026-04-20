"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/19_gelu.ipynb)

# 🟢 Easy: GELU Activation

Implement the **GELU** (Gaussian Error Linear Unit) activation.

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot 0.5 \cdot (1 + \text{erf}(x / \sqrt{2}))$$

### Signature
```python
def my_gelu(x: Tensor) -> Tensor: ...
```

### Rules
- Do NOT use `F.gelu`, `nn.GELU`, or `torch.nn.functional.gelu`
- Use `torch.erf` for the exact version
"""

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE


def my_gelu(x):
    return 0.5 * x * (1 + torch.erf(x / (math.sqrt(2))))


if __name__ == "__main__":
    # 🧪 Debug
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("Output:", my_gelu(x))
    print("Ref:   ", torch.nn.functional.gelu(x))
    # ✅ SUBMIT
    from torch_judge import check

    check("gelu")
