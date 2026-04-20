"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/20_weight_init.ipynb)

# 🟢 Easy: Kaiming Initialization

Implement **Kaiming (He) normal initialization** for weight tensors.

$$W \sim \mathcal{N}(0, \text{std}^2) \quad \text{where} \quad \text{std} = \sqrt{\frac{2}{\text{fan\_in}}}$$

### Signature
```python
def kaiming_init(weight: Tensor) -> Tensor:
    # Initialize weight in-place with Kaiming normal
    # fan_in = weight.shape[1]
    # Returns the weight tensor
"""

import torch
import math
# ✏️ YOUR IMPLEMENTATION HERE


def kaiming_init(weight):
    """For a weight matrix W of shape [fan_out, fan_in]:
    Kaiming Normal: each element W[i][j] ~ N(0, sqrt(2/fan_in))
        Args:
            weight (_type_): tensor with shape [fan_out, fan_in]
    """
    fan_in = weight.shape[1]  # the size of the previous layer
    std = math.sqrt(2 / fan_in)
    weight.normal_(0, std)
    #  Tensor.normal_(mean=0, std=1, *, generator=None) → Tensor
    #  Fills self tensor with elements samples from the normal distribution parameterized by mean and std.
    return weight


if __name__ == "__main__":
    # 🧪 Debug
    import math

    w = torch.empty(256, 512)
    kaiming_init(w)
    print(f"Mean: {w.mean():.4f} (expect ~0)")
    print(f"Std:  {w.std():.4f} (expect {math.sqrt(2 / 512):.4f})")

    # ✅ SUBMIT
    from torch_judge import check

    check("weight_init")
