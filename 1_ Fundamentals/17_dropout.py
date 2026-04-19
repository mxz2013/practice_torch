"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/17_dropout.ipynb)

# 🟢 Easy: Implement Dropout

Implement **Dropout** regularization from scratch.

### Signature
```python
class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5): ...
    def forward(self, x: Tensor) -> Tensor: ...
```

### Rules
- During **training**: zero each element with probability `p`, scale remaining by `1/(1-p)`
- During **eval**: return input unchanged (identity)
- Do NOT use `nn.Dropout` or `F.dropout`
"""

import torch
import torch.nn as nn
# ✏️ YOUR IMPLEMENTATION HERE


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.prob = p

    def forward(self, x):
        # nn.Module already gives you self.training
        if not self.training:
            return x
        # e.g., p = 0.2, keep_prob = 0.8
        keep_prob = 1 - self.prob
        mask = torch.rand_like(
            x
        )  # return a tensor with the same size as x, with random numbers from a uniform distribution on the interval [0,1)
        mask = mask < keep_prob  # all random numbers that < keep_prob will be True
        # pytorch treat True as 1 and False as 0
        return x * mask / keep_prob


if __name__ == "__main__":
    # 🧪 Debug
    d = MyDropout(p=0.5)
    d.train()
    x = torch.ones(10)
    print("Train:", d(x))
    d.eval()
    print("Eval: ", d(x))

    # ✅ SUBMIT
    from torch_judge import check

    check("dropout")
