"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/02_softmax.ipynb)

# 🟢 Easy: Implement Softmax

Implement the **Softmax** function from scratch.

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

### Signature
```python
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ...
```

### Rules
- Do **NOT** use `torch.softmax`, `F.softmax`, or `torch.nn.Softmax`
- Must be **numerically stable** (hint: subtract `max` before `exp`)

### Example
```
Input:  tensor([1., 2., 3.])
Output: tensor([0.0900, 0.2447, 0.6652])  # sums to 1.0
```
"""

import torch


def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """apply softmax in dim==dim
        official doc https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    Args:
        x (torch.Tensor): input tensor
        dim (int, optional): the dimension to apply softmax. Defaults to -1.

    Returns:
        torch.Tensor: a Tensor of the same dimension and shape as the input with values in the range [0,1]
    """
    # 2 requirements, input output the same dim, and carefully handel the large values
    x_max = torch.max(x, dim=dim, keepdim=True).values  # values or indices
    exp_x = torch.exp(x - x_max)
    # we can substract exp(-x_max) because exp(a-b) = exp(a) * exp (-b)
    # thus exp(a-b) / sum_i(exp(a_i -b)) == exp(a)/ sum_i(exp(a_i))

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


if __name__ == "__main__":
    # 🧪 Debug
    x = torch.tensor([1.0, 2.0, 3.0])
    print("Output:", my_softmax(x, dim=-1))
    print("Sum:   ", my_softmax(x, dim=-1).sum())  # should be ~1.0
    print("Ref:   ", torch.softmax(x, dim=-1))

    from torch_judge import check

    check("softmax")
