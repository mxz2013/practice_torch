"""
# 🟢 Easy: Implement ReLU

Implement the **ReLU** (Rectified Linear Unit) activation function from scratch.

$$\text{ReLU}(x) = \max(0, x)$$

### Signature
```python
def relu(x: torch.Tensor) -> torch.Tensor:
    ...
```

### Rules
- Do **NOT** use `torch.relu`, `F.relu`, `torch.clamp`, or any built-in activation
- Must support autograd (gradients should flow back)

### Example
```
Input:  tensor([-2., -1., 0., 1., 2.])
Output: tensor([ 0.,  0., 0., 1., 2.])
"""

import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    mask = x > 0
    return x * mask


# ❌ using NumPy → breaks autograd
# ❌ in-place ops like x[x < 0] = 0 → can break gradients

# (x > 0) → returns a boolean tensor
# In PyTorch, it becomes {1, 0} when multiplied
# So:
# positive values → x * 1 = x
# negative values → x * 0 = 0

# ✔ Fully differentiable
# ✔ Autograd works correctly
# ✔ No forbidden functions used

if __name__ == "__main__":
    # 🧪 Test your implementation (feel free to add more debug prints)
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("Input: ", x)
    print("Output:", relu(x))
    print("Shape: ", relu(x).shape)

    from torch_judge import check

    check("relu")
