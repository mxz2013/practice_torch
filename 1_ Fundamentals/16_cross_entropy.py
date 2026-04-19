"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/16_cross_entropy.ipynb)

# 🟢 Easy: Cross-Entropy Loss

Implement **cross-entropy loss** from scratch.

$$\text{CE}(x, y) = -\log\frac{e^{x_y}}{\sum_j e^{x_j}}$$

### Signature
```python
def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    # logits: (B, C) float, targets: (B,) long indices
    # Returns: scalar loss (mean over batch)
```

### Rules
- Do NOT use `F.cross_entropy` or `nn.CrossEntropyLoss`
- Must be numerically stable (use logsumexp trick)

# math
For one training example, suppose the model outputs logits:
x = [x_0, x_1, ..., x_C]
and the correct class is y -> an integer representing the class ID

CE(x,y) = -log softmax(x)_y = -log[ exp(x_y) / sum_j exp(x_j) ]
= -log exp(x_y) + log[sum_j exp(x_j)]
= - x_y + log(sum_j exp(x_j))
"""

import torch
# ✏️ YOUR IMPLEMENTATION HERE


def cross_entropy_loss(logits, targets):
    """

    Args:
        logits (_type_): _description_
        targets (_type_): _description_
    """
    # logits: (B, C) float, targets: (B,) long indices
    # Returns: scalar loss (mean over batch)
    # first convert logits to prob
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    # to make exp stable, we substract max
    exp_logits = torch.exp(logits - max_logits)

    log_sum_exp = max_logits + torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
    # we need to add max_logits because log(exp(a-b)) = log (exp(a)/exp(b)) = log exp(a) - log exp(b)

    # now we compute the prob, p_j = exp(x_j) / sum_k exp(x_k)
    # and log(p_j) = log(exp(x_j)) - log(sum_k exp(x_K))
    # log(p_j) = x_j - log(sum_k exp (x_k))

    log_probs = logits - log_sum_exp

    # selects the correct class for each batch item -> log_probs[torch.arange(B), targets]
    target_log_probs = log_probs[torch.arange(logits.shape[0]), targets]
    # for example, batch_size = 2, with 3 classes, logits.shape = 2, 3
    # logits = torch.tensor([
    # [1.0, 2.0, 3.0],
    # [5.0, 1.0, 0.0],
    # ])

    # targets = torch.tensor([2, 0])

    # row 0 correct class = 2
    # row 1 correct class = 0

    # log_probs =
    #  [-2.4076, -1.4076, -0.4076],
    #  [-0.0247, -4.0247, -5.0247]
    # ]

    # torch.arange(logits.shape[0]) gives [0,1], and target = [2,0]
    # log_probs[torch.arange(logits.shape[0]), targets] -> log_probs[[0, 1], [2, 0]]
    # so the picked is row 0 col 2, and row 1 col 0

    # Then cross entropy is negative average log probability:
    return -target_log_probs.mean()

    # pass  # log_probs = logits - logsumexp(...)


if __name__ == "__main__":
    # 🧪 Debug
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    print("Loss:", cross_entropy_loss(logits, targets))
    print("Ref: ", torch.nn.functional.cross_entropy(logits, targets))

    # SUBMIT
    from torch_judge import check

    check("cross_entropy")
