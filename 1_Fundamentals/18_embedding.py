"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/18_embedding.ipynb)

# 🟢 Easy: Embedding Layer

Implement an **embedding lookup table** from scratch.

### Signature
```python
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int): ...
    def forward(self, indices: Tensor) -> Tensor: ...
```

### Rules
- `self.weight`: `nn.Parameter` of shape `(num_embeddings, embedding_dim)`
- Forward: index into weight matrix — `weight[indices]`
- Do NOT use `nn.Embedding`
"""

import torch
import torch.nn as nn
# ✏️ YOUR IMPLEMENTATION HERE


class MyEmbedding(nn.Module):
    """MyEmbedding(10, 4) means: Create a lookup table with 10 rows.
        Each row is a vector of length 4. so the layer owns a trainable matrix weight.shape = (10, 4)
        e.g.,
        weight =
    [
      row 0: [ ..., ..., ..., ... ],
      row 1: [ ..., ..., ..., ... ],
      row 2: [ ..., ..., ..., ... ],
      row 3: [ ..., ..., ..., ... ],
      ...
      row 9: [ ..., ..., ..., ... ],
    ]

       if we input idx = torch.tensor([0, 3, 7])
        we got weight[idx] = row 0, 3, and 7, resulting a size of (3,4)



            Args:
                nn (_type_): _description_
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # nn.Parameter tells pytorch that, it should be learned by gradient descent
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):
        return self.weight[indices]


if __name__ == "__main__":
    # 🧪 Debug
    emb = MyEmbedding(10, 4)
    idx = torch.tensor([0, 3, 7])
    print("Output shape:", emb(idx).shape)
    print("Matches manual:", torch.equal(emb(idx)[0], emb.weight[0]))
    # ✅ SUBMIT
    from torch_judge import check

    check("embedding")
