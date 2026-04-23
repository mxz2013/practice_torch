"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/22_conv2d.ipynb)

# 🟠 Medium: 2D Convolution

Implement **2D convolution** from scratch.

### Signature
```python
def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    # x: (B, C_in, H, W), weight: (C_out, C_in, kH, kW)
    # Returns: (B, C_out, H_out, W_out)
```

### Rules
- Do NOT use `F.conv2d` or `nn.Conv2d`
- Support `stride` and `padding` parameters
- `F.pad` for zero-padding is allowed
"""

import torch
import torch.nn.functional as F
# ✏️ YOUR IMPLEMENTATION HERE


def my_conv2d(
    x: torch.Tensor, weight: torch.Tensor, bias=None, stride: int = 1, padding: int = 0
):
    """perform 2D converlution

    Args:
        x (torch.Tensor): input tensor in batch, c, h, w
        weight (torch.Tensor): kernel, in out_channels, in_channles, kh, kw
        bias (_type_, optional): _description_. Defaults to None.
        stride (int, optional): _description_. Defaults to 1.
        padding (int, optional): _description_. Defaults to 0.
    """
    B, C_in, H, W = x.shape
    C_out, _, kh, kw = weight.shape
    if padding > 0:
        x = F.pad(x, [padding] * 4)  # pad all dimensions

    # F.pad expects padding specified as (left, right, top, bottom) — 4 values, one per side.
    # When you write [padding] * 4, you're creating:
    # [padding, padding, padding, padding] left    right    top    bottom
    # Which pads all 4 sides of the spatial dimensions equally — the standard "same padding" behavior.
    # F.pad works from the last dimension inward, so the 4 values apply to W (left/right) then H (top/bottom). The B and C_in dimensions are untouched.
    # x shape before: (B, C_in, H,        W)
    # x shape after:  (B, C_in, H+2p, W+2p)
    # remember this formula !!!
    # H_out = (
    #     H - kh
    # ) // stride + 1  # e.g., H=6, kh=2, stride = 2, [123456] -> [12,34,56]
    # W_out = (W - kw) // stride + 1
    # # Tensor.unfold(dimension, size, step)
    patches = x.unfold(2, kh, stride).unfold(3, kw, stride)
    #  Step 1: x.unfold(2, kh, stride)
    #  unfold(dimension, size, step) slides a window along a dimension and stacks the windows as a new trailing dimension.
    #  x:                      (B, C_in, H_pad, W_pad)
    #  after .unfold(2, kh, stride):  (B, C_in, H_out, W_pad, kh)
    #                                                   ↑ new dim: each height window
    # Step 2: .unfold(3, kw, stride)
    # Same thing along the width dimension (dim 3 is still W_pad at this point):
    # after .unfold(3, kw, stride):  (B, C_in, H_out, W_out, kh, kw)
    # So patches[b, i, h, w, :, :] is the (kh, kw) input patch at position (h, w) for channel i.
    # Every sliding window position is now a tensor slice — no Python loop needed.
    out = torch.einsum("bihwjk,oijk->bohw", patches, weight)
    # Step 3: einsum("bihwjk,oijk->bohw", patches, weight)
    # Map the index letters to shapes:
    # patches: b=B,  i=C_in, h=H_out, w=W_out, j=kh, k=kw
    # weight:       o=C_out, i=C_in,           j=kh, k=kw
    # output:  b=B, o=C_out, h=H_out, w=W_out
    # Indices i, j, k appear in inputs but not in output → they are summed over.
    # This computes:
    # out[b, o, h, w] = Σ_{i,j,k}  patches[b,i,h,w,j,k] × weight[o,i,j,k]
    # Which is exactly the convolution: multiply each patch element by the corresponding kernel weight, sum across all input channels and kernel spatial positions. Repeated for each output channel o and each spatial position (h, w).
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
        # bias has shape (C_out,) — a flat vector. out has shape (B, C_out, H_out, W_out).
        # PyTorch broadcasts from the right, so you need bias to align with the C_out dimension (dim 1), not the last dimension:
        # out:   (B,  C_out, H_out, W_out)
        # bias:       (C_out,)              ← aligns to the RIGHT → matches W_out, wrong!
        # After .view(1, -1, 1, 1):
        # out:   (B,  C_out, H_out, W_out)
        # bias:  (1,  C_out,     1,     1)  ← aligns to C_out dim ✓, broadcasts over B/H/W
    return out
    # pass  # extract patches, apply kernel, handle stride/padding


if __name__ == "__main__":
    # 🧪 Debug
    x = torch.randn(1, 3, 8, 8)  # batch, c, h, w
    w = torch.randn(16, 3, 3, 3)  # n_kernels, chw
    print("Output:", my_conv2d(x, w).shape)
    print("Match:", torch.allclose(my_conv2d(x, w), F.conv2d(x, w), atol=1e-4))
    # ✅ SUBMIT
    from torch_judge import check

    check("conv2d")
