import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import repeat


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 14, emb_dim: int = 768, img_size=224) -> None:
        super().__init__()
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        assert emb_dim == in_channels * \
            (patch_size ** 2), "Embedding dimension must be equal to in_channels * (patch_size ** 2)."

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(
            1, (img_size // patch_size) ** 2 + 1, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        return x
