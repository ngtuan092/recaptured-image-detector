import torch
import torch.nn as nn
import torch.nn.functional as F
from .Block import ResidualBlock, MLPBlock
from .PatchEmbedding import PatchEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim=768, num_heads=12, drop_rate=0.) -> None:
        super().__init__()
        self.multiHead = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, dropout=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multiHead(x, x, x)[0]


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, num_heads, expansion: int, drop_p: float,  feed_forward_drop_p: float, depth: int = 12) -> None:
        super().__init__()
        # patches
        emb_dim = in_channels * (patch_size ** 2)
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, emb_dim=emb_dim, img_size=224)
        self.transformer_blocks = nn.Sequential(
            ResidualBlock(
                nn.Sequential(
                    nn.LayerNorm(emb_dim),
                    MultiHeadAttention(emb_dim=emb_dim, num_heads=num_heads),
                    nn.Dropout(drop_p),
                )),
            ResidualBlock(
                nn.Sequential(
                    nn.LayerNorm(emb_dim),
                    MLPBlock(emb_dim, emb_dim,
                             emb_dim, feed_forward_drop_p),
                    nn.Dropout(drop_p),
                ))

        )
        self.transformer_encoder = nn.Sequential(
            *[self.transformer_blocks for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        return x


if __name__ == "__main__":
    x = torch.randn(5, 32, 224, 224).to("cuda")
    model = VisionTransformer(in_channels=32, patch_size=16, num_heads=128, expansion=4, drop_p=0.1, feed_forward_drop_p=0.1, depth=4)
    model.to("cuda")
    print(model(x).shape)
