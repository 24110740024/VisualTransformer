# model.py
import math
import torch
import torch.nn as nn

class ViT(nn.Module):
    """
    最小可用 ViT（batch_first）：
    - Patch Embedding = Conv2d(in_chans, embed_dim, kernel=patch, stride=patch)
    - CLS token + 可学习位置编码（1 + N_patches, D）
    - TransformerEncoder (nn.TransformerEncoderLayer, GELU)
    - LayerNorm + 线性分类头
    """
    def __init__(self,
                 in_channels=3,
                 img_size=224,
                 patch_size=16,
                 num_classes=2,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=3072,
                 dropout=0.1,
                 pos_dropout=0.0):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.embed_dim = embed_dim

        # [B, C, H, W] -> [B, D, H/P, W/P]
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size, bias=True)

        # CLS + Position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop  = nn.Dropout(pos_dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=mlp_dim, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        B = x.shape[0]
        x = self.patch_embed(x)               # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)      # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)          # [B, 1+N, D]

        x = x + self.pos_embed[:, :x.size(1), :] # 位置编码（若日后换分辨率，可做2D插值）
        x = self.pos_drop(x)

        x = self.encoder(x)                    # [B, 1+N, D]
        x = self.norm(x[:, 0])                # 取 CLS
        logits = self.head(x)                 # [B, C]
        return logits
