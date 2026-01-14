import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from performer_pytorch import Performer

# =======================================================
# 1. Image Encoder (Patch → Spot-level visual feature)
# =======================================================
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)

    def forward(self, x):
        """
        x: (N_spots, 3, 224, 224)
        return: (N_spots, embed_dim)
        """
        return self.backbone(x)


# =======================================================
# 2. Spatial ST Encoder (HVG-only, scBERT-style)
# =======================================================
class SpatialSTEncoder(nn.Module):
    """
    scBERT-style encoder with explicit spatial token

    Input:
      - expr   : (N_spots, K)   [already HVG-filtered]
      - coords : (N_spots, 2)   [normalized]

    Output:
      - (N_spots, embed_dim) spot-level ST embedding
    """

    def __init__(
        self,
        num_genes,        # K = number of HVGs (e.g., 2000)
        embed_dim=256,
        num_layers=2,
        num_heads=4,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_genes = num_genes

        # Gene identity embedding (HVG-only)
        self.gene_embedding = nn.Embedding(num_genes, embed_dim)

        # Gene positional embedding (gene order)
        self.gene_pos_embedding = nn.Embedding(num_genes, embed_dim)

        # Expression value embedding
        self.value_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Spatial token embedding
        self.spatial_embedding = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Performer (efficient transformer)
        self.transformer = Performer(
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=embed_dim // num_heads,
            causal=False,
            ff_mult=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        # Spatial-query pooling
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, expr, coords):
        """
        expr   : (N, K)
        coords : (N, 2)
        """
        N, K = expr.shape
        device = expr.device

        # Gene IDs: 0 ... K-1
        gene_ids = torch.arange(K, device=device).unsqueeze(0).expand(N, -1)

        gene_embed = self.gene_embedding(gene_ids)
        gene_pos   = self.gene_pos_embedding(gene_ids)
        value_emb  = self.value_embedding(expr.unsqueeze(-1))

        gene_tokens = gene_embed + gene_pos + value_emb  # (N, K, D)

        spatial_token = self.spatial_embedding(coords).unsqueeze(1)
        tokens = torch.cat([spatial_token, gene_tokens], dim=1)

        tokens = self.transformer(tokens)

        spatial_out = tokens[:, :1]
        gene_out    = tokens[:, 1:]

        q = self.q_proj(spatial_out)
        k = self.k_proj(gene_out)
        v = self.v_proj(gene_out)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5),
            dim=-1
        )

        pooled = torch.matmul(attn, v).squeeze(1)
        return self.out_proj(pooled)



# =======================================================
# 3. Spot Fusion Module (Image + ST)
# =======================================================
class SpotFusionModule(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, img_feat, st_feat):
        return self.fusion(torch.cat([img_feat, st_feat], dim=-1))


# =======================================================
# 4. MIL Attention Pooling (Spot → WSI)
# =======================================================
class MILAttentionPooling(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=128):
        super().__init__()
        self.attn_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        self.attn_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, spot_embeds):
        """
        spot_embeds: (N_spots, D)
        """
        A = self.attn_w(self.attn_V(spot_embeds) * self.attn_U(spot_embeds))
        weights = F.softmax(A, dim=0)
        wsi_embed = torch.sum(weights * spot_embeds, dim=0)
        return wsi_embed, weights


# =======================================================
# 5. Full Multi-Modal MIL Model
# =======================================================
class MultiModalMILModel(nn.Module):
    def __init__(
        self,
        num_genes=2000,
        num_classes=2,
        embed_dim=256,
    ):
        super().__init__()

        self.img_encoder = ImageEncoder(embed_dim)
        self.st_encoder  = SpatialSTEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
        )

        self.fusion      = SpotFusionModule(embed_dim)
        self.mil_pooling = MILAttentionPooling(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, expr, coords):
        """
        images: (N_spots, 3, 224, 224)
        expr  : (N_spots, K)
        coords: (N_spots, 2)
        """
        img_feat = self.img_encoder(images)
        st_feat  = self.st_encoder(expr, coords)

        spot_embeds = self.fusion(img_feat, st_feat)
        wsi_embed, attn = self.mil_pooling(spot_embeds)

        logits = self.classifier(wsi_embed.unsqueeze(0))
        return logits.squeeze(0), attn


