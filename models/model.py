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
        top_k_genes=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_genes = num_genes
        self.top_k_genes = top_k_genes

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

        # ✅ Top-K gene selection 
        if self.top_k_genes and self.top_k_genes < K:
            # top k gene
            topk_values, topk_indices = torch.topk(expr, k=self.top_k_genes, dim=1)
            
            gene_embed = self.gene_embedding(topk_indices)  # (N, top_k, D)
            gene_pos = self.gene_pos_embedding(topk_indices)
            value_emb = self.value_embedding(topk_values.unsqueeze(-1))
            
            gene_tokens = gene_embed + gene_pos + value_emb
        else:
            # all gene
            gene_ids = torch.arange(K, device=device).unsqueeze(0).expand(N, -1)
            gene_embed = self.gene_embedding(gene_ids)
            gene_pos = self.gene_pos_embedding(gene_ids)
            value_emb = self.value_embedding(expr.unsqueeze(-1))
            gene_tokens = gene_embed + gene_pos + value_emb

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
# 3. Spot Fusion Module (4 options: concat, attn, sim, gate)
# =======================================================
class SpotFusionModule(nn.Module):
    """
    Fusion options:
    - 'concat': Simple concatenation + MLP
    - 'attn': Cross-attention between img and st
    - 'sim': Similarity-based fusion (cosine, product, diff)
    - 'gate': Gated fusion with learnable weights
    """
    def __init__(
        self,
        embed_dim=256,
        fusion_option='concat',
        attn_heads=4,
        dropout=0.2,
        use_l2norm_for_sim=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_option = fusion_option
        self.dropout = dropout
        self.use_l2norm_for_sim = use_l2norm_for_sim
        
        # Pre-normalization
        self.pre_norm_img = nn.LayerNorm(embed_dim)
        self.pre_norm_st = nn.LayerNorm(embed_dim)

        if fusion_option == 'concat':
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif fusion_option == 'attn':
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        elif fusion_option == 'sim':
            # 4D + 1 = img, st, product, abs_diff, cosine_sim
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 4 + 1, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif fusion_option == 'gate':
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
                nn.Softmax(dim=-1),
            )
            self.proj = nn.Linear(embed_dim, embed_dim)
        
        else:
            raise ValueError(f"Unknown fusion_option: {fusion_option}")

    def forward(self, img_feat, st_feat):
        """
        img_feat: (N, D)
        st_feat: (N, D)
        return: (N, D)
        """
        # Pre-norm
        img_feat = self.pre_norm_img(img_feat)
        st_feat = self.pre_norm_st(st_feat)

        if self.fusion_option == 'concat':
            # Simple concatenation
            x = torch.cat([img_feat, st_feat], dim=-1)  # (N, 2D)
            return self.fuse(x)  # (N, D)

        elif self.fusion_option == 'attn':
            # Cross-attention: [img, st] as 2 tokens
            tokens = torch.stack([img_feat, st_feat], dim=1)  # (N, 2, D)
            
            # Self-attention
            attn_out, _ = self.attn(tokens, tokens, tokens)  # (N, 2, D)
            tokens = self.norm1(tokens + attn_out)
            
            # FFN
            ffn_out = self.ffn(tokens)  # (N, 2, D)
            tokens = self.norm2(tokens + ffn_out)
            
            # Pool (average)
            pooled = tokens.mean(dim=1)  # (N, D)
            return self.out_proj(pooled)

        elif self.fusion_option == 'sim':
            # Similarity-based features
            if self.use_l2norm_for_sim:
                img_n = F.normalize(img_feat, p=2, dim=-1, eps=1e-8)
                st_n = F.normalize(st_feat, p=2, dim=-1, eps=1e-8)
            else:
                img_n = img_feat
                st_n = st_feat
            
            # Cosine similarity
            sim = F.cosine_similarity(img_n, st_n, dim=-1, eps=1e-8).unsqueeze(-1)  # (N, 1)
            
            # Element-wise product
            prod = img_n * st_n  # (N, D)
            
            # Absolute difference
            abs_diff = torch.abs(img_n - st_n)  # (N, D)
            
            # Concatenate all features
            x = torch.cat([img_n, st_n, prod, abs_diff, sim], dim=-1)  # (N, 4D+1)
            return self.fuse(x)  # (N, D)

        elif self.fusion_option == 'gate':
            # Gated fusion
            x = torch.cat([img_feat, st_feat], dim=-1)  # (N, 2D)
            weights = self.gate(x)  # (N, 2)
            
            # Weighted sum
            fused = weights[:, 0:1] * img_feat + weights[:, 1:2] * st_feat  # (N, D)
            return self.proj(fused)  # (N, D)


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
        fusion_option='concat',
        dropout=0.3,  
        top_k_genes=None,
    ):
        super().__init__()
        
        self.fusion_option = fusion_option

        self.img_encoder = ImageEncoder(embed_dim)
        self.st_encoder  = SpatialSTEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            top_k_genes=top_k_genes,
        )

        self.fusion      = SpotFusionModule(
            embed_dim=embed_dim,
            fusion_option=fusion_option,
        )
        self.mil_pooling = MILAttentionPooling(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        print(f"✓ Model initialized with fusion_option='{fusion_option}'")

    def forward(self, images, expr, coords):
        """
        images: (N_spots, 3, 224, 224)
        expr  : (N_spots, K)
        coords: (N_spots, 2)
        
        Returns:
          logits: (num_classes,)  # Single WSI prediction
          attn: (N_spots, 1)      # Attention weights
        """
        # Spot-level encoding
        img_feat = self.img_encoder(images)      # (N, D)
        st_feat  = self.st_encoder(expr, coords) # (N, D)

        # Spot-level fusion
        spot_embeds = self.fusion(img_feat, st_feat)  # (N, D)
        
        # WSI-level aggregation
        wsi_embed, attn = self.mil_pooling(spot_embeds)  # (D,), (N, 1)

        # Classification
        logits = self.classifier(wsi_embed)  # (num_classes,)
        
        return logits, attn

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
