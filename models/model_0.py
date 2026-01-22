import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# -------------------------------------------------------
# 1. Image Encoder
# -------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, backbone='resnet18', pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)
        
    def forward(self, x):
        return self.backbone(x)


# -------------------------------------------------------
# 2. Spatial Transcriptomics (ST) Encoder
# -------------------------------------------------------
class STEncoder(nn.Module):
    def __init__(
        self, 
        num_genes: int, 
        embed_dim: int=256, 
        nhead: int=4, 
        num_layers: int=2, 
        dropout: float=0.1,
        top_k_genes: int=None  
    ):
        super().__init__()
        
        if top_k_genes is not None:
            print(f"⚠️  top_k_genes={top_k_genes} provided but not implemented in this encoder")
        
        self.gene_embed = nn.Sequential(
            nn.Linear(num_genes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, expr: torch.Tensor, coord: torch.Tensor=None) -> torch.Tensor:
        x = self.gene_embed(expr)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.proj(x)


# -------------------------------------------------------
# 3. Fusion Layer
# -------------------------------------------------------
class FusionLayer(nn.Module):
    """
    Fusion options (fused_dim=256):
    - 'concat'  : concat([img, st]) -> MLP -> fused (B, fused_dim)
    - 'attn'    : treat [img, st] as 2 tokens -> self-attention -> pool -> fused (B, fused_dim)
    - 'sim'     : concat([img, st, img*st, |img-st|, cosine(img, st)]) -> MLP -> fused (B, fused_dim)
    - 'gate'    : gated fusion
    """
    def __init__(
            self,
            embed_dim: int=256,
            fusion_option: str='concat',
            fused_dim: int=256,
            attn_heads: int=4,
            dropout: float=0.2,
            use_l2norm_for_sim: bool=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_option = fusion_option
        self.fused_dim = fused_dim
        self.dropout = dropout
        self.use_l2norm_for_sim = use_l2norm_for_sim
        
        self.pre_norm_img = nn.LayerNorm(embed_dim)
        self.pre_norm_st = nn.LayerNorm(embed_dim)

        if fusion_option == 'concat':
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 2, fused_dim),
                nn.ReLU(),
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
            self.out_proj = nn.Linear(embed_dim, fused_dim)

        elif fusion_option == 'sim':
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 4 + 1, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif fusion_option == 'gate':
            # Two-modality gating: [img, st]
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
                nn.Softmax(dim=-1),
            )
            self.proj = nn.Linear(embed_dim, fused_dim)
                
        else:
            raise ValueError(f"Unknown fusion option={fusion_option}")

    def forward(self, img_feat: torch.Tensor, st_feat: torch.Tensor) -> torch.Tensor:

        # Basic input validation
        if img_feat.ndim != 2 or st_feat.ndim != 2:
            raise ValueError("img_feat and st_feat must be 2D tensors of shape (B, D)")
        if img_feat.shape != st_feat.shape:
            raise ValueError(f"Shape mismatch: {img_feat.shape} vs {st_feat.shape}")

        # Pre-normalization to align feature distributions
        img_feat = self.pre_norm_img(img_feat)
        st_feat = self.pre_norm_st(st_feat)

        if self.fusion_option == 'concat':
            x = torch.cat([img_feat, st_feat], dim=1)   # (B, 2D)
            return self.fuse(x)                         # (B, fused_dim)

        elif self.fusion_option == 'attn':
            # Two tokens: [img_feat, st_feat]
            tokens = torch.stack([img_feat, st_feat], dim=1)    # (B, 2, D)
            attn_out, _ = self.attn(tokens, tokens, tokens)     # (B, 2, D)
            tokens = self.norm1(tokens + attn_out)
            ffn_out = self.ffn(tokens)                          # (B, 2, D)
            tokens = self.norm2(tokens + ffn_out)
            pooled = tokens.mean(dim=1)                         # (B, D)
            return self.out_proj(pooled)                        # (B, fused_dim)
        
        elif self.fusion_option == 'sim':
            if self.use_l2norm_for_sim:
                img_n = F.normalize(img_feat, p=2, dim=1, eps=1e-8)
                st_n = F.normalize(st_feat, p=2, dim=1, eps=1e-8)
            else:
                img_n = img_feat
                st_n = st_feat

            sim = F.cosine_similarity(img_n, st_n, dim=1, eps=1e-8).unsqueeze(1)  # (B, 1)
            prod = img_n * st_n
            abs_diff = torch.abs(img_n - st_n)

            x = torch.cat([img_n, st_n, prod, abs_diff, sim], dim=1)  # (B, 4D+1)
            return self.fuse(x)  # (B, fused_dim)
        
        elif self.fusion_option == 'gate':
            # Concatenate both modalities
            x = torch.cat([img_feat, st_feat], dim=1)  # (B, 2D)
            weights = self.gate(x)                      # (B, 2)
            fused = weights[:, 0:1] * img_feat + weights[:, 1:2] * st_feat  # (B, D)
            return self.proj(fused)   # (B, fused_dim)


# -------------------------------------------------------
# 4. MIL Pooling (WSI-Level Aggregation)
# -------------------------------------------------------
class MILAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_dim: int=None, dropout: float=0.0):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(64, embed_dim)

        self.attn_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        self.attn_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, spot_embeds: torch.Tensor):
        """
        Modified to support batch processing,
        while preserving original functionality
        """
        # spot_embeds: (N, D) or (B, N, D)
        if spot_embeds.ndim == 2:
            H = spot_embeds.unsqueeze(0)  # (1, N, D)
            squeeze_back = True
        elif spot_embeds.ndim == 3:
            H = spot_embeds              # (B, N, D)
            squeeze_back = False
        else:
            raise ValueError("spot_embeds must be (N, D) or (B, N, D)")

        H = self.dropout(H)

        V = self.attn_V(H)               # (B, N, A)
        U = self.attn_U(H)               # (B, N, A)
        A = self.attn_w(V * U)           # (B, N, 1)
        alpha = torch.softmax(A, dim=1)  # (B, N, 1)

        Z = torch.sum(alpha * H, dim=1)  # (B, D)

        if squeeze_back:
            return Z.squeeze(0), alpha.squeeze(0)  # (D,), (N, 1)
        return Z, alpha     


# --------------------------------------------------------
# 5. Post-Encoder Projection Layer
# --------------------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, dim: int, use_ln: bool=True):
        super().__init__()
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.fc = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.ln(x))


# -------------------------------------------------------
# 6. Full Multi-Modal MIL Model
# -------------------------------------------------------
class MultiModalMILModel(nn.Module):
    def __init__(
        self,
        num_genes: int, 
        num_classes: int=2,
        embed_dim: int=256,
        fusion_option: str='concat',
        top_k_genes: int=None,             
        dropout: float=0.3,                
        freeze_image_encoder: bool=True,   
        img_backbone: str='resnet18',
        img_pretrained: bool=True,
        mil_hidden_dim: int=None,
        mil_dropout: float=0.0,
        fusion_dropout: float=0.2,
        head_use_ln: bool=True
    ):
        super().__init__()
        
        if top_k_genes is not None:
            print(f"⚠️  top_k_genes={top_k_genes} provided but not used")
        
        self.img_encoder = ImageEncoder(
            embed_dim=embed_dim, 
            backbone=img_backbone, 
            pretrained=img_pretrained
        )
        self.st_encoder = STEncoder(
            num_genes=num_genes, 
            embed_dim=embed_dim, 
            nhead=4, 
            num_layers=2, 
            dropout=0.1,
            top_k_genes=top_k_genes
        )
        
        self.img_head = LinearHead(dim=embed_dim, use_ln=head_use_ln)
        self.st_head = nn.Identity()

        # Conditional freezing of image encoder
        if freeze_image_encoder:
            self.freeze_encoders()

        print(f"✓ Model initialized with fusion_option='{fusion_option}'")
        
        self.fusion = FusionLayer(
            embed_dim=embed_dim,
            fusion_option=fusion_option,
            fused_dim=embed_dim,
            attn_heads=4,
            dropout=fusion_dropout,
            use_l2norm_for_sim=True,
        )
        
        self.mil_pooling = MILAttentionPooling(
            embed_dim=embed_dim,
            hidden_dim=mil_hidden_dim,
            dropout=mil_dropout,
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def freeze_encoders(self):
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.img_encoder.eval()
    
    def train(self, mode: bool=True):
        super().train(mode)
        self.img_encoder.eval()

    def forward(self, img: torch.Tensor, expr: torch.Tensor, coords: torch.Tensor=None):
        with torch.no_grad():
            img_feat = self.img_encoder(img)
        st_feat = self.st_encoder(expr, coords)

        img_feat = self.img_head(img_feat)
        st_feat = self.st_head(st_feat)

        spot_embeds = self.fusion(img_feat, st_feat)
        wsi_embed, attn = self.mil_pooling(spot_embeds)
        logits = self.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
        
        return logits, attn
