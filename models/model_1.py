import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.performer_pytorch import Performer


# =======================================================
# 1. Image Encoder (Patch → Spot-level visual feature)
# =======================================================

class ImageEncoder(nn.Module):

    def __init__(self, embed_dim=256, backbone='resnet18', pretrained=True):

        super().__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)

    def forward(self, x):
        """
        x: (N_spots, 3, 224, 224)
        return: (N_spots, embed_dim)
        """
        return self.backbone(x)


class SCEncoder(nn.Module):

    def __init__(self, num_genes, embed_dim=256, top_k_genes=None):  

        super().__init__()

        self.top_k_genes = top_k_genes  

        # scBERT Performer
        self.scbert = Performer(
            dim=256,
            depth=6,
            dim_head=32,
            heads=8,
            ff_mult=4,
            causal=False,
            attn_dropout=0.1
        )

        # Token embedding + projection
        self.token_emb = nn.Embedding(7, 256)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        self.proj = nn.Linear(256, embed_dim)

    def preprocess(self, expr):
        """Raw counts → 7 bins"""
        expr = torch.log1p(F.normalize(expr, p=1, dim=-1) * 1e4)
        bins = torch.linspace(0, expr.max(), 8, device=expr.device)
        tokens = torch.bucketize(expr, bins[:-1]).long()
        tokens = torch.clamp(tokens, 0, 6)
        return tokens

    def forward(self, expr):  # (B, num_genes)

        B = expr.shape[0]

        if self.top_k_genes is not None and self.top_k_genes < expr.shape[1]:

            topk_values, topk_indices = torch.topk(expr, k=self.top_k_genes, dim=1)
            expr_filtered = topk_values  # (B, K)

        else:

            expr_filtered = expr  # (B, 2000)

        # 7-bin tokenization
        tokens = self.preprocess(expr_filtered)  # (B, K or 2000)

        # CLS + tokens
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, self.token_emb(tokens)], dim=1)  # (B, K+1, 256)

        # Performer forward
        emb = self.scbert(x)  # (B, K+1, 256)
        cell_emb = emb[:, 0]  # CLS pooling

        return self.proj(cell_emb)  # (B, embed_dim)


class STEncoder(nn.Module):

    def __init__(self, embed_dim=256):

        super().__init__()

        # Convert (x, y) -> embedding
        self.spatial_embed = nn.Sequential(
            nn.Linear(2, embed_dim // 2),  # 2D -> D/2
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU()
        )

        # Positional encoding
        self.pos_enc = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.Tanh()   # range: [-1, 1]
        )

        # CLS token + spatial token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Final projection layer
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):

        # x: (B, 2) / spatial codinates (x, y)
        B = x.shape[0]

        spatial_emb = self.spatial_embed(x)   # (B, D/2)
        pos_emb = self.pos_enc(x)             # (B, D/2)

        token_emb = torch.cat([spatial_emb, pos_emb], dim=-1)

        # CLS token + spatial token
        cls = self.cls_token.expand(B, 1, -1)                 # (B, 1, D)
        seq = torch.cat([cls, token_emb.unsqueeze(1)], dim=1) # (B, 2, D)

        # Apply transformer
        seq = self.transformer(seq) # (B, 2, D)

        # Apply CLS token
        seq = seq[:, 0, :]            # (B, D)

        return self.fc(seq)


# -------------------------------------------------------
# 3. Fusion Layer
# -------------------------------------------------------

class FusionLayer(nn.Module):

    """
    Fusion options (fused_dim=256):

    - 'concat'  : concat([img, sc, st]) -> MLP -> fused (B, fused_dim)

    - 'attn'    : treat [img, sc, st] as 3 tokens -> self-attn -> pool -> fused (B, fused_dim)

    - 'sim'    : concat([img, sc, st, img*sc, img*st, sc*st, |img-sc|, |img-st|, |sc-st|, cosine(img,sc), cosine(img,st), cosine(sc,st)])
                -> MLP -> fused

    - 'gate'    : gate ([img, sc, st]) -> MLP -> fused (B, fused_dim)
    """

    def __init__(
            self,
            embed_dim: int = 256,
            fusion_option: str = 'concat',
            fused_dim: int = 128,
            attn_heads: int = 4,
            dropout: float = 0.2,
            use_l2norm_for_sim: bool = True
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.fusion_option = fusion_option
        self.fused_dim = fused_dim
        self.dropout = dropout
        self.use_l2norm_for_sim = use_l2norm_for_sim

        # pre-norm for modality feature alignment
        self.pre_norm_img = nn.LayerNorm(embed_dim)
        self.pre_norm_sc = nn.LayerNorm(embed_dim)
        self.pre_norm_st = nn.LayerNorm(embed_dim)

        if fusion_option == 'concat':

            self.concat_fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, fused_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim * 2, fused_dim)
            )

        elif fusion_option == 'attn':

            self.attn_fusion = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True
            )

            self.norm1 = nn.LayerNorm(embed_dim)

            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )

            self.norm2 = nn.LayerNorm(embed_dim)

            self.out_proj = nn.Linear(embed_dim, fused_dim)

        elif fusion_option == 'sim':

            self.sim_fusion = nn.Sequential(
                nn.Linear(embed_dim * 9 + 3, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.3),
            )

        elif fusion_option == 'gate':

            self.gate_fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 3),
                nn.Softmax(dim=-1)
            )

            self.proj_img = nn.Linear(embed_dim, fused_dim)
            self.proj_sc = nn.Linear(embed_dim, fused_dim)
            self.proj_st = nn.Linear(embed_dim, fused_dim)

            self.gate_final = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        else:

            raise ValueError(f"Unknown fusion option={fusion_option}")

    def forward(self, img_feat: torch.Tensor, sc_feat: torch.Tensor, st_feat: torch.Tensor) -> torch.Tensor:

        if img_feat.ndim != 2 or st_feat.ndim != 2 or sc_feat.ndim != 2:
            raise ValueError("Input features must be 2D tensors of shape (B, D)")

        if not (img_feat.shape == sc_feat.shape == st_feat.shape):
            raise ValueError(f"Shape mismatch: {img_feat.shape} vs {sc_feat.shape} vs {st_feat.shape}")

        if not (img_feat.device == sc_feat.device == st_feat.device):
            raise ValueError(f"Device mismatch: {img_feat.device} vs {sc_feat.device} vs {st_feat.device}")

        if not (img_feat.dtype == sc_feat.dtype == st_feat.dtype):
            raise ValueError(f"Dtype mismatch: {img_feat.dtype} vs {sc_feat.dtype} vs {st_feat.dtype}")

        img_feat = self.pre_norm_img(img_feat)
        sc_feat = self.pre_norm_sc(sc_feat)
        st_feat = self.pre_norm_st(st_feat)

        if self.fusion_option == 'concat':

            x = torch.cat([img_feat, sc_feat, st_feat], dim=1)
            return self.concat_fusion(x)

        elif self.fusion_option == 'attn':

            tokens = torch.stack([img_feat, sc_feat, st_feat], dim=1)

            attn_out, _ = self.attn_fusion(tokens, tokens, tokens)
            tokens = self.norm1(tokens + attn_out)

            ffn_out = self.ffn(tokens)
            tokens = self.norm2(tokens + ffn_out)

            pooled = tokens.mean(dim=1)

            return self.out_proj(pooled)

        elif self.fusion_option == 'sim':

            img_n = img_feat
            sc_n = sc_feat
            st_n = st_feat

            img_norm = F.normalize(img_feat, p=2, dim=1, eps=1e-8)
            sc_norm = F.normalize(sc_feat, p=2, dim=1, eps=1e-8)
            st_norm = F.normalize(st_feat, p=2, dim=1, eps=1e-8)

            cos_img_sc = F.cosine_similarity(img_norm, sc_norm, dim=1).unsqueeze(1)
            cos_img_st = F.cosine_similarity(img_norm, st_norm, dim=1).unsqueeze(1)
            cos_sc_st = F.cosine_similarity(sc_norm, st_norm, dim=1).unsqueeze(1)

            prod_img_sc = img_n * sc_n
            prod_img_st = img_n * st_n
            prod_sc_st = sc_n * st_n

            diff_img_sc = torch.abs(img_n - sc_n)
            diff_img_st = torch.abs(img_n - st_n)
            diff_sc_st = torch.abs(sc_n - st_n)

            x = torch.cat([
                img_n, sc_n, st_n,
                prod_img_sc, prod_img_st, prod_sc_st,
                diff_img_sc, diff_img_st, diff_sc_st,
                cos_img_sc, cos_img_st, cos_sc_st
            ], dim=1)

            return self.sim_fusion(x)

        elif self.fusion_option == 'gate':

            x = torch.cat([img_feat, sc_feat, st_feat], dim=-1)

            weights = self.gate_fusion(x)

            proj_img = self.proj_img(img_feat)
            proj_sc = self.proj_sc(sc_feat)
            proj_st = self.proj_st(st_feat)

            weighted = (
                weights[:, 0:1] * proj_img +
                weights[:, 1:2] * proj_sc +
                weights[:, 2:3] * proj_st
            )

            return self.gate_final(weighted)


# =======================================================
# 4. MIL Attention Pooling (Spot → WSI)
# =======================================================

class MILAttentionPooling(nn.Module):

    def __init__(self, embed_dim=256, hidden_dim: int = None, dropout: float = 0.0):

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

    def forward(self, spot_embeds):

        A = self.attn_w(self.attn_V(spot_embeds) * self.attn_U(spot_embeds))
        weights = F.softmax(A, dim=0)
        wsi_embed = torch.sum(weights * spot_embeds, dim=0)

        return wsi_embed, weights


# =======================================================
# 5. Layer after encoders
# =======================================================

class LinearHead(nn.Module):

    def __init__(self, dim: int, use_ln: bool = True):

        super().__init__()

        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.fc(self.ln(x))


# -------------------------------------------------------
# 6. Full Multi-Modal Model (Fusion + Classifier)
# -------------------------------------------------------

class MultiModalMILModel(nn.Module):

    def __init__(
            self,
            num_genes: int,
            modality_option: str = 'multi',
            num_classes: int = 2,
            embed_dim: int = 256,
            fusion_option: str = 'concat',
            top_k_genes: int = None,
            img_backbone: str = 'resnet18',
            img_pretrained: bool = True,
            mil_hidden_dim: int = None,
            mil_dropout: float = 0.0,
            fusion_dropout: float = 0.2,
            head_use_ln: bool = True
    ):

        super().__init__()

        self.img_encoder = ImageEncoder(embed_dim=embed_dim, backbone=img_backbone, pretrained=img_pretrained)

        self.sc_encoder = SCEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            top_k_genes=top_k_genes
        )

        self.st_encoder = STEncoder(embed_dim=embed_dim)

        self.img_head = LinearHead(dim=embed_dim, use_ln=head_use_ln)
        self.sc_head = nn.Identity()
        self.st_head = nn.Identity()

        self.freeze_encoders()

        print(f"✓ Model initialized with fusion_option='{fusion_option}'")
        if top_k_genes:
            print(f"✓ Using top_k_genes={top_k_genes} for SCEncoder")

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

    def train(self, mode: bool = True):

        super().train(mode)
        self.img_encoder.eval()

    def forward(self, modality_option, img: torch.Tensor, expr: torch.Tensor, coord: torch.Tensor):

        with torch.no_grad():

            img_feat = self.img_encoder(img)
            sc_feat = self.sc_encoder(expr)
            st_feat = self.st_encoder(coord)

        if modality_option == 'st':

            sc_feat = self.sc_head(sc_feat)
            st_feat = self.st_head(st_feat)

            spot_embeds = torch.cat([img_feat, st_feat], dim=1)

            wsi_embed, attn = self.mil_pooling(spot_embeds)

            logits = self.classifier(wsi_embed.unsqueeze(0)).squeeze(0)

        elif modality_option == 'img':

            img_feat = self.img_head(img_feat)

            wsi_embed, attn = self.mil_pooling(img_feat)

            logits = self.classifier(wsi_embed.unsqueeze(0)).squeeze(0)

        elif modality_option == 'multi':

            img_feat = self.img_head(img_feat)
            sc_feat = self.sc_head(sc_feat)
            st_feat = self.st_head(st_feat)

            spot_embeds = self.fusion(img_feat, sc_feat, st_feat)

            wsi_embed, attn = self.mil_pooling(spot_embeds)

            logits = self.classifier(wsi_embed.unsqueeze(0)).squeeze(0)

        else:

            raise ValueError(f"Unknown modality option={modality_option}")

        return logits, attn
