import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from performer_pytorch import Performer

# -------------------------------------------------------
# 1. Image Encoder
# -------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)

    def forward(self, x):
        return self.backbone(x)

# -------------------------------------------------------
# 2. Spatial ST Encoder (HVG-only scBERT-style)
# -------------------------------------------------------
class SpatialSTEncoder(nn.Module):
    """
    HVG-only scBERT-style encoder with spatial token

    Assumption:
    - expr is already HVG-filtered
    - gene indices are 0 ~ K-1
    """

    def __init__(
        self,
        top_k_genes,          # K = number of HVGs
        embed_dim=256,
        num_layers=2,
        num_heads=4,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.top_k_genes = top_k_genes

        # -------------------------------------------------
        # Gene identity embedding (HVG-only)
        # -------------------------------------------------
        self.gene_embedding = nn.Embedding(top_k_genes, embed_dim)

        # Gene positional embedding
        self.gene_pos_embedding = nn.Embedding(top_k_genes, embed_dim)

        # Continuous expression value embedding (SAFE)
        self.value_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Spatial token embedding
        self.spatial_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Performer Transformer
        self.performer = Performer(
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
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj   = nn.Linear(embed_dim, embed_dim)

    def forward(self, expr, coords):
        """
        Args:
            expr:   (B, K) HVG expression
            coords: (B, 2) normalized spatial coordinates

        Returns:
            (B, embed_dim) spot-level ST embedding
        """
        B, K = expr.shape
        device = expr.device

        assert K <= self.top_k_genes, f"K={K} exceeds top_k_genes={self.top_k_genes}"

        # Gene IDs: 0 ~ K-1
        gene_ids = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)

        gene_id_embeds  = self.gene_embedding(gene_ids)
        gene_pos_embeds = self.gene_pos_embedding(gene_ids)

        value_embeds = self.value_embedding(expr.unsqueeze(-1))

        gene_tokens = gene_id_embeds + gene_pos_embeds + value_embeds

        # Spatial token
        spatial_token = self.spatial_embed(coords).unsqueeze(1)

        # Token sequence
        tokens = torch.cat([spatial_token, gene_tokens], dim=1)

        # Transformer
        tokens = self.performer(tokens)

        spatial_out = tokens[:, :1]   # (B,1,D)
        gene_out    = tokens[:, 1:]   # (B,K,D)

        q = self.query_proj(spatial_out)
        k = self.key_proj(gene_out)
        v = self.value_proj(gene_out)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5),
            dim=-1
        )

        pooled = torch.matmul(attn, v).squeeze(1)

        return self.out_proj(pooled)

# -------------------------------------------------------
# 3. Spot Fusion Module
# -------------------------------------------------------
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

# -------------------------------------------------------
# 4. MIL Attention Pooling
# -------------------------------------------------------
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
        A = self.attn_w(self.attn_V(spot_embeds) * self.attn_U(spot_embeds))
        weights = F.softmax(A, dim=0)
        wsi_embed = torch.sum(weights * spot_embeds, dim=0)
        return wsi_embed, weights

# -------------------------------------------------------
# 5. Full Multi-Modal MIL Model
# -------------------------------------------------------
class MultiModalMILModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
        embed_dim=256,
        top_k_genes=256,
    ):
        super().__init__()

        self.img_encoder = ImageEncoder(embed_dim)
        self.st_encoder  = SpatialSTEncoder(
            top_k_genes=top_k_genes,
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
        img_feat = self.img_encoder(images)
        st_feat  = self.st_encoder(expr, coords)

        spot_embeds = self.fusion(img_feat, st_feat)
        wsi_embed, attn = self.mil_pooling(spot_embeds)

        logits = self.classifier(wsi_embed.unsqueeze(0))
        return logits.squeeze(0), attn_weights
    
    def set_selected_genes(self, gene_indices):
        """Propagate HVG indices to ST encoder"""
        self.st_encoder.set_selected_genes(gene_indices)

# -------------------------------------------------------
# 6. Test with loader integration
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Modal MIL Model Test (Integrated with Loader)")
    print("=" * 70)
    
    # Simulate loader output
    print("\n[Step 1] Simulating loader preprocessing...")
    
    # Case 1: Intersection (small dataset)
    # total_genes = 362
    # Case 2: Union (large dataset)
    total_genes = 37094  # From loader: Union strategy
    
    print(f"  Total genes after preprocessing: {total_genes}")
    
    # HVG selection
    actual_hvg_count = min(512, total_genes)
    print(f"  HVG count: {actual_hvg_count}")
    
    # Simulate HVG indices from loader
    import numpy as np
    np.random.seed(42)
    hvg_indices = torch.from_numpy(
        np.sort(np.random.choice(total_genes, actual_hvg_count, replace=False))
    ).long()
    
    print(f"  HVG indices (first 10): {hvg_indices[:10].tolist()}")
    
    # Initialize model
    print("\n[Step 2] Initializing model...")
    model = MultiModalMILModel(
        total_vocab_size=total_genes,  # Must match loader!
        num_classes=2,
        embed_dim=128,
        top_k_genes=actual_hvg_count,
        use_value_binning=True,
        num_bins=5
    )
    
    # Set HVG
    model.set_selected_genes(hvg_indices)
    print(f"  Model configured with {actual_hvg_count} HVG genes")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model memory: {total_params * 4 / (1024**2):.1f} MB")
    
    # Simulate loader batch
    print("\n[Step 3] Simulating loader batch...")
    N_spots = 100
    
    dummy_images = torch.randn(N_spots, 3, 224, 224)
    dummy_expr = torch.rand(N_spots, actual_hvg_count)  # Already filtered!
    dummy_coords = torch.rand(N_spots, 2)
    
    print(f"  Batch from loader:")
    print(f"    images: {dummy_images.shape}")
    print(f"    expr: {dummy_expr.shape}  ← {actual_hvg_count} HVG genes")
    print(f"    coords: {dummy_coords.shape}")
    
    # Forward pass
    print("\n[Step 4] Forward pass...")
    try:
        with torch.no_grad():
            logits, attn = model(dummy_images, dummy_expr, dummy_coords)
        
        print("  ✓ Success!")
        print(f"\n  Outputs:")
        print(f"    logits: {logits.shape}")
        print(f"    attention weights: {attn.shape}")
        
        probs = torch.softmax(logits, dim=0)
        print(f"\n  Predictions:")
        print(f"    Healthy: {probs[0]:.4f}")
        print(f"    Tumor: {probs[1]:.4f}")
        
        print(f"\n  Top 5 important spots:")
        top5 = torch.topk(attn.squeeze(), k=5)
        for i, (idx, weight) in enumerate(zip(top5.indices, top5.values), 1):
            print(f"    {i}. Spot {idx.item():3d}: {weight.item():.4f}")
        
        print("\n" + "=" * 70)
        print("✓ Model is ready for training!")
        print("=" * 70)
        
    except RuntimeError as e:
        print(f"  ✗ Error: {e}")
        print("\n  Debugging info:")
        print(f"    total_vocab_size: {total_genes}")
        print(f"    top_k_genes: {actual_hvg_count}")
        print(f"    hvg_indices range: [{hvg_indices.min()}, {hvg_indices.max()}]")
        print(f"    expr shape: {dummy_expr.shape}")
