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
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)
        
    def forward(self, x):
        return self.backbone(x)

# -------------------------------------------------------
# 2. Spatial ST Encoder (scBERT-style)
# -------------------------------------------------------
class SpatialSTEncoder(nn.Module):
    """
    scBERT-based encoder with spatial integration
    
    Key Design:
    - total_vocab_size: Total number of genes
    - top_k_genes: Number of HVG genes
    """
    def __init__(
        self,
        total_vocab_size,    # Total genes in preprocessed data
        embed_dim=256,
        num_layers=2,
        num_heads=4,
        top_k_genes=512,     # HVG count
        use_value_binning=True,
        num_bins=5
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.total_vocab_size = total_vocab_size
        self.top_k_genes = top_k_genes
        self.use_value_binning = use_value_binning
        self.num_bins = num_bins

        # Gene Identity Embedding
        # Size = total_vocab_size 
        self.gene_embedding = nn.Embedding(total_vocab_size, embed_dim)
        
        # Gene Positional Embedding
        # Size = top_k_genes (0~511)
        self.gene_pos_embedding = nn.Embedding(top_k_genes, embed_dim)

        # Expression Value Embedding
        if use_value_binning:
            self.value_embedding = nn.Embedding(num_bins, embed_dim)
        else:
            self.value_embedding = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )

        # Spatial Token Embedding
        self.spatial_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Performer (Efficient Transformer)
        self.performer = Performer(
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=embed_dim // num_heads,
            causal=False,
            ff_mult=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
            qkv_bias=True
        )

        # Spatial-Query Pooling
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Buffer to store selected gene indices
        self.register_buffer(
            'selected_gene_indices', 
            torch.arange(min(top_k_genes, total_vocab_size))
        )
        
    def set_selected_genes(self, gene_indices):
        """
        Set HVG indices (from loader)
        
        Args:
            gene_indices: Tensor of shape (K,) where K <= top_k_genes
                         Values should be in range [0, total_vocab_size)
        """
        if gene_indices.device != self.selected_gene_indices.device:
            gene_indices = gene_indices.to(self.selected_gene_indices.device)
        
        # Update buffer
        if len(gene_indices) <= len(self.selected_gene_indices):
            self.selected_gene_indices[:len(gene_indices)] = gene_indices
        else:
            self.register_buffer('selected_gene_indices', gene_indices)
        
    def expression_to_bins(self, expr):
        """Convert continuous expression to discrete bins"""
        bins = (expr * self.num_bins).long()
        bins = torch.clamp(bins, 0, self.num_bins - 1)
        return bins
        
    def forward(self, expr, coords):
        """
        Forward pass
        
        Args:
            expr: (B, K) - Expression values of K selected genes
                  K should match len(selected_gene_indices)
            coords: (B, 2) - Normalized spatial coordinates
        
        Returns:
            (B, embed_dim) - Spot-level ST embeddings
        """
        B, K = expr.shape
        device = expr.device
        
        # Gene ID Embedding
        gene_ids = self.selected_gene_indices[:K].unsqueeze(0).expand(B, -1)
        gene_id_embeds = self.gene_embedding(gene_ids)  # (B, K, embed_dim)

        # Positional Embedding 
        pos_ids = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
        gene_pos_embeds = self.gene_pos_embedding(pos_ids)  # (B, K, embed_dim)

        # Value Embedding
        if self.use_value_binning:
            bins = self.expression_to_bins(expr)
            value_embeds = self.value_embedding(bins)
        else:
            value_embeds = self.value_embedding(expr.unsqueeze(-1))

        # Combine: scBERT style
        gene_embeds = gene_id_embeds + gene_pos_embeds + value_embeds

        # Spatial Token
        spatial_emb = self.spatial_embed(coords).unsqueeze(1)  # (B, 1, embed_dim)

        # Concatenate: [Spatial Token, Gene Tokens]
        tokens = torch.cat([spatial_emb, gene_embeds], dim=1)  # (B, 1+K, embed_dim)

        # Performer
        tokens = self.performer(tokens)

        # Spatial-Query Pooling
        spatial_out = tokens[:, :1]  # (B, 1, embed_dim)
        gene_out = tokens[:, 1:]     # (B, K, embed_dim)

        q = self.query_proj(spatial_out)
        k = self.key_proj(gene_out)
        v = self.value_proj(gene_out)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5),
            dim=-1
        )

        pooled = torch.matmul(attn, v).squeeze(1)  # (B, embed_dim)
        
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
            nn.Dropout(0.2)
        )
        
    def forward(self, img_feat, st_feat):
        combined = torch.cat([img_feat, st_feat], dim=-1)
        return self.fusion(combined)

# -------------------------------------------------------
# 4. MIL Attention Pooling
# -------------------------------------------------------
class MILAttentionPooling(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=128):
        super().__init__()
        
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.attention_w = nn.Linear(hidden_dim, 1)
        
    def forward(self, spot_embeds):
        """
        Args:
            spot_embeds: (N_spots, embed_dim)
        Returns:
            wsi_embed: (embed_dim,)
            attn_weights: (N_spots, 1)
        """
        A_V = self.attention_V(spot_embeds)
        A_U = self.attention_U(spot_embeds)
        A = self.attention_w(A_V * A_U)
        
        attn_weights = F.softmax(A, dim=0)
        wsi_embed = torch.sum(attn_weights * spot_embeds, dim=0)
        
        return wsi_embed, attn_weights

# -------------------------------------------------------
# 5. Full Model
# -------------------------------------------------------
class MultiModalMILModel(nn.Module):
    """
    Multi-modal MIL model for WSI classification
    
    Pipeline:
    1. Encode each spot (image + ST with spatial info)
    2. Fuse modalities at spot level
    3. Aggregate spots to WSI level using MIL
    4. Classify WSI
    """
    def __init__(self, total_vocab_size, num_classes=2, embed_dim=256,
                 top_k_genes=512, use_value_binning=True, num_bins=5):
        super().__init__()
        
        self.img_encoder = ImageEncoder(embed_dim=embed_dim)
        
        self.st_encoder = SpatialSTEncoder(
            total_vocab_size=total_vocab_size,
            embed_dim=embed_dim,
            num_layers=2,
            num_heads=4,
            top_k_genes=top_k_genes,
            use_value_binning=use_value_binning,
            num_bins=num_bins
        )
        
        self.fusion = SpotFusionModule(embed_dim=embed_dim)
        self.mil_pooling = MILAttentionPooling(embed_dim=embed_dim, hidden_dim=128)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, expr, coords):
        """
        Args:
            images: (N_spots, 3, 224, 224)
            expr: (N_spots, K) - K = number of HVG genes
            coords: (N_spots, 2)
        
        Returns:
            logits: (num_classes,)
            attn_weights: (N_spots, 1)
        """
        # Spot-level encoding
        img_features = self.img_encoder(images)
        st_features = self.st_encoder(expr, coords)
        
        # Spot-level fusion
        spot_embeds = self.fusion(img_features, st_features)
        
        # WSI-level aggregation
        wsi_embed, attn_weights = self.mil_pooling(spot_embeds)
        
        # Classification
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
