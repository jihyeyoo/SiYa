import torch
import torch.nn as nn
import torchvision.models as models

# -------------------------------------------------------
# 1. Image Encoder (CNN or ViT alternative)
# -------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Use lightweight ResNet18 (can replace with ViT if needed)
        self.backbone = models.resnet18(pretrained=True)
        # Replace final classification layer with embed_dim output
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)
        
    def forward(self, x):
        # x: (B, 3, 224, 224)
        return self.backbone(x)  # -> (B, embed_dim)

# -------------------------------------------------------
# 2. ST Encoder (scBERT-style simplified version)
# -------------------------------------------------------
class STEncoder(nn.Module):
    def __init__(self, num_genes, embed_dim=256):
        super().__init__()
        # Convert raw gene expression vector → embedding
        self.gene_embed = nn.Sequential(
            nn.Linear(num_genes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # Transformer Encoder (core structure of scBERT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final projection layer
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, num_genes)
        x = self.gene_embed(x)  # -> (B, embed_dim)

        # Add sequence dimension for transformer: (B, 1, embed_dim)
        x = x.unsqueeze(1)

        # Apply transformer
        x = self.transformer(x)  # -> (B, 1, embed_dim)

        # Remove sequence dimension
        x = x.squeeze(1)

        return self.fc(x)

# -------------------------------------------------------
# 3. Full Multi-Modal Model (Fusion + Classifier)
# -------------------------------------------------------
class MultiModalHestModel(nn.Module):
    def __init__(self, num_genes, num_classes=2):
        super().__init__()
        
        embed_dim = 256
        
        # Two encoders
        self.img_encoder = ImageEncoder(embed_dim=embed_dim)
        self.st_encoder = STEncoder(num_genes=num_genes, embed_dim=embed_dim)
        
        # Fusion module (Concat + MLP)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final classifier
        self.classifier = nn.Linear(128, num_classes)

        # (Optional) hooks for XAI can be added later

    def forward(self, img, expr):
        # Encode each modality
        img_feat = self.img_encoder(img)      # (B, 256)
        st_feat = self.st_encoder(expr)       # (B, 256)
        
        # Concatenate embeddings
        combined = torch.cat([img_feat, st_feat], dim=1)  # (B, 512)
        fused = self.fusion_layer(combined)               # (B, 128)
        
        # Final prediction
        logits = self.classifier(fused)                   # (B, num_classes)
        
        return logits
