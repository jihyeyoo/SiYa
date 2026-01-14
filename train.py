import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from loader import CustomSample, create_wsi_dataloader
from model import MultiModalMILModel

# -------------------------------------------------------
# Configuration (Overfitting-oriented setup)
# -------------------------------------------------------
CONFIG = {
    "root_dir": "/workspace/Temp/ver1/hest_data",
    "epochs": 100,           # Train long enough to fully memorize
    "lr": 2e-4,             # Slightly increased learning rate
    "embed_dim": 64,
    "num_genes": 2000,      # (Will be replaced by aligned gene size)
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Memory & Sampling
    "chunk_size": 16,      # Chunk size
    "max_spots": 16,       # Increase spots to capture more information (if memory allows)
}

# -------------------------------------------------------
# Gene alignment (Union strategy - required)
# -------------------------------------------------------
def align_genes_union(samples):
    print("Aligning genes (Union)...")
    all_genes = set()
    for s in samples:
        all_genes.update(s.adata.var_names)
    union_genes = sorted(list(all_genes))
    
    for s in samples:
        orig_df = pd.DataFrame(
            s.adata.X.toarray() if hasattr(s.adata.X, 'toarray') else s.adata.X,
            index=s.adata.obs_names,
            columns=s.adata.var_names
        )
        # Fill missing genes with zeros
        new_df = orig_df.reindex(columns=union_genes, fill_value=0.0)
        
        # Overwrite AnnData while preserving metadata
        import scanpy as sc
        new_adata = sc.AnnData(X=new_df.values, obs=s.adata.obs)
        new_adata.var_names = union_genes
        new_adata.obsm = s.adata.obsm
        s.adata = new_adata
        
    return len(union_genes)

# -------------------------------------------------------
# Training Function
# -------------------------------------------------------
def train_overfit(cfg):
    # Fix random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = cfg["device"]
    print(f"Using device: {device}")

    # 1. Load and select samples (2 Tumor, 2 Healthy)
    st_dir = os.path.join(cfg["root_dir"], "st_preprocessed")
    all_files = [f for f in os.listdir(st_dir) if f.endswith(".h5ad")]
    
    samples = []
    tumor_cnt, healthy_cnt = 0, 0
    
    print("\nLoading Samples for Overfitting...")
    for fname in all_files:
        if tumor_cnt >= 2 and healthy_cnt >= 2:
            break
            
        sid = fname.replace(".h5ad", "")
        try:
            s = CustomSample(cfg["root_dir"], sid)
            label = s.adata.obs['disease_state'].values[0]
            
            # Collect 2 Tumor (1) and 2 Healthy (0) samples
            if label == 1 and tumor_cnt < 2:
                samples.append(s)
                tumor_cnt += 1
                print(f" + [Tumor] {sid} loaded")
            elif label == 0 and healthy_cnt < 2:
                samples.append(s)
                healthy_cnt += 1
                print(f" + [Normal] {sid} loaded")
                
        except Exception as e:
            print(f"Skip {sid}: {e}")

    if not samples:
        print("Error: No samples loaded.")
        return

    # 2. Gene alignment (Union)
    real_num_genes = align_genes_union(samples)
    print(f"Feature aligned! Model input gene dim: {real_num_genes}")

    # 3. DataLoader (shuffle can be enabled even for overfitting)
    loader = create_wsi_dataloader(
        samples, 
        batch_size=1, 
        shuffle=True,  # Even for overfitting, shuffling is fine (only 4 samples)
        max_spots=cfg["max_spots"]
    )

    # 4. Model initialization
    model = MultiModalMILModel(
        num_genes=real_num_genes,  # Aligned gene dimension
        num_classes=cfg["num_classes"],
        embed_dim=cfg["embed_dim"],
    ).to(device)

    # Disable weight decay for overfitting (no regularization)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    # 5. Training loop
    print("\n=== Start Overfitting Test ===")
    
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # tqdm progress bar
        loop = tqdm(loader, desc=f"Ep {epoch+1}/{cfg['epochs']}", leave=False)
        
        for batch in loop:
            images = batch["images"].to(device)
            expr   = batch["expr"].to(device)
            coords = batch["coords"].to(device)
            label  = batch["label"].unsqueeze(0).to(device)  # Shape [1]

            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(images, expr, coords)  # [num_classes]
            
            # Loss computation
            loss = criterion(logits.unsqueeze(0), label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax().item()
            correct += (pred == label.item())
            total += 1
            
            loop.set_postfix(loss=loss.item(), acc=100*correct/total)

        avg_loss = total_loss / total
        avg_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | Acc: {avg_acc:.1f}%")

        # Early stop if perfectly overfitted with very low loss
        if avg_acc == 100.0 and avg_loss < 0.01:
            print("\n🎉 SUCCESS! Model successfully overfitted.")
            break

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_overfit(CONFIG)
