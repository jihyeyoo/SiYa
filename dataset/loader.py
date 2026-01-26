import os
import h5py
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc

# -------------------------------------------------------
# Global gene order loader
# -------------------------------------------------------
def load_global_gene_order(root_dir):
    """Load global HVG gene list to ensure consistent gene ordering"""
    hvg_path = os.path.join(root_dir, 'global_hvg_genes_unified.txt')
    
    if os.path.exists(hvg_path):
        with open(hvg_path, 'r') as f:
            global_genes = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded global gene order: {len(global_genes)} genes")
        return global_genes
    else:
        print("⚠️  global_hvg_genes.txt not found, will infer from first sample")
        return None


# -------------------------------------------------------
# Custom Sample class
# -------------------------------------------------------
class CustomSample:
    def __init__(self, root, sample_id):
        self.sample_id = sample_id
        self.st_path = os.path.join(root, "global_hvg_unified", f"{sample_id}.h5ad")
        self.patch_path = os.path.join(root, "patches", f"{sample_id}.h5")
        
        if not os.path.exists(self.st_path):
            raise FileNotFoundError(f"{self.st_path} not found.")
        if not os.path.exists(self.patch_path):
            raise FileNotFoundError(f"{self.patch_path} not found.")

        self.label = self._load_label()
    
    def _load_label(self):
        """Extract sample-level label from h5ad obs['disease_state']"""
        adata = sc.read_h5ad(self.st_path, backed='r')
        val = adata.obs['disease_state'].values[0]
        del adata
        
        import pandas as pd
        if pd.isna(val):
            return 0
        return int(val)


# -------------------------------------------------------
# WSI-level Dataset (unified gene order + zero-padding)
# -------------------------------------------------------
class WSIDataset(Dataset):
    def __init__(self, samples, max_spots=2000, global_gene_order=None):
        self.samples = samples
        self.max_spots = max_spots
        self.global_gene_order = global_gene_order
        
        # Fallback: infer gene order from the first sample
        if self.global_gene_order is None:
            print("Inferring gene order from first sample...")
            adata = sc.read_h5ad(samples[0].st_path, backed='r')
            self.global_gene_order = adata.var_names.tolist()
            del adata
            print(f"✓ Using {len(self.global_gene_order)} genes as reference order")
    
    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        sample = self.samples[idx]
        adata = None
        
        try:
            # 1. Load AnnData
            adata = sc.read_h5ad(sample.st_path, backed='r')
            
            # Gene mapping
            sample_genes = adata.var_names.tolist()
            sample_gene_set = set(sample_genes)
            gene_to_idx = {g: i for i, g in enumerate(sample_genes)}
            
            # Load expression matrix
            X_raw = adata.X[:]
            
            # Convert sparse matrix to dense
            if hasattr(X_raw, "toarray"):
                full_expr = X_raw.toarray()
            elif hasattr(X_raw, "todense"):
                full_expr = np.array(X_raw.todense())
            else:
                full_expr = np.asarray(X_raw)
            
            # Shape validation
            if full_expr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {full_expr.shape}")
            
            n_spots = full_expr.shape[0]
            
            # Zero-padded aligned expression matrix
            expr_aligned = np.zeros((n_spots, len(self.global_gene_order)), dtype=np.float32)
            
            for new_idx, gene in enumerate(self.global_gene_order):
                if gene in sample_gene_set:
                    old_idx = gene_to_idx[gene]
                    expr_aligned[:, new_idx] = full_expr[:, old_idx]
            
            barcodes_st = adata.obs_names.to_numpy()
            coords = np.array(adata.obsm["spatial"])
            label_val = sample.label
            
            del adata, full_expr, X_raw
            adata = None
            
            # 2. Load patches
            with h5py.File(sample.patch_path, "r") as f:
                imgs = f["img"][:]
                raw_bar = np.array(f["barcode"])
            
            # Barcode handling
            if raw_bar.ndim == 0:
                patch_barcodes = [raw_bar.item().decode() if isinstance(raw_bar.item(), bytes) else str(raw_bar.item())]
            elif raw_bar.ndim == 1:
                patch_barcodes = [
                    b.decode() if isinstance(b, bytes) else str(b)
                    for b in raw_bar
                ]
            elif raw_bar.ndim == 2:
                if raw_bar.shape[1] == 1:
                    patch_barcodes = []
                    for i in range(len(raw_bar)):
                        item = raw_bar[i, 0]
                        if isinstance(item, (bytes, np.bytes_)):
                            patch_barcodes.append(item.decode())
                        else:
                            patch_barcodes.append(str(item))
                else:
                    raw_bar_flat = raw_bar.flatten()
                    patch_barcodes = [
                        b.decode() if isinstance(b, (bytes, np.bytes_)) else str(b)
                        for b in raw_bar_flat
                    ]
            else:
                raise ValueError(f"Unexpected barcode shape: {raw_bar.shape}")
            
            if len(patch_barcodes) == 0:
                raise ValueError("No patch barcodes found")
            
            b2i = {b: i for i, b in enumerate(patch_barcodes)}
            
            patch_idx, st_idx = [], []
            for i, b in enumerate(barcodes_st):
                if b in b2i:
                    patch_idx.append(b2i[b])
                    st_idx.append(i)
            
            if len(patch_idx) == 0:
                raise ValueError("No aligned spots")
            
            images = imgs[patch_idx]
            expr = expr_aligned[st_idx]
            coords = coords[st_idx]
            
            # 3. Spot sampling
            if self.max_spots and len(images) > self.max_spots:
                sel = np.random.choice(len(images), self.max_spots, replace=False)
                images, expr, coords = images[sel], expr[sel], coords[sel]
            
            # 4. Tensor conversion
            images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.
            expr = torch.from_numpy(expr).float()
            coords = torch.from_numpy(coords).float()
            
            # 5. Coordinate normalization
            if coords.shape[0] > 1:
                c_min = coords.min(dim=0, keepdim=True).values
                c_max = coords.max(dim=0, keepdim=True).values
                c_range = c_max - c_min
                c_range[c_range == 0] = 1.0
                coords = (coords - c_min) / c_range
            elif coords.shape[0] == 1:
                coords = torch.tensor([[0.5, 0.5]], dtype=coords.dtype)
            else:
                coords = torch.zeros((0, 2), dtype=torch.float32)
            
            return {
                "images": images,
                "expr": expr,
                "coords": coords,
                "label": torch.tensor(label_val).long(),
                "sample_id": sample.sample_id,
                "num_spots": len(images)
            }
            
        except Exception as e:
            print(f"⚠️ Error loading {sample.sample_id}: {e}")
            import traceback
            print(traceback.format_exc())
            raise
            
        finally:
            if adata is not None:
                try:
                    del adata
                except:
                    pass
            gc.collect()


def wsi_collate_fn(batch):
    if len(batch) == 1:
        return batch[0]
    return batch


def create_wsi_dataloader(samples, batch_size=1, shuffle=True, max_spots=2000, root_dir=None):
    """
    root_dir added for loading global gene order
    """
    global_gene_order = None
    
    if root_dir is not None:
        global_gene_order = load_global_gene_order(root_dir)
    
    dataset = WSIDataset(samples, max_spots=max_spots, global_gene_order=global_gene_order)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=wsi_collate_fn
    )
    return loader


# -------------------------------------------------------
# Gene information utility
# -------------------------------------------------------
def get_gene_info(samples):
    """Retrieve gene information from the first sample"""
    sample = samples[0]
    
    adata = sc.read_h5ad(sample.st_path, backed='r')
    num_genes = adata.n_vars
    gene_names = adata.var_names.tolist()
    del adata
    
    print(f"Gene info:")
    print(f"  Total genes: {num_genes}")
    print(f"  Gene names (first 10): {gene_names[:10]}")
    
    return num_genes, gene_names


# -------------------------------------------------------
# Dataloader validation utility
# -------------------------------------------------------
def validate_dataloader(loader, num_batches=5):
    """Validate dataloader outputs for debugging"""
    print("\n=== Dataloader Validation ===")
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        print(f"\nBatch {i}:")
        print(f"  Sample ID: {batch['sample_id']}")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Expression shape: {batch['expr'].shape}")
        print(f"  Coords shape: {batch['coords'].shape}")
        print(f"  Label: {batch['label'].item()}")
        print(f"  Num spots: {batch['num_spots']}")
        
        # Range checks
        print(f"  Image range: [{batch['images'].min():.3f}, {batch['images'].max():.3f}]")
        print(f"  Expr range: [{batch['expr'].min():.3f}, {batch['expr'].max():.3f}]")
        print(f"  Coord range: [{batch['coords'].min():.3f}, {batch['coords'].max():.3f}]")
        
        # Zero-padding check
        nonzero_genes = (batch['expr'].sum(dim=0) != 0).sum()
        print(f"  Non-zero genes: {nonzero_genes} / {batch['expr'].shape[1]}")
    
    print("\n✓ Validation complete")
