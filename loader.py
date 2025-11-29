import os
import json
import h5py
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------------------------------------
# 0) Custom Sample class (manual loader instead of iter_hest)
# -------------------------------------------------------
class CustomSample:
    def __init__(self, root, sample_id):
        self.sample_id = sample_id
        
        st_path = os.path.join(root, sample_id, "st", "st.h5ad")
        patch_path = os.path.join(root, sample_id, "patches", "patches.h5")
        meta_path = os.path.join(root, sample_id, "metadata", "metadata.json")

        # 1. Load ST data (.h5ad)
        if not os.path.exists(st_path):
            raise FileNotFoundError(f"{st_path} not found.")
        
        self.adata = sc.read_h5ad(st_path)

        # Make gene names unique (important for merging)
        self.adata.var_names_make_unique()

        # 2. Load patch image data (.h5)
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"{patch_path} not found.")
        self.patches = h5py.File(patch_path, "r")

        # 3. Load metadata (.json)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

# -------------------------------------------------------
# 1) PyTorch Dataset wrapper for HEST sample
# -------------------------------------------------------
class HESTDataset(Dataset):
    def __init__(self, sample, use_label=True):
        self.sample = sample
        self.use_label = use_label

        # ----- Load ST expression matrix -----
        adata = sample.adata
        if hasattr(adata.X, "toarray"):  # convert sparse to dense if needed
            self.expr = adata.X.toarray()
        else:
            self.expr = adata.X

        self.barcodes_st = adata.obs_names.to_numpy()
        self.coords_st = adata.obsm["spatial"]

        # ----- Load patch image and patch-level metadata -----
        patches = sample.patches
        self.imgs = patches["img"]
        raw_bar = patches["barcode"]
        self.patch_coords = patches["coords"]

        # Decode barcode from bytes → string
        raw_bar_np = np.array(raw_bar)
        patch_barcodes = [
            b.decode() if isinstance(b, bytes) else str(b)
            for b in raw_bar_np.squeeze()
        ]
        self.b2i = {b: i for i, b in enumerate(patch_barcodes)}

        # ----- Align ST barcodes with patch barcodes -----
        valid_indices = []
        valid_st_indices = []
        for i, b in enumerate(self.barcodes_st):
            if b in self.b2i:  # keep only matching barcodes
                valid_indices.append(self.b2i[b])
                valid_st_indices.append(i)

        self.patch_idx = np.array(valid_indices)
        self.expr = self.expr[valid_st_indices]
        self.coords_st = self.coords_st[valid_st_indices]
        self.barcodes_st = self.barcodes_st[valid_st_indices]

        # ----- Assign label (Healthy / Tumor) -----
        if use_label:
            meta = sample.metadata
            if "disease_state" in meta:
                disease = meta["disease_state"]
                self.label_map = {"Healthy": 0, "Tumor": 1, "Cancer": 1}
                self.label = self.label_map.get(disease, 0)
            else:
                self.label = 0

    def __len__(self):
        return len(self.barcodes_st)

    def __getitem__(self, idx):
        # Load image patch
        img = self.imgs[self.patch_idx[idx]]
        img_tensor = torch.tensor(img).permute(2,0,1).float() / 255.

        # Expression vector
        expr = self.expr[idx]
        expr_tensor = torch.tensor(expr).float()

        # Spatial coordinates
        coord = self.patch_coords[self.patch_idx[idx]]
        coord_tensor = torch.tensor(coord).float()

        if self.use_label:
            return {
                "image": img_tensor,
                "expr": expr_tensor,
                "coord": coord_tensor,
                "label": torch.tensor(self.label).long(),
            }
        else:
            return {
                "image": img_tensor,
                "expr": expr_tensor,
                "coord": coord_tensor,
            }

# -------------------------------------------------------
# 2) Create PyTorch DataLoader for multiple samples
# -------------------------------------------------------
def create_hest_dataloader(samples, batch_size=4, shuffle=True):
    datasets = [HESTDataset(s) for s in samples]
    concat_ds = torch.utils.data.ConcatDataset(datasets)
    loader = DataLoader(
        concat_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,   # h5py requires single-process loading
        pin_memory=True
    )
    return loader

# -------------------------------------------------------
# 3) Manual execution for sanity check
# -------------------------------------------------------
if __name__ == "__main__":
    root = "/workspace/Temp"
    id_list = ["TENX24", "TENX39", "TENX97", "MISC61", "TENX153"]

    print("1. Loading samples manually...")
    samples = []
    for sample_id in id_list:
        try:
            s = CustomSample(root, sample_id)
            samples.append(s)
            print(f" - Loaded {sample_id} (Genes: {s.adata.shape[1]})")
        except Exception as e:
            print(f" - Failed to load {sample_id}: {e}")

    # -------------------------------------------------------
    # Align gene set across all samples (intersection)
    # -------------------------------------------------------
    if len(samples) > 0:
        print("\n2. Aligning genes across samples...")

        common_genes = set(samples[0].adata.var_names)
        for s in samples[1:]:
            common_genes = common_genes.intersection(set(s.adata.var_names))

        common_genes = sorted(list(common_genes))
        print(f" -> Found {len(common_genes)} common genes across all samples.")

        if len(common_genes) == 0:
            raise ValueError("No common genes found! Check data processing.")

        for s in samples:
            s.adata = s.adata[:, common_genes]

        # -------------------------------------------------------
        # Create DataLoader
        # -------------------------------------------------------
        print("\n3. Creating DataLoader...")
        loader = create_hest_dataloader(samples, batch_size=4)

        print("Checking first batch...")
        try:
            batch = next(iter(loader))
            print("Image shape:", batch["image"].shape)
            print("Expr shape :", batch["expr"].shape)
            print("Coord shape:", batch["coord"].shape)
            print("Labels     :", batch["label"])
            print("Success!")
        except Exception as e:
            print("Error during iteration:", e)
            import traceback
            traceback.print_exc()
    else:
        print("No samples loaded.")
