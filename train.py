import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from loader import CustomSample, create_wsi_dataloader
from model import MultiModalMILModel

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
CONFIG = {
    "root_dir": "/workspace/Temp/ver1/hest_data",
    "epochs": 20,
    "lr": 1e-4,
    "embed_dim": 64,
    "num_genes": 2000,   # Fixed number of HVGs
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Memory-related settings
    "chunk_size": 64,
    "max_spots": 50,  # Set small for sanity check (good)
}

# -------------------------------------------------------
# Training
# -------------------------------------------------------
def train(cfg):
    device = cfg["device"]
    print(f"Using device: {device}")

    # ---------------------------------------------------
    # Load samples
    # ---------------------------------------------------
    samples = []

    st_dir = os.path.join(cfg["root_dir"], "st_preprocessed") 
    
    if not os.path.exists(st_dir):
        print(f"Error: Directory not found: {st_dir}")
        return

    # Get list of files
    all_files = [f for f in os.listdir(st_dir) if f.endswith(".h5ad")]
    all_ids = [f.replace(".h5ad", "") for f in all_files]

    target_ids = all_ids[:1] 

    print(f"Target Samples for Sanity Check: {target_ids}")

    for sid in target_ids:
        try:
            samples.append(CustomSample(cfg["root_dir"], sid))
            print(f"✓ Loaded {sid}")
        except Exception as e:
            print(f"✗ Failed {sid}: {e}")

    if len(samples) == 0:
        print("No samples loaded. Exiting.")
        return

    loader = create_wsi_dataloader(
        samples,
        batch_size=1,
        shuffle=True,
        max_spots=cfg["max_spots"]
    )

    # ---------------------------------------------------
    # Model
    # ---------------------------------------------------
    model = MultiModalMILModel(
        num_genes=cfg["num_genes"], 
        num_classes=cfg["num_classes"],
        embed_dim=cfg["embed_dim"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------
    # Training loop
    # ---------------------------------------------------
    print("\nStarting Sanity Check Training...")
    
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss, correct = 0, 0
        total_samples = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")

        for batch in loop:
            images = batch["images"].to(device)
            expr   = batch["expr"].to(device)
            coords = batch["coords"].to(device)
            label  = batch["label"].unsqueeze(0).to(device)

            # Sanity check: skip if the number of spots is zero
            if images.size(0) == 0:
                continue

            optimizer.zero_grad()
            
            # Since max_spots=50, we can feed everything at once without OOM
            logits, attn_weights = model(images, expr, coords)
            
            loss = criterion(logits.unsqueeze(0), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            
            # Accuracy calculation
            pred = logits.argmax().item()
            correct += int(pred == label.item())
            total_samples += 1

            loop.set_postfix(
                loss=total_loss / total_samples,
                acc=100 * correct / total_samples
            )
        
        # Prevent division by zero
        final_acc = 100 * correct / max(1, total_samples)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {final_acc:.2f}%")

    print("✓ Training finished")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train(CONFIG)
