import torch
import torch.nn as nn
import torch.optim as optim
from loader import create_hest_dataloader, CustomSample
from model import MultiModalHestModel
from tqdm import tqdm

def get_samples_by_ids(root, id_list):
    """Return list of CustomSample objects given sample IDs."""
    samples = []
    for sample_id in id_list:
        try:
            s = CustomSample(root, sample_id)
            samples.append(s)
            print(f"[{sample_id}] Loaded.")
        except Exception as e:
            print(f"[{sample_id}] Load Failed: {e}")
    return samples

def main():
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    ROOT = "/workspace/Temp"
    BATCH_SIZE = 8        # Adjust depending on GPU memory
    LEARNING_RATE = 1e-4  # Learning rate
    EPOCHS = 5            # Small number for testing/trial
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using Device: {DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Preparation (Train/Validation Split)
    # ---------------------------------------------------------
    # Training samples + Validation sample
    train_ids = ["TENX24", "TENX39", "MISC61", "TENX153"]
    val_ids = ["TENX97"]
    
    print("\n>>> 1. Loading Data...")
    # Load all samples first to align gene spaces
    all_samples = get_samples_by_ids(ROOT, train_ids + val_ids)
    
    # Find intersection of gene lists across all samples
    if not all_samples:
        print("No samples loaded.")
        return

    common_genes = set(all_samples[0].adata.var_names)
    for s in all_samples[1:]:
        common_genes = common_genes.intersection(set(s.adata.var_names))
    common_genes = sorted(list(common_genes))
    num_genes = len(common_genes)
    
    print(f" -> Common Genes Count: {num_genes}")
    
    # Subset each sample to the same gene set
    for s in all_samples:
        s.adata = s.adata[:, common_genes]

    # Split back into train / validation samples
    train_samples = [s for s in all_samples if s.sample_id in train_ids]
    val_samples = [s for s in all_samples if s.sample_id in val_ids]
    
    # Create dataloaders
    train_loader = create_hest_dataloader(train_samples, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_hest_dataloader(val_samples, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f" -> Train Patches Batches: {len(train_loader)}")
    print(f" -> Valid Patches Batches: {len(val_loader)}")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print(f"\n>>> 2. Initializing Model (Input Genes: {num_genes})...")
    model = MultiModalHestModel(num_genes=num_genes, num_classes=2).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    class_weights = torch.tensor([4.0, 1.0]).to(DEVICE)  # Handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n>>> 3. Start Training Loop...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs = batch["image"].to(DEVICE)
            exprs = batch["expr"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs, exprs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                exprs = batch["expr"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                outputs = model(imgs, exprs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        # Print epoch results
        print("-"*60)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Result:")
        print(f"   Train | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Valid | Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")
        print("-"*60)

    print("\n>>> Training Finished!")
    
    # Save model
    torch.save(model.state_dict(), "hest_model_sanity_check.pth")
    print("Model saved to 'hest_model_sanity_check.pth'")

if __name__ == "__main__":
    main()
