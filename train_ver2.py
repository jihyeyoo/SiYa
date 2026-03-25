import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np
import json

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from dataset.loader import CustomSample, create_wsi_dataloader
from models.model_2 import MultiModalMILModel


# ===============================================
# Utils
# ===============================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def plot_confusion_matrix(cm, class_names=('Healthy', 'Cancer'), title="Confusion Matrix", save_path=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


# ===============================================
# Data Loading
# ===============================================
def load_samples_from_split(split_file, root_dir, metadata_dir):
    with open(split_file) as f:
        sample_ids = [line.strip() for line in f]

    samples = []
    for sid in sample_ids:
        try:
            sample = CustomSample(root_dir, sid, metadata_dir=metadata_dir)
            if sample.label in [0, 1]:
                samples.append(sample)
        except Exception as e:
            print(f"⚠️  Failed to load {sid}: {e}")

    return samples


def load_global_hvg(fold_dir):
    hvg_file = fold_dir / "global_hvg.txt"

    if not hvg_file.exists():
        raise FileNotFoundError(f"Global HVG file not found: {hvg_file}")

    with open(hvg_file) as f:
        global_hvg = [line.strip() for line in f if line.strip()]

    print(f"✓ Loaded {len(global_hvg)} pre-computed global HVGs")
    return global_hvg


# ===============================================
# Training
# ===============================================
def train_epoch(model, loader, criterion, optimizer, scaler, config, device):
    model.train()
    if config["freeze_image_encoder"]:
        model.img_encoder.eval()

    epoch_loss = 0.0
    correct = 0
    optimizer.zero_grad()

    loop = tqdm(loader, desc="Training")

    for step, batch in enumerate(loop):
        images = batch["images"].to(device)
        expr   = batch["expr"].to(device)
        coords = batch["coords"].to(device)
        label  = batch["label"].to(device)

        N = images.size(0)
        # train_epoch() 안에서

        spot_embeds_list = []

        for i in range(0, N, config["batch_spots"]):
            j = min(i + config["batch_spots"], N)

            img_b   = images[i:j]
            expr_b  = expr[i:j]
            coord_b = coords[i:j]

            with autocast():
                if config["freeze_image_encoder"]:
                    with torch.no_grad():
                        img_feat = model.img_encoder(img_b)
                else:
                    img_feat = model.img_encoder(img_b)

                img_feat = model.img_head(img_feat)
                st_feat  = model.st_encoder(expr_b, coord_b, return_gene_attn=False)
                fused    = model.fusion(img_feat, st_feat)

            spot_embeds_list.append(fused.detach().cpu())

            del img_b, expr_b, coord_b, img_feat, st_feat, fused
            torch.cuda.empty_cache()


        spot_embeds = torch.cat(spot_embeds_list, dim=0).to(device)

        with autocast():
            wsi_embed, _ = model.mil_pooling(spot_embeds)
            logits = model.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
            loss   = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss   = loss / config["accum_steps"]

        scaler.scale(loss).backward()

        if (step + 1) % config["accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * config["accum_steps"]
        correct    += int(logits.argmax().item() == label.item())

        loop.set_postfix(
            loss=f"{epoch_loss / (step + 1):.4f}",
            acc=f"{100 * correct / (step + 1):.1f}%"
        )

        del spot_embeds, wsi_embed, logits, loss, spot_embeds_list
        torch.cuda.empty_cache()

    return epoch_loss / len(loader), 100 * correct / len(loader)


# ===============================================
# Validation
# ===============================================
@torch.no_grad()
def validate(model, loader, criterion, config, device, save_embeddings=False):
    """
    Validate model.

    Args:
        save_embeddings: If True, collect both WSI-level and spot-level embeddings.

    Embedding shapes:
        wsi_embeds   : (N_samples, D)          — stacked, shape fixed
        spot_embeds  : list of (N_spots_i, D)  — variable N_spots per sample
        attn_weights : list of (N_spots_i, 1)  — MIL attention per spot
        labels / preds / scores : (N_samples,)
        sample_ids   : list of str

    Returns:
        (metrics_tuple, embeddings_dict or None)
    """
    model.eval()

    val_loss = 0.0
    correct  = 0
    y_true, y_score, y_pred = [], [], []

    wsi_embeds_list   = []
    spot_embeds_all   = []   # list of (N_spots_i, D)
    attn_weights_list = []
    sample_ids_list   = []

    for batch in tqdm(loader, desc="Validation"):
        images    = batch["images"].to(device)
        expr      = batch["expr"].to(device)
        coords    = batch["coords"].to(device)
        label     = batch["label"].to(device)
        sample_id = batch.get("sample_id", "unknown")

        # ── Chunked spot encoding ──────────────────────────────
        spot_embeds_chunks = []
        for i in range(0, images.size(0), config["batch_spots"]):
            j = min(i + config["batch_spots"], images.size(0))

            with autocast():
                img_feat = model.img_encoder(images[i:j])
                img_feat = model.img_head(img_feat)
                st_feat  = model.st_encoder(expr[i:j], coords[i:j], return_gene_attn=False)
                fused    = model.fusion(img_feat, st_feat)

            spot_embeds_chunks.append(fused.cpu())

        # (N_spots, D) on CPU — full sample
        spot_embeds = torch.cat(spot_embeds_chunks, dim=0)

        # ── WSI pooling & classification ───────────────────────
        with autocast():
            wsi_embed, attn = model.mil_pooling(spot_embeds.to(device))
            logits = model.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
            loss   = criterion(logits.unsqueeze(0), label.unsqueeze(0))

        val_loss += loss.item()
        pred      = logits.argmax().item()
        correct  += int(pred == label.item())

        prob_pos = torch.softmax(logits, dim=0)[1].item()
        y_true.append(label.item())
        y_score.append(prob_pos)
        y_pred.append(pred)

        if save_embeddings:
            wsi_embeds_list.append(wsi_embed.cpu().float().numpy())  # (D,)
            spot_embeds_all.append(spot_embeds.float().numpy())      # (N_spots, D)
            attn_weights_list.append(attn.cpu().float().numpy())     # (N_spots, 1)
            sample_ids_list.append(sample_id)

    val_loss /= len(loader)
    val_acc   = 100 * correct / len(loader)

    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = float('nan')

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = (val_loss, val_acc, auc, float(p), float(r), float(f1), cm)

    if save_embeddings:
        embeddings = {
            "wsi_embeds":   np.stack(wsi_embeds_list, axis=0),  # (N, D)
            "spot_embeds":  spot_embeds_all,                     # list of (N_spots_i, D)
            "attn_weights": attn_weights_list,                   # list of (N_spots_i, 1)
            "labels":       np.array(y_true),
            "preds":        np.array(y_pred),
            "scores":       np.array(y_score),
            "sample_ids":   sample_ids_list,
        }
        return metrics, embeddings

    return metrics, None


# ===============================================
# Save Embeddings
# ===============================================
def save_embeddings_to_disk(embeddings, output_path, epoch, split="val"):
    """
    WSI embeddings  → embeddings_{split}_epoch{N:02d}.npz   (fixed shape)
    Spot embeddings → spot_embeds_{split}_epoch{N:02d}.npy  (object array)
    Attn weights    → attn_weights_{split}_epoch{N:02d}.npy (object array)

    Load example:
        # WSI level
        data       = np.load("embeddings_val_best_epoch04.npz", allow_pickle=True)
        wsi_embeds = data['wsi_embeds']    # (N, D)
        labels     = data['labels']
        sample_ids = data['sample_ids']

        # Spot level
        spots = np.load("spot_embeds_val_best_epoch04.npy", allow_pickle=True)
        attn  = np.load("attn_weights_val_best_epoch04.npy", allow_pickle=True)
        # spots[i].shape → (N_spots_of_sample_i, D)
        # attn[i].shape  → (N_spots_of_sample_i, 1)
    """
    embed_path = output_path / f"embeddings_{split}_epoch{epoch:02d}.npz"
    np.savez(
        embed_path,
        wsi_embeds=embeddings["wsi_embeds"],
        labels=embeddings["labels"],
        preds=embeddings["preds"],
        scores=embeddings["scores"],
        sample_ids=np.array(embeddings["sample_ids"], dtype=object),
    )

    spot_path = output_path / f"spot_embeds_{split}_epoch{epoch:02d}.npy"
    np.save(
        spot_path,
        np.array(embeddings["spot_embeds"], dtype=object),
        allow_pickle=True
    )

    attn_path = output_path / f"attn_weights_{split}_epoch{epoch:02d}.npy"
    np.save(
        attn_path,
        np.array(embeddings["attn_weights"], dtype=object),
        allow_pickle=True
    )

    n, d = embeddings["wsi_embeds"].shape
    print(f"  ✓ WSI embeds  : {embed_path.name}  ({n} samples, D={d})")
    print(f"  ✓ Spot embeds : {spot_path.name}  [{n} samples, N_spots varies]")
    print(f"  ✓ Attn weights: {attn_path.name}")


# ===============================================
# Main Training Function
# ===============================================
def train_fold(
    dataset_name,
    fold_idx,
    fusion_option,
    root_dir,
    split_dir="./data_splits",
    output_dir="./hest_ver2_results",
    metadata_dir="./hest_data/metadata",
    config=None
):
    if config is None:
        config = {}

    config.setdefault('num_genes', 2000)
    config.setdefault('num_classes', 2)
    config.setdefault('embed_dim', 256)
    config.setdefault('top_k_genes', 512)
    config.setdefault('epochs', 10)
    config.setdefault('lr', 3e-5)
    config.setdefault('weight_decay', 1e-3)
    config.setdefault('batch_size', 1)
    config.setdefault('batch_spots', 500)
    config.setdefault('accum_steps', 4)
    config.setdefault('freeze_image_encoder', True)
    config.setdefault('max_spots', 300)
    config.setdefault('device', 'cuda')
    config.setdefault('seed', 42)
    config.setdefault('embed_save_epochs', [])  # 0-indexed; last epoch always saved

    set_seed(config['seed'])
    device = torch.device(config['device'])

    fold_dir    = Path(split_dir) / dataset_name / f"fold_{fold_idx}"
    output_path = Path(output_dir) / dataset_name / f"fold_{fold_idx}" / "ver2" / fusion_option
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Training: {dataset_name} / Fold {fold_idx} / {fusion_option}")
    print('='*80)

    # ── Load samples ──────────────────────────────────────────────
    print("\nLoading samples...")
    train_samples = load_samples_from_split(fold_dir / "train.txt", root_dir, metadata_dir)
    val_samples   = load_samples_from_split(fold_dir / "val.txt",   root_dir, metadata_dir)
    print(f"✓ Train: {len(train_samples)} | Val: {len(val_samples)}")

    hvg_genes = load_global_hvg(fold_dir)
    with open(output_path / "hvg_genes_used.txt", "w") as f:
        f.write("\n".join(hvg_genes))

    # ── Class weights ─────────────────────────────────────────────
    from collections import Counter
    train_labels = [s.label for s in train_samples]
    label_counts = Counter(train_labels)
    print(f"\nLabel distribution: {dict(label_counts)}")

    n_samples     = len(train_labels)
    class_weights = torch.tensor([
        n_samples / (2 * label_counts[0]),
        n_samples / (2 * label_counts[1])
    ], dtype=torch.float).to(device)
    print(f"Class weights: Healthy={class_weights[0]:.2f}, Cancer={class_weights[1]:.2f}")

    # ── Dataloaders ───────────────────────────────────────────────
    train_loader = create_wsi_dataloader(
        train_samples, batch_size=1, shuffle=True,
        max_spots=config['max_spots'], hvg_genes=hvg_genes
    )
    val_loader = create_wsi_dataloader(
        val_samples, batch_size=1, shuffle=False,
        max_spots=config['max_spots'], hvg_genes=hvg_genes
    )

    # ── Model ─────────────────────────────────────────────────────
    model = MultiModalMILModel(
        num_genes=config['num_genes'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        fusion_option=fusion_option,
        top_k_genes=config['top_k_genes'],
        freeze_image_encoder=config['freeze_image_encoder'],
        use_image=True,
        use_st=True,
    ).to(device)

    if config['freeze_image_encoder']:
        for p in model.img_encoder.parameters():
            p.requires_grad = False
        model.img_encoder.eval()

    # Partially freeze ST encoder (first 1 of 2 Performer layers)
    for i, layer in enumerate(model.st_encoder.transformer.net.layers):
        if i < 1:
            for p in layer.parameters():
                p.requires_grad = False
    print(f"✓ Frozen ST encoder layer 0/{len(model.st_encoder.transformer.net.layers)}")

    # ── Optimizer & criterion ─────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler    = GradScaler()

    # ── Training loop (fixed epochs, no early stopping) ───────────
    best_val_acc  = 0.0
    total_epochs  = config['epochs']

    embed_save_epochs = set(config['embed_save_epochs'])
    embed_save_epochs.add(total_epochs - 1)  # always save last epoch

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [], 'val_auc': [],
    }

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config, device
        )

        save_embed = epoch in embed_save_epochs
        metrics, embeddings = validate(
            model, val_loader, criterion, config, device,
            save_embeddings=save_embed
        )
        val_loss, val_acc, val_auc, val_p, val_r, val_f1, cm = metrics

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f"Train : Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val   : Loss={val_loss:.4f},  Acc={val_acc:.2f}%, AUC={val_auc:.4f}")
        print(f"P/R/F1: {val_p:.3f}/{val_r:.3f}/{val_f1:.3f}")

        # ── Best model ────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / "best_model.pt")
            plot_confusion_matrix(
                cm, title=f"Val CM (Epoch {epoch + 1})",
                save_path=output_path / f"confusion_matrix_epoch_{epoch + 1}.png"
            )
            print(f"✓ Saved best model (val_acc={val_acc:.2f}%)")

            # Best-epoch embeddings
            if embeddings is None:
                _, embeddings_best = validate(
                    model, val_loader, criterion, config, device,
                    save_embeddings=True
                )
            else:
                embeddings_best = embeddings
            save_embeddings_to_disk(embeddings_best, output_path, epoch, split="val_best")

        # Designated-epoch embeddings
        if save_embed and embeddings is not None:
            save_embeddings_to_disk(embeddings, output_path, epoch, split="val")

    # ── Save history & results ────────────────────────────────────
    with open(output_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    results = {
        'dataset': dataset_name,
        'fold': fold_idx,
        'fusion': fusion_option,
        'best_val_acc': best_val_acc,
        'final_metrics': {
            'val_loss': val_loss, 'val_acc': val_acc,
            'val_auc': val_auc,   'precision': val_p,
            'recall': val_r,      'f1': val_f1,
        }
    }

    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Fold {fold_idx} done!  Best val_acc: {best_val_acc:.2f}%")
    return results


# ===============================================
# Run All Experiments
# ===============================================
def main():
    config = {
        'num_genes': 2000,
        'num_classes': 2,
        'embed_dim': 256,
        'top_k_genes': 512,
        'epochs': 10,           # fixed 10 epochs, no early stopping
        'lr': 0.0005,           # 5e-4 → 1e-4 (safer)
        'weight_decay': 0.0001,
        'batch_size': 1,
        'batch_spots': 500,
        'accum_steps': 4,
        'freeze_image_encoder': True,
        'max_spots': 300,
        'device': 'cuda',
        'seed': 42,
        'embed_save_epochs': [],  # epoch 9 (last) always saved automatically
    }

    dataset_name   = "HEST"
    root_dir       = "./hest_data/"
    metadata_dir   = "./hest_data/metadata"
    fusion_options = ['concat', 'attn', 'sim', 'gate']
    n_folds        = 5

    all_results = []

    for fusion_opt in fusion_options:
        for fold_idx in range(n_folds):
            print(f"\n{'#'*80}")
            print(f"HEST / Fold {fold_idx} / {fusion_opt}")
            print('#'*80)

            results = train_fold(
                dataset_name=dataset_name,
                fold_idx=fold_idx,
                fusion_option=fusion_opt,
                root_dir=root_dir,
                split_dir="./data_splits",
                output_dir="./hest_ver2_results",
                metadata_dir=metadata_dir,
                config=config,
            )
            all_results.append(results)

    output_file = "./hest_ver2_results/HEST_ver2_all_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("✓ ALL EXPERIMENTS COMPLETE!")
    print(f"✓ Results saved to: {output_file}")
    print('='*80)


if __name__ == "__main__":
    main()
