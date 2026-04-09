import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import numpy as np
import pandas as pd
import json

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from dataset.loader import CustomSample, create_wsi_dataloader
from models.model_1 import MultiModalMILModel


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_samples(split_file, st_dir, patch_dir, meta_df):
    """
    Args:
        split_file : path to train.txt / val.txt (one sample_id per line)
        st_dir     : directory with {sample_id}.h5ad
        patch_dir  : directory with {sample_id}.h5
        meta_df    : DataFrame indexed by 'slide' with 'involve_cancer' column
    """
    with open(split_file) as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    samples = []
    for sid in sample_ids:
        try:
            label = int(meta_df.loc[sid, "involve_cancer"])
            sample = CustomSample(sid, st_dir, patch_dir, label)
            samples.append(sample)
        except KeyError:
            print(f"  {sid}: not found in metadata — skipped")
        except Exception as e:
            print(f"  {sid}: {e} — skipped")
    return samples


def load_global_hvg(hvg_path):
    with open(hvg_path) as f:
        genes = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(genes)} global HVGs from {hvg_path}")
    return genes


def train_epoch(model, loader, criterion, optimizer, scaler, config, device):
    model.train()
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
        spot_embeds_list = []

        for i in range(0, N, config["batch_spots"]):
            j = min(i + config["batch_spots"], N)

            with autocast():
                with torch.no_grad():
                    img_feat = model.img_encoder(images[i:j])
                img_feat = model.img_head(img_feat)
                sc_feat  = model.sc_encoder(expr[i:j])
                st_feat  = model.st_encoder(coords[i:j])
                fused    = model.fusion(img_feat, sc_feat, st_feat)

            spot_embeds_list.append(fused.detach().cpu())

            del img_feat, sc_feat, st_feat, fused
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
            loss=f"{epoch_loss/(step+1):.4f}",
            acc=f"{100*correct/(step+1):.1f}%"
        )

        del spot_embeds, wsi_embed, logits, loss, spot_embeds_list
        torch.cuda.empty_cache()

    return epoch_loss / len(loader), 100 * correct / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, config, device, save_embeddings=False):
    model.eval()

    val_loss = 0.0
    correct  = 0
    y_true, y_score, y_pred = [], [], []

    # Per-module embedding collectors
    all_img_feats   = []   # after image encoding
    all_sc_feats    = []   # after sc encoding
    all_st_feats    = []   # after st encoding
    all_fused       = []   # after fusion
    all_wsi_embeds  = []   # after MIL pooling
    all_attn        = []
    all_sample_ids  = []
    all_labels      = []
    all_preds       = []
    all_scores      = []

    for batch in tqdm(loader, desc="Validation"):
        images    = batch["images"]
        expr      = batch["expr"]
        coords    = batch["coords"]
        label     = batch["label"].to(device)
        sample_id = batch.get("sample_id", "unknown")

        img_feats_list = []
        sc_feats_list  = []
        st_feats_list  = []
        fused_list     = []

        for i in range(0, images.size(0), config["batch_spots"]):
            j = min(i + config["batch_spots"], images.size(0))

            with autocast():
                img_feat = model.img_encoder(images[i:j].to(device))
                img_feat = model.img_head(img_feat)
                sc_feat  = model.sc_encoder(expr[i:j].to(device))
                st_feat  = model.st_encoder(coords[i:j].to(device))
                fused    = model.fusion(img_feat, sc_feat, st_feat)

            img_feats_list.append(img_feat.cpu())
            sc_feats_list.append(sc_feat.cpu())
            st_feats_list.append(st_feat.cpu())
            fused_list.append(fused.cpu())

        # After encoding (spot-level)
        img_feats_spot = torch.cat(img_feats_list, dim=0)  # (N_spots, D)
        sc_feats_spot  = torch.cat(sc_feats_list,  dim=0)
        st_feats_spot  = torch.cat(st_feats_list,  dim=0)
        # After fusion (spot-level)
        fused_spot     = torch.cat(fused_list,     dim=0)  # (N_spots, D)

        with autocast():
            # After MIL pooling (WSI-level)
            wsi_embed, attn = model.mil_pooling(fused_spot.to(device))
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
            all_img_feats.append(img_feats_spot.float().numpy())
            all_sc_feats.append(sc_feats_spot.float().numpy())
            all_st_feats.append(st_feats_spot.float().numpy())
            all_fused.append(fused_spot.float().numpy())
            all_wsi_embeds.append(wsi_embed.cpu().float().numpy())
            all_attn.append(attn.cpu().float().numpy())
            all_sample_ids.append(sample_id)
            all_labels.append(label.item())
            all_preds.append(pred)
            all_scores.append(prob_pos)

    val_loss /= len(loader)
    val_acc   = 100 * correct / len(loader)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = (val_loss, val_acc, auc, float(p), float(r), float(f1), cm)

    embeddings = None
    if save_embeddings:
        embeddings = {
            # Spot-level: list of (N_spots_i, D) arrays
            "img_feats":   all_img_feats,   # after image encoder
            "sc_feats":    all_sc_feats,    # after sc encoder
            "st_feats":    all_st_feats,    # after st encoder
            "fused":       all_fused,       # after fusion
            # WSI-level: (N_samples, D)
            "wsi_embeds":  np.stack(all_wsi_embeds, axis=0),
            "attn":        all_attn,
            # Meta
            "labels":      np.array(all_labels),
            "preds":       np.array(all_preds),
            "scores":      np.array(all_scores),
            "sample_ids":  all_sample_ids,
        }

    return metrics, embeddings


def save_embeddings_to_disk(embeddings, output_path, epoch, split="val"):
    tag = f"{split}_epoch{epoch:02d}"

    # WSI-level embeddings
    np.savez(
        output_path / f"wsi_embeds_{tag}.npz",
        wsi_embeds = embeddings["wsi_embeds"],
        labels     = embeddings["labels"],
        preds      = embeddings["preds"],
        scores     = embeddings["scores"],
        sample_ids = np.array(embeddings["sample_ids"], dtype=object),
    )

    # Spot-level per-module embeddings (list of arrays → object array)
    for key in ["img_feats", "sc_feats", "st_feats", "fused", "attn"]:
        np.save(
            output_path / f"{key}_{tag}.npy",
            np.array(embeddings[key], dtype=object),
            allow_pickle=True,
        )

    print(f"  WSI embeds  : wsi_embeds_{tag}.npz  {embeddings['wsi_embeds'].shape}")
    print(f"  Spot embeds : img_feats / sc_feats / st_feats / fused / attn  [{len(embeddings['fused'])} samples]")


def train(config):
    set_seed(config['seed'])
    device = torch.device(config['device'])

    split_dir   = Path(config['split_dir'])
    output_path = Path(config['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Training: Ver1 (3-modality) / {config['fusion_option']}")
    print('='*80)

    # Load metadata CSV once
    meta_df = pd.read_csv(config['metadata_csv']).set_index('slide')

    # Load samples
    print("\nLoading samples...")
    train_samples = load_samples(split_dir / "train.txt", config['st_dir'], config['patch_dir'], meta_df)
    val_samples   = load_samples(split_dir / "val.txt",   config['st_dir'], config['patch_dir'], meta_df)
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)}")

    # Load global HVG
    hvg_genes = load_global_hvg(config['hvg_path'])

    # Class weights
    from collections import Counter
    train_labels  = [s.label for s in train_samples]
    label_counts  = Counter(train_labels)
    n_samples     = len(train_labels)
    class_weights = torch.tensor([
        n_samples / (2 * label_counts[0]),
        n_samples / (2 * label_counts[1]),
    ], dtype=torch.float).to(device)
    print(f"  Label dist: {dict(label_counts)}")
    print(f"  Class weights: Healthy={class_weights[0]:.2f}, Cancer={class_weights[1]:.2f}")

    # Dataloaders
    train_loader = create_wsi_dataloader(
        train_samples, batch_size=1, shuffle=True,
        max_spots=config['max_spots'], hvg_genes=hvg_genes
    )
    val_loader = create_wsi_dataloader(
        val_samples, batch_size=1, shuffle=False,
        max_spots=config['max_spots'], hvg_genes=hvg_genes
    )

    # Model
    model = MultiModalMILModel(
        num_genes      = config['num_genes'],
        modality_option= 'multi',
        num_classes    = config['num_classes'],
        embed_dim      = config['embed_dim'],
        fusion_option  = config['fusion_option'],
        top_k_genes    = config['top_k_genes'],
    ).to(device)

    for p in model.img_encoder.parameters():
        p.requires_grad = False
    model.img_encoder.eval()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'], weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler    = GradScaler()

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    total_epochs      = config['epochs']
    embed_save_epochs = set(config.get('embed_save_epochs', []))
    embed_save_epochs.add(total_epochs - 1)  # always save last epoch

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")

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

        print(f"  Train: loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val:   loss={val_loss:.4f}  acc={val_acc:.2f}%  AUC={val_auc:.4f}")
        print(f"  P/R/F1: {val_p:.3f}/{val_r:.3f}/{val_f1:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / "best_model.pt")
            print(f"  Saved best model (val_acc={val_acc:.2f}%)")

            # Save best-epoch embeddings (all modules)
            if embeddings is None:
                _, embeddings_best = validate(
                    model, val_loader, criterion, config, device, save_embeddings=True
                )
            else:
                embeddings_best = embeddings
            save_embeddings_to_disk(embeddings_best, output_path, epoch, split="val_best")

        # Save designated-epoch embeddings
        if save_embed and embeddings is not None:
            save_embeddings_to_disk(embeddings, output_path, epoch, split="val")

    # Save history & results
    with open(output_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    results = {
        'model': 'ver1',
        'fusion': config['fusion_option'],
        'best_val_acc': best_val_acc,
        'final_metrics': {
            'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc,
            'precision': val_p, 'recall': val_r, 'f1': val_f1,
        }
    }
    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Best val_acc: {best_val_acc:.2f}%")
    return results


def main():
    config = {
        'st_dir':       './stimage/st_preprocessed',
        'patch_dir':    './merged_data/extracted',
        'metadata_csv': './stimage/meta/meta_sampled.csv',
        'hvg_path':     './stimage/global_hvg_genes.txt',
        'split_dir':    './data_splits/STimage',
        'output_dir':   './results/STimage_ver1_attn',
        'fusion_option':'attn',   # options: 'concat', 'attn', 'spatial_attn'
        'num_genes':    2000,
        'num_classes':  2,
        'embed_dim':    256,
        'top_k_genes':  512,
        'epochs':       10,
        'lr':           5e-4,
        'weight_decay': 1e-4,
        'batch_size':   1,
        'batch_spots':  500,
        'accum_steps':  4,
        'max_spots':    300,
        'device':       'cuda',
        'seed':         42,
        # Epochs (0-indexed) to save embeddings. Last epoch always included.
        'embed_save_epochs': [4],
    }

    results = train(config)

    with open(f"{config['output_dir']}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nALL DONE")


if __name__ == "__main__":
    main()
