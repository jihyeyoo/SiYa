"""HEST + STimage unified Global HVG preprocessing"""

import os
import json
from tqdm import tqdm
from collections import Counter
import pandas as pd
import scanpy as sc
import numpy as np
import h5py
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Configuration
# ============================================
STIMAGE_DIRS = ['./stimage_data']
HEST_DIR = './hest_data'
OUTPUT_DIR = 'merged_data'
NUM_WORKERS = min(cpu_count() - 2, 16)

os.makedirs(OUTPUT_DIR, exist_ok=True)
temp_root = os.path.join(OUTPUT_DIR, 'st_temp')
os.makedirs(temp_root, exist_ok=True)

print(f"Using {NUM_WORKERS} workers for parallel processing")

# ============================================
# Parallel processing helper functions
# ============================================

def extract_categorical_value(f_obs, key, index=0):
    """Extract categorical or direct value from obs"""
    if 'categories' in f_obs[key]:
        # Categorical type
        cats = f_obs[key]['categories'][:]
        codes = f_obs[key]['codes'][:]
        val = cats[codes[index]]
        return val.decode() if isinstance(val, bytes) else str(val)
    else:
        # Direct array type
        val = f_obs[key][index]
        if isinstance(val, bytes):
            return val.decode()
        elif isinstance(val, (np.integer, np.floating)):
            return int(val)  # Convert numpy scalar to Python int
        else:
            return str(val)


def process_stimage_sample(args):
    """Process a single STimage sample (parallel) - supports direct arrays"""
    fpath, stimage_dir, temp_root = args
    fname = os.path.basename(fpath)
    sample_id = fname.replace('.h5ad', '')
    dataset_name = os.path.basename(stimage_dir)
    
    try:
        with h5py.File(fpath, 'r') as f:
            f_obs = f['obs']
            
            # sample_id
            sample_id_stored = extract_categorical_value(f_obs, 'sample_id') if 'sample_id' in f_obs else sample_id
            
            # disease_state - direct array (0 or 1)
            if 'disease_state' in f_obs:
                ds_val = extract_categorical_value(f_obs, 'disease_state')
                
                # Numeric or string handling
                if isinstance(ds_val, (int, float)):
                    disease_state = int(ds_val)
                else:
                    ds_str = str(ds_val).strip()
                    if ds_str in ['1', '1.0', 'Tumor', 'Cancer', 'cancer', 'tumor']:
                        disease_state = 1
                    elif ds_str in ['0', '0.0', 'Healthy', 'Normal', 'healthy', 'normal']:
                        disease_state = 0
                    else:
                        disease_state = -1
            else:
                disease_state = -1
            
            # organ
            organ = extract_categorical_value(f_obs, 'organ') if 'organ' in f_obs else 'Unknown'
            
            # species
            species = extract_categorical_value(f_obs, 'species') if 'species' in f_obs else 'Unknown'
        
        # Load AnnData
        adata = sc.read_h5ad(fpath)
        
        if 'raw' not in adata.layers:
            adata.layers["raw"] = adata.X.copy()
        
        adata.var_names_make_unique()
        
        # Update metadata
        adata.obs['sample_id'] = sample_id_stored
        adata.obs['disease_state'] = disease_state
        adata.obs['organ'] = organ
        adata.obs['species'] = species
        adata.obs['dataset'] = f'STimage-{dataset_name}'
        
        # Preprocessing
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        
        # HVG computation
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', layer='raw')
        hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
        
        # Temporary save
        temp_path = os.path.join(temp_root, f"STimage_{dataset_name}_{sample_id}.h5ad")
        adata.write_h5ad(temp_path, compression='gzip', compression_opts=9)
        
        metadata = {
            'sample_id': str(sample_id_stored),
            'dataset': f'STimage-{dataset_name}',
            'organ': str(organ),
            'disease_state': int(disease_state),
            'n_obs': int(adata.n_obs),
            'n_vars': int(adata.n_vars),
            'n_hvg': int(len(hvg_genes)),
        }
        
        return hvg_genes, metadata, None
        
    except Exception as e:
        import traceback
        return [], None, f"STimage {sample_id}: {str(e)[:100]}\n{traceback.format_exc()[:200]}"


def process_hest_sample(args):
    """Process a single HEST sample (parallel)"""
    fpath, hest_meta_root, temp_root = args
    fname = os.path.basename(fpath)
    sample_id = fname.replace('.h5ad', '')
    
    try:
        # Load metadata
        meta_path = os.path.join(hest_meta_root, sample_id + '.json')
        with open(meta_path) as f:
            meta = json.load(f)
        
        disease_state_raw = meta.get('disease_state')
        if disease_state_raw in ['Tumor', 'Cancer']:
            disease_state = 1
        elif disease_state_raw == 'Healthy':
            disease_state = 0
        else:
            disease_state = -1
        
        # Load AnnData
        adata = sc.read_h5ad(fpath)
        
        if 'raw' not in adata.layers:
            adata.layers["raw"] = adata.X.copy()
        
        adata.var_names_make_unique()
        
        adata.obs['sample_id'] = sample_id
        adata.obs['organ'] = meta.get('organ', 'Unknown')
        adata.obs['disease_state'] = disease_state
        adata.obs['species'] = meta.get('species', 'Unknown')
        adata.obs['dataset'] = 'HEST'
        
        # Preprocessing
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        
        # HVG computation
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', layer='raw')
        hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
        
        # Temporary save
        temp_path = os.path.join(temp_root, f"HEST_{sample_id}.h5ad")
        adata.write_h5ad(temp_path, compression='gzip', compression_opts=9)
        
        metadata = {
            'sample_id': str(sample_id),
            'dataset': 'HEST',
            'organ': str(meta.get('organ', 'Unknown')),
            'disease_state': int(disease_state),
            'n_obs': int(adata.n_obs),
            'n_vars': int(adata.n_vars),
            'n_hvg': int(len(hvg_genes)),
        }
        
        return hvg_genes, metadata, None
        
    except Exception as e:
        return [], None, f"HEST {sample_id}: {str(e)[:100]}"


# ============================================
# STEP 1: Sample collection and HVG computation (parallel)
# ============================================
print("\n" + "="*70)
print("STEP 1: Collect samples from STimage + HEST (Parallel)")
print("="*70)

all_hvg_genes = []
sample_metadata = []

# ----- Process STimage samples (parallel) -----
print("\n[Processing STimage samples]")

for stimage_dir in STIMAGE_DIRS:
    stimage_st_root = os.path.join(stimage_dir, 'st')
    
    if not os.path.exists(stimage_st_root):
        print(f"Skipping {stimage_dir} (not found)")
        continue
    
    dataset_name = os.path.basename(stimage_dir)
    
    # Collect all file paths
    file_paths = [
        os.path.join(stimage_st_root, f) 
        for f in sorted(os.listdir(stimage_st_root)) 
        if f.endswith('.h5ad')
    ]
    
    # Parallel processing
    args_list = [(fpath, stimage_dir, temp_root) for fpath in file_paths]
    
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_stimage_sample, args_list),
            total=len(args_list),
            desc=f"STimage-{dataset_name}"
        ))
    
    # Collect results
    for hvg_genes, metadata, error in results:
        if error:
            print(f"  ✗ {error}")
        else:
            all_hvg_genes.extend(hvg_genes)
            sample_metadata.append(metadata)

# ----- Process HEST samples (parallel) -----
print("\n[Processing HEST samples]")
hest_st_root = os.path.join(HEST_DIR, 'st')
hest_meta_root = os.path.join(HEST_DIR, 'metadata')

file_paths = [
    os.path.join(hest_st_root, f) 
    for f in sorted(os.listdir(hest_st_root)) 
    if f.endswith('.h5ad')
]

args_list = [(fpath, hest_meta_root, temp_root) for fpath in file_paths]

with Pool(NUM_WORKERS) as pool:
    results = list(tqdm(
        pool.imap(process_hest_sample, args_list),
        total=len(args_list),
        desc="HEST"
    ))

for hvg_genes, metadata, error in results:
    if error:
        print(f"  ✗ {error}")
    else:
        all_hvg_genes.extend(hvg_genes)
        sample_metadata.append(metadata)

# Summary
print(f"\nTotal samples collected: {len(sample_metadata)}")
print(f"  - HEST: {sum(1 for s in sample_metadata if s['dataset'] == 'HEST')}")
for stimage_dir in STIMAGE_DIRS:
    dataset_name = os.path.basename(stimage_dir)
    count = sum(1 for s in sample_metadata if s['dataset'] == f'STimage-{dataset_name}')
    print(f"  - STimage-{dataset_name}: {count}")

label_dist = pd.DataFrame(sample_metadata)['disease_state'].value_counts().sort_index()
print(f"\nLabel distribution:")
for label, count in label_dist.items():
    label_name = {0: 'Healthy', 1: 'Cancer', -1: 'Unknown'}[label]
    print(f"  {label_name} ({label}): {count}")

# ============================================
# STEP 2: Unified Global HVG selection
# ============================================
print("\n" + "="*70)
print("STEP 2: Select MERGED Global HVG (Top 2000 genes)")
print("="*70)

gene_counter = Counter(all_hvg_genes)
most_common = gene_counter.most_common(2000)
global_hvg = sorted([gene for gene, count in most_common])

print(f"\n Merged Global HVG Selection:")
print(f"  Total unique HVG candidates: {len(set(all_hvg_genes))}")
print(f"  Selected global HVG: {len(global_hvg)}")
print(f"\nTop 10 genes by frequency:")
for i, (gene, count) in enumerate(most_common[:10], 1):
    pct = 100 * count / len(sample_metadata)
    print(f"  {i:2d}. {gene:15s}: {count:3d}/{len(sample_metadata):3d} samples ({pct:5.1f}%)")

# ============================================
# STEP 3: Filter to Global HVG and save (parallel)
# ============================================
print("\n" + "="*70)
print("STEP 3: Filter to Global HVG and Save (Parallel)")
print("="*70)

processed_root = os.path.join(OUTPUT_DIR, 'global_hvg_unified')
os.makedirs(processed_root, exist_ok=True)

def filter_to_global_hvg(args):
    """Filter AnnData to Global HVG (parallel)"""
    temp_path, global_hvg, processed_root = args
    fname = os.path.basename(temp_path)
    
    try:
        adata = sc.read_h5ad(temp_path)
        
        # Intersection with Global HVG
        available_genes = np.intersect1d(adata.var_names, global_hvg)
        coverage = len(available_genes) / len(global_hvg) * 100
        
        if len(available_genes) == 0:
            return None, None, f"{fname}: No common genes!"
        
        # Filter to Global HVG
        adata = adata[:, available_genes].copy()
        
        # Final save
        output_name = fname.replace('HEST_', '').replace('STimage_stimage_data_', '')
        output_path = os.path.join(processed_root, output_name)
        adata.write_h5ad(output_path, compression='gzip', compression_opts=9)
        
        disease_state = adata.obs['disease_state'].values[0]
        dataset = adata.obs['dataset'].values[0]
        organ = adata.obs['organ'].values[0]
        
        info = {
            'disease_state': disease_state,
            'coverage': coverage,
            'n_obs': adata.n_obs,
            'n_vars': adata.n_vars,
            'dataset': dataset,
            'organ': organ,
            'output_name': output_name
        }
        
        return info, coverage, None
        
    except Exception as e:
        return None, None, f"{fname}: {e}"

# Parallel filtering
temp_files = [
    os.path.join(temp_root, f) 
    for f in sorted(os.listdir(temp_root)) 
    if f.endswith('.h5ad')
]

args_list = [(fpath, global_hvg, processed_root) for fpath in temp_files]

healthy_count = 0
cancer_count = 0
unknown_count = 0
gene_coverage = []

with Pool(NUM_WORKERS) as pool:
    results = list(tqdm(
        pool.imap(filter_to_global_hvg, args_list),
        total=len(args_list),
        desc="Filtering"
    ))

for info, coverage, error in results:
    if error:
        print(f"  ✗ {error}")
    else:
        gene_coverage.append(coverage)
        if info['disease_state'] == 1:
            cancer_count += 1
        elif info['disease_state'] == 0:
            healthy_count += 1
        else:
            unknown_count += 1

# ============================================
# STEP 4: Cleanup & summary
# ============================================
print("\n" + "="*70)
print("Cleaning up temporary files...")
print("="*70)

import shutil
shutil.rmtree(temp_root)
print("✓ Temporary files removed")

print("\n" + "="*70)
print("MERGED PREPROCESSING SUMMARY")
print("="*70)
print(f"Total samples processed: {healthy_count + cancer_count + unknown_count}")
print(f"  - Healthy (0): {healthy_count}")
print(f"  - Cancer (1): {cancer_count}")
print(f"  - Unknown (-1): {unknown_count}")
print(f"\nGlobal HVG: {len(global_hvg)} genes")
print(f"Gene coverage across samples:")
print(f"  - Mean: {np.mean(gene_coverage):.2f}%")
print(f"  - Min:  {np.min(gene_coverage):.2f}%")
print(f"  - Max:  {np.max(gene_coverage):.2f}%")

# Save Global HVG list
hvg_path = os.path.join(OUTPUT_DIR, 'global_hvg_genes_unified.txt')
with open(hvg_path, 'w') as f:
    f.write('\n'.join(global_hvg))
print(f"\n✓ Global HVG list saved to: {hvg_path}")

# Save sample metadata
metadata_df = pd.DataFrame(sample_metadata)
metadata_path = os.path.join(OUTPUT_DIR, 'sample_metadata_unified.csv')
metadata_df.to_csv(metadata_path, index=False)
print(f"✓ Sample metadata saved to: {metadata_path}")

print("\n" + "="*70)
print("MERGED PREPROCESSING COMPLETE!")
print("="*70)
print(f"\nMerged data location: {processed_root}")
print(f"Number of files: {len([f for f in os.listdir(processed_root) if f.endswith('.h5ad')])}")
