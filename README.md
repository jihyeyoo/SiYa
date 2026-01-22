# Repo for SiYa Project

## 📂 Data Preparation

To run this project, the dataset must be organized in the following directory structure. This project supports **HEST-like datasets** where each sample (slide) is stored in its own folder containing Spatial Transcriptomics (ST) data, image patches, and metadata.

### 1. Directory Structure

Ensure your data root folder (e.g., `hest_data/`) follows this hierarchy:

```text
hest_data/
├── st/                                # Spatial Transcriptomics data
│   ├── TENX24.h5ad
│   ├── TENX39.h5ad
│   ├── TENX97.h5ad
│   └── ...
│
├── patches/                           # Image patches (H&E)
│   ├── TENX24.h5
│   ├── TENX39.h5
│   ├── TENX97.h5
│   └── ...
│
├── metadata/                          # Clinical / sample-level metadata
│   ├── TENX24.json
│   ├── TENX39.json
│   ├── TENX97.json
│   └── ...
│
├── st_preprocessed_global_hvg/        # ST data filtered with GLOBAL HVGs
│   ├── TENX24.h5ad
│   ├── TENX39.h5ad
│   └── ...
│
├── st_preprocessed_sample_hvg/        # (Optional) sample-wise HVG preprocessing
│   ├── TENX24.h5ad
│   ├── TENX39.h5ad
│   └── ...
│
├── global_hvg_genes.txt               # Global HVG list (shared gene order)
└── sample_metadata.csv                # Aggregated sample-level metadata

