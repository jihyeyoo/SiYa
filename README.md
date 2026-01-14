# Repo for EWHA Capstone Design Project

## 📂 Data Preparation

To run this project, the dataset must be organized in the following directory structure. This project supports **HEST-like datasets** where each sample (slide) is stored in its own folder containing Spatial Transcriptomics (ST) data, image patches, and metadata.

### 1. Directory Structure

Ensure your data root folder (e.g., `hest_data/`) follows this hierarchy:

```text
hest_data/
├── TENX24/                       # Sample ID (Slide Name)
│   ├── st/
│   │   └── st.h5ad               # Spatial Transcriptomics data (AnnData format)
│   ├── patches/
│   │   └── patches.h5            # H&E Image patches & coordinates (h5 format)
│   └── metadata/
│       └── metadata.json         # Clinical metadata (e.g., disease_state)
│
├── TENX39/
│   ├── st/
│   │   └── st.h5ad
│   ├── patches/
│   │   └── patches.h5
│   └── metadata/
│       └── metadata.json
│
└── ... (Other samples)
