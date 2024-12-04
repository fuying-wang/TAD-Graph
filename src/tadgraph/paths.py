import os
from pathlib import Path
import pyrootutils

# find the root dir of package
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

TAD_GRAPH_DIR = Path(root) / "src/tadgraph"
WSI_DATA_DIR = Path("/data1/r20user2/wsi_data")
KIMIANET_WEIGHT = WSI_DATA_DIR / "KimiaNet_weights"
DATASET_CSV_DIR = TAD_GRAPH_DIR / "dataset_csv"

# path for NSCLC dataset
NSCLC_DATA_DIR = WSI_DATA_DIR / "TCGA_NSCLC"
NSCLC_WSI_DIR = NSCLC_DATA_DIR / "WSIs"
NSCLC_DATASET_CSV = DATASET_CSV_DIR / "tcga_nsclc"
NSCLC_METADATA_DIR = NSCLC_DATA_DIR / "metadata"

# path for RCC dataset
RCC_DATA_DIR = WSI_DATA_DIR / "TCGA_RCC"
RCC_DATASET_CSV = DATASET_CSV_DIR / "tcga_rcc"
RCC_METADATA_DIR = RCC_DATA_DIR / "metadata"

# path for BRCA dataset
BRCA_DATA_DIR = WSI_DATA_DIR / "TCGA_BRCA"
BRCA_WSI_DIR = BRCA_DATA_DIR / "WSIs"
BRCA_DATASET_CSV = DATASET_CSV_DIR / "tcga_brca"
BRCA_METADATA_DIR = BRCA_DATA_DIR / "metadata"

# path for BLCA dataset
BLCA_DATA_DIR = WSI_DATA_DIR / "TCGA_BLCA"
BLCA_WSI_DIR = BLCA_DATA_DIR / "WSIs"
BLCA_METADATA_DIR = BLCA_DATA_DIR / "metadata"
BLCA_DATASET_CSV = DATASET_CSV_DIR / "tcga_blca"

ESCA_DATA_DIR = WSI_DATA_DIR / "TCGA_ESCA"
ESCA_WSI_DIR = ESCA_DATA_DIR / "WSIs"
ESCA_METADATA_DIR = ESCA_DATA_DIR / "metadata"
ESCA_DATASET_CSV = DATASET_CSV_DIR / "tcga_esca"

PRAD_DATA_DIR = WSI_DATA_DIR / "TCGA_PRAD"
PRAD_WSI_DIR = PRAD_DATA_DIR / "WSIs"
PRAD_DATASET_CSV = DATASET_CSV_DIR / "tcga_prad"

RESULTS_DIR = Path(root) / "results"
SPLIT_DIR = TAD_GRAPH_DIR / "splits"
PRESET_DIR = TAD_GRAPH_DIR / "presets"

# path for heatmap
HEATMAP_DIR = TAD_GRAPH_DIR / "heatmaps"
SUMMARY_CSV = TAD_GRAPH_DIR / "summary/summarize_results.csv"
