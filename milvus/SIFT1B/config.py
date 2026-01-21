"""
SIFT1B Benchmark Configuration
==============================
Configuration constants for billion-scale vector search benchmark.
"""

import os
from pathlib import Path

# ==================== Paths ====================
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"

# Dataset files
BASE_VECTORS_FILE = DATA_DIR / "bigann_base.bvecs"
QUERY_VECTORS_FILE = DATA_DIR / "bigann_query.bvecs"
GROUND_TRUTH_DIR = DATA_DIR / "bigann_gnd"

# ==================== Dataset Properties ====================
DIMENSION = 128
TOTAL_BASE_VECTORS = 1_000_000_000  # 1 billion
TOTAL_QUERY_VECTORS = 10_000
GROUND_TRUTH_K = 100  # Ground truth contains top-100 neighbors

# ==================== Milvus Configuration ====================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "sift1b_benchmark"

# ==================== Benchmark Presets ====================
PRESETS = {
    "10M": {
        "vectors": 10_000_000,
        "batch_size": 50_000,
        "description": "Quick test (10 million vectors)",
    },
    "100M": {
        "vectors": 100_000_000,
        "batch_size": 100_000,
        "description": "Medium test (100 million vectors)",
    },
    "500M": {
        "vectors": 500_000_000,
        "batch_size": 100_000,
        "description": "Large test (500 million vectors)",
    },
    "1B": {
        "vectors": 1_000_000_000,
        "batch_size": 100_000,
        "description": "Full benchmark (1 billion vectors)",
    },
}

# ==================== DiskANN Parameters ====================
INDEX_PARAMS = {
    "index_type": "DISKANN",
    "metric_type": "L2",
    "params": {},
}

# Search parameter presets (search_list must be >= limit)
SEARCH_PRESETS = {
    "fast": {"search_list": 100},      # Fastest, lower recall
    "balanced": {"search_list": 200},  # Good balance
    "accurate": {"search_list": 500},  # Highest recall, slower
}

# ==================== Download URLs ====================
DOWNLOAD_URLS = {
    "base": "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz",
    "query": "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz",
    "groundtruth": "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz",
}

# ==================== Hardware Recommendations ====================
HARDWARE_REQUIREMENTS = {
    "10M": {"ram_gb": 8, "disk_gb": 10},
    "100M": {"ram_gb": 16, "disk_gb": 50},
    "500M": {"ram_gb": 32, "disk_gb": 200},
    "1B": {"ram_gb": 64, "disk_gb": 500},
}
