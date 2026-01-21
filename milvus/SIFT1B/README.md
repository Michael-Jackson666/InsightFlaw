# SIFT1B Billion-Scale Vector Benchmark

Large-scale vector search benchmark using the SIFT1B dataset (1 billion 128-dimensional vectors).

## Dataset Overview

| Property | Value |
|----------|-------|
| Base Vectors | 1,000,000,000 (1B) |
| Query Vectors | 10,000 |
| Dimensions | 128 |
| Distance Metric | L2 (Euclidean) |
| Raw Data Size | ~128 GB |
| Index Size (DiskANN) | ~150-200 GB |

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 32 GB | 64 GB |
| Disk | 500 GB SSD | 1 TB NVMe SSD |
| CPU | 8 cores | 16+ cores |

## Files

```
SIFT1B/
├── README.md           # This file
├── config.py           # Configuration and constants
├── download.py         # Dataset download script
├── benchmark.py        # Full benchmark test
└── data/               # Dataset storage (gitignored)
    ├── bigann_base.bvecs      # Base vectors (128 GB)
    ├── bigann_query.bvecs     # Query vectors
    └── bigann_gnd/            # Ground truth
```

## Download Dataset

The SIFT1B dataset is available from the INRIA BigANN project.

### Option 1: Download Script (Recommended)

```bash
python download.py
```

### Option 2: Manual Download

```bash
# Create data directory
mkdir -p data && cd data

# Download base vectors (128 GB, takes several hours)
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
gunzip bigann_base.bvecs.gz

# Download query vectors
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
gunzip bigann_query.bvecs.gz

# Download ground truth
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
tar -xzf bigann_gnd.tar.gz
```

## Quick Start

1. Ensure Milvus Standalone is running with sufficient resources:

```yaml
# docker-compose.yml - increase memory limit
services:
  standalone:
    deploy:
      resources:
        limits:
          memory: 48G
```

2. Download the dataset:

```bash
python download.py --subset 100M  # Start with 100M subset
```

3. Run the benchmark:

```bash
python benchmark.py --vectors 100000000  # 100M vectors first
```

## Benchmark Configurations

### Test Subsets (for incremental testing)

| Subset | Vectors | Disk Usage | Est. Insert Time |
|--------|---------|------------|------------------|
| 10M | 10,000,000 | ~1.5 GB | 5-10 min |
| 100M | 100,000,000 | ~15 GB | 30-60 min |
| 500M | 500,000,000 | ~75 GB | 3-5 hours |
| 1B | 1,000,000,000 | ~150 GB | 8-12 hours |

### DiskANN Parameters for Billion-Scale

```python
# Recommended index parameters for 1B vectors
index_params = {
    "index_type": "DISKANN",
    "metric_type": "L2",
    "params": {}  # DiskANN auto-tunes for dataset size
}

# Search parameters (adjust based on recall/latency tradeoff)
search_params = {
    "params": {
        "search_list": 100,   # Minimum for Top-100
        # "search_list": 200, # Higher recall
        # "search_list": 500, # Maximum recall
    }
}
```

## Expected Performance

Based on 64GB RAM + NVMe SSD configuration:

| Metric | 100M | 500M | 1B |
|--------|------|------|-----|
| Index Build Time | ~30 min | ~3 hrs | ~8 hrs |
| QPS (search_list=100) | ~500 | ~300 | ~200 |
| Recall@100 | 0.95+ | 0.93+ | 0.90+ |
| P99 Latency | <50ms | <100ms | <150ms |

## Notes

- **Incremental Testing**: Start with 10M, then 100M, before full 1B
- **Disk I/O**: NVMe SSD strongly recommended; HDD will be 10x slower
- **Memory**: DiskANN is disk-optimized but benefits from more RAM for caching
- **Index Persistence**: Milvus stores index on disk, restart preserves data

## References

- [BigANN Benchmark](http://corpus-texmex.irisa.fr/)
- [DiskANN Paper](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
- [Milvus DiskANN Guide](https://milvus.io/docs/disk_index.md)
