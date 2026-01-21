# Milvus Vector Database Deployment

This directory contains Milvus vector database deployment guides, tutorials, and test scripts for the InsightFlaw RAG application.

## Directory Structure

```
milvus/
├── README.md                           # This file
├── tutorial/                           # Interactive tutorials (Jupyter Notebooks)
│   ├── lite-tutorial.ipynb             # Milvus Lite quickstart
│   └── diskann-tutorial.ipynb          # DiskANN index tutorial
├── SIFT1M/                             # SIFT1M benchmark (1M vectors)
│   ├── diskann-test.py                 # DiskANN performance benchmark
│   └── SIFT1M.md                       # Dataset documentation
├── SIFT1B/                             # SIFT1B benchmark (1B vectors) ⭐
│   ├── benchmark.py                    # Billion-scale benchmark script
│   ├── download.py                     # Dataset download utility
│   └── config.py                       # Benchmark configuration
├── diskann.py                          # DiskANN index implementation
├── standalone.py                       # Milvus Standalone connection test
├── test-lite.py                        # Milvus Lite basic test
├── Milvus Lite部署与应用-EasyVectordb.md  # Detailed deployment guide (CN)
└── Milvus-lite部署流程.md               # Quick start guide (CN)
```

## Deployment Modes

### Milvus Lite (Recommended for Development)
- Embedded mode, no server required
- Data stored in local SQLite file
- Perfect for prototyping and testing

```python
from pymilvus import MilvusClient
client = MilvusClient("./milvus_demo.db")
```

### Milvus Standalone (Production)
- Docker-based deployment
- Full feature support including DiskANN
- Suitable for production workloads

```bash
# Start Milvus Standalone
docker-compose up -d
```

```python
from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")
```

## Quick Start

1. Install dependencies:
```bash
pip install pymilvus
```

2. Run the lite tutorial:
```bash
jupyter notebook tutorial/lite-tutorial.ipynb
```

## Index Types

| Index Type | Use Case | Memory | Performance |
|------------|----------|--------|-------------|
| FLAT | Small datasets (<10K) | High | Exact search |
| IVF_FLAT | Medium datasets | Medium | Good recall |
| HNSW | Real-time search | High | Low latency |
| DiskANN | Large datasets (1M+) | Low | High throughput |

## Benchmark Datasets

| Dataset | Vectors | Dimensions | Use Case |
|---------|---------|------------|----------|
| SIFT1M | 1,000,000 | 128 | Quick testing |
| SIFT1B | 1,000,000,000 | 128 | Billion-scale production test |

### SIFT1B Requirements (64GB RAM + 1TB SSD)

```bash
# Download and run billion-scale benchmark
cd SIFT1B
python download.py           # Download ~130GB dataset
python benchmark.py -n 100M  # Start with 100M subset
python benchmark.py -n 1B    # Full billion-scale test
```

## Requirements

- Python 3.8+
- pymilvus >= 2.5.0
- Docker (for Standalone mode)
