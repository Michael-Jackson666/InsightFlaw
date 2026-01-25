# InsightFlaw

InsightFlaw is a lightweight RAG (Retrieval-Augmented Generation) project combining LLMs and a vector database to turn fragmented data into actionable insights.

## Overview

- **LLM**: Large language models for understanding and generating text
- **VectorDB**: Milvus for efficient vector retrieval (Lite and Standalone)
- **Framework**: LangChain for building RAG workflows
- **Indexer**: DiskANN for large-scale approximate nearest neighbor search

## Features

- Document vectorization and storage
- Semantic search and question answering
- Support for multiple data formats and datasets
- High-performance vector retrieval (DiskANN)

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run Milvus Lite (for development):

```bash
# open the tutorial notebook or run the lite demo
jupyter notebook milvus/lite-tutorial.ipynb
```

3. Run Milvus Standalone (for production-like testing):

```bash
# start Milvus Standalone
docker-compose -f milvus/docker-compose.yml up -d
```

4. Use the unified benchmark suite:

```bash
# list datasets
python milvus/benchmark/diskann-test.py --list

# download and run GIST-960 benchmark
python milvus/benchmark/diskann-test.py --dataset gist --download
python milvus/benchmark/diskann-test.py --dataset gist
```

## Repository Layout

```
InsightFlaw/
├── milvus/                    # Milvus deployment, tutorials, benchmarks
│   ├── benchmark/             # Unified benchmark scripts and datasets
│   ├── tutorial/              # Jupyter notebooks and guides
│   └── ...
├── LICENSE
└── README.md
```

## Roadmap

- Integrate LangChain for orchestration
- Build document ingestion & vectorization pipelines
- Add a web UI for interactive search and QA
- Add more LLM backends and improve retrieval latency

## License

See the LICENSE file for details.
