# Milvus Tutorials

Interactive Jupyter notebooks for learning Milvus vector database.

## Tutorials

### 1. Milvus Lite Tutorial (`lite-tutorial.ipynb`)
A beginner-friendly introduction to Milvus Lite:
- Database initialization
- Collection creation
- Data insertion
- Vector similarity search
- Query with filters

**Prerequisites:** `pip install pymilvus`

### 2. DiskANN Tutorial (`diskann-tutorial.ipynb`)
Advanced tutorial on DiskANN index for large-scale vector search:
- DiskANN algorithm overview
- Index creation and configuration
- Performance tuning with search_list parameter
- Benchmark and recall analysis

**Prerequisites:** 
- `pip install pymilvus`
- Docker with Milvus Standalone running

## Getting Started

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter Notebook
jupyter notebook

# Open any tutorial and run cells sequentially
```

## Notes

- Run cells in order to maintain kernel state
- Milvus Lite creates local `.db` files in the working directory
- DiskANN requires Milvus Standalone (Docker)
