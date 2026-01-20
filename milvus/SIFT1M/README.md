# SIFT1M Benchmark Tests

Performance benchmarking using the SIFT1M dataset for DiskANN index evaluation.

## Dataset

**SIFT1M** is a standard benchmark dataset for approximate nearest neighbor (ANN) algorithms:
- 1,000,000 base vectors (128-dimensional)
- 10,000 query vectors
- Ground truth neighbors for recall calculation

## Files

| File | Description |
|------|-------------|
| `diskann-test.py` | DiskANN performance benchmark script |
| `SIFT1M.md` | Dataset documentation and download instructions |
| `sift-128-euclidean.hdf5` | Dataset file (not tracked in git) |

## Download Dataset

```bash
curl -L -o sift-128-euclidean.hdf5 \
  "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
```

## Run Benchmark

```bash
# Ensure Milvus Standalone is running
docker ps | grep milvus

# Run the benchmark
python diskann-test.py
```

## Metrics

The benchmark measures:
- **QPS**: Queries per second (throughput)
- **Latency**: Average query response time
- **Recall@K**: Percentage of true nearest neighbors found

## Requirements

- Milvus Standalone (Docker)
- Python packages: `pymilvus`, `h5py`, `numpy`
