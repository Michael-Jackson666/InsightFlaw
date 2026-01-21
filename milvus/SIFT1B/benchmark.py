"""
SIFT1B Billion-Scale Vector Search Benchmark
=============================================
Benchmarks DiskANN index performance on 1 billion 128-dimensional vectors.

Usage:
    python benchmark.py                    # Full 1B benchmark
    python benchmark.py --vectors 100M     # 100M subset
    python benchmark.py --vectors 10M      # 10M quick test

Prerequisites:
    - Milvus Standalone running with sufficient resources
    - SIFT1B dataset downloaded (run download.py first)
    - pip install pymilvus numpy
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np

from config import (
    BASE_VECTORS_FILE,
    COLLECTION_NAME,
    DIMENSION,
    GROUND_TRUTH_DIR,
    INDEX_PARAMS,
    MILVUS_URI,
    PRESETS,
    QUERY_VECTORS_FILE,
    SEARCH_PRESETS,
)


def read_bvecs(filepath: Path, num_vectors: int = -1) -> np.ndarray:
    """
    Read vectors from .bvecs file format (BigANN format).
    
    Format: Each vector is stored as [dim (4 bytes)] + [dim * uint8 values]
    """
    print(f"📂 Reading vectors from {filepath.name}...")
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Each vector: 4 bytes (dimension) + 128 bytes (data)
    vector_size = 4 + DIMENSION
    
    with open(filepath, "rb") as f:
        if num_vectors == -1:
            # Calculate total vectors from file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            num_vectors = file_size // vector_size
            f.seek(0)  # Seek back to start
        
        print(f"   Loading {num_vectors:,} vectors...")
        
        vectors = np.zeros((num_vectors, DIMENSION), dtype=np.float32)
        
        for i in range(num_vectors):
            # Read dimension (should be 128)
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            assert dim == DIMENSION, f"Unexpected dimension: {dim}"
            
            # Read vector data
            vec_bytes = f.read(DIMENSION)
            vectors[i] = np.frombuffer(vec_bytes, dtype=np.uint8).astype(np.float32)
            
            if (i + 1) % 10_000_000 == 0:
                print(f"   Progress: {(i + 1):,} / {num_vectors:,}")
    
    print(f"   ✅ Loaded {num_vectors:,} vectors, shape: {vectors.shape}")
    return vectors


def read_ivecs(filepath: Path) -> np.ndarray:
    """Read ground truth from .ivecs file format."""
    print(f"📂 Reading ground truth from {filepath.name}...")
    
    with open(filepath, "rb") as f:
        # Read first dimension
        dim_bytes = f.read(4)
        dim = struct.unpack("i", dim_bytes)[0]
        
        # Calculate number of queries
        f.seek(0, 2)
        file_size = f.tell()
        num_queries = file_size // (4 + dim * 4)
        f.seek(0)
        
        result = np.zeros((num_queries, dim), dtype=np.int32)
        
        for i in range(num_queries):
            dim_check = struct.unpack("i", f.read(4))[0]
            result[i] = np.frombuffer(f.read(dim * 4), dtype=np.int32)
        
        print(f"   ✅ Loaded ground truth: {result.shape}")
        return result


def check_prerequisites():
    """Verify dataset files exist."""
    missing = []
    
    if not QUERY_VECTORS_FILE.exists():
        missing.append(str(QUERY_VECTORS_FILE))
    
    # Check ground truth
    gnd_file = GROUND_TRUTH_DIR / "idx_1000M.ivecs"
    if not gnd_file.exists():
        # Try alternative paths
        alt_gnd = GROUND_TRUTH_DIR / "gnd" / "idx_1000M.ivecs"
        if not alt_gnd.exists():
            missing.append(str(gnd_file))
    
    if missing:
        print("❌ Missing required files:")
        for f in missing:
            print(f"   - {f}")
        print("\n💡 Run 'python download.py --query-only' to download query data")
        return False
    
    return True


def benchmark(num_vectors: int, batch_size: int = 100_000):
    """Run the full benchmark."""
    from pymilvus import DataType, MilvusClient
    
    print("=" * 70)
    print(f"🚀 SIFT1B Benchmark: {num_vectors:,} vectors")
    print("=" * 70)
    
    # 1. Connect to Milvus
    print("\n📡 Connecting to Milvus...")
    try:
        client = MilvusClient(uri=MILVUS_URI)
        print(f"   ✅ Connected to {MILVUS_URI}")
        print(f"   Server version: {client.get_server_version()}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("   Ensure Milvus Standalone is running:")
        print("   docker-compose up -d")
        return
    
    # 2. Load base vectors
    print("\n" + "=" * 70)
    print("📂 Loading Base Vectors")
    print("=" * 70)
    
    if not BASE_VECTORS_FILE.exists():
        print(f"❌ Base vectors not found: {BASE_VECTORS_FILE}")
        print("   Run 'python download.py' to download the dataset")
        return
    
    base_vectors = read_bvecs(BASE_VECTORS_FILE, num_vectors)
    
    # 3. Create collection
    print("\n" + "=" * 70)
    print("📦 Creating Collection")
    print("=" * 70)
    
    if client.has_collection(COLLECTION_NAME):
        print(f"   ⚠️  Dropping existing collection: {COLLECTION_NAME}")
        client.drop_collection(COLLECTION_NAME)
    
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    
    # Create with DiskANN index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=INDEX_PARAMS["index_type"],
        metric_type=INDEX_PARAMS["metric_type"],
        index_name="vector_index"
    )
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print(f"   ✅ Created collection: {COLLECTION_NAME}")
    print(f"   Index: DiskANN (L2)")
    
    # 4. Insert data
    print("\n" + "=" * 70)
    print("📤 Inserting Vectors")
    print("=" * 70)
    
    insert_start = time.time()
    total_inserted = 0
    
    for i in range(0, num_vectors, batch_size):
        end = min(i + batch_size, num_vectors)
        batch_data = [
            {"id": j, "vector": base_vectors[j].tolist()}
            for j in range(i, end)
        ]
        
        client.insert(collection_name=COLLECTION_NAME, data=batch_data)
        total_inserted = end
        
        progress = (end / num_vectors) * 100
        elapsed = time.time() - insert_start
        rate = total_inserted / elapsed if elapsed > 0 else 0
        eta = (num_vectors - total_inserted) / rate if rate > 0 else 0
        
        print(f"   Progress: {end:>12,} / {num_vectors:,} ({progress:5.1f}%) | "
              f"Rate: {rate:,.0f} vec/s | ETA: {eta/60:.1f} min", end="\r")
    
    insert_time = time.time() - insert_start
    print(f"\n   ✅ Inserted {total_inserted:,} vectors in {insert_time:.1f}s")
    print(f"   Insert rate: {total_inserted / insert_time:,.0f} vectors/sec")
    
    # 5. Flush and build index
    print("\n" + "=" * 70)
    print("🔨 Building DiskANN Index")
    print("=" * 70)
    print("   ⏳ This may take a while for large datasets...")
    
    index_start = time.time()
    client.flush(collection_name=COLLECTION_NAME)
    
    # Wait for index to complete
    while True:
        info = client.describe_index(collection_name=COLLECTION_NAME, index_name="vector_index")
        indexed = info.get("indexed_rows", 0)
        total = info.get("total_rows", num_vectors)
        
        if indexed >= num_vectors:
            break
        
        print(f"   Indexing: {indexed:,} / {total:,} ({indexed/total*100:.1f}%)", end="\r")
        time.sleep(5)
    
    index_time = time.time() - index_start
    print(f"\n   ✅ Index built in {index_time:.1f}s ({index_time/60:.1f} min)")
    
    # 6. Load collection
    print("\n📥 Loading collection into memory...")
    client.load_collection(COLLECTION_NAME)
    print("   ✅ Collection loaded")
    
    # 7. Load query vectors and ground truth
    print("\n" + "=" * 70)
    print("🔍 Running Search Benchmark")
    print("=" * 70)
    
    query_vectors = read_bvecs(QUERY_VECTORS_FILE)
    
    # Find ground truth file (depends on number of vectors)
    gnd_file = None
    for suffix in ["1000M", "500M", "100M", "10M"]:
        candidate = GROUND_TRUTH_DIR / f"idx_{suffix}.ivecs"
        if candidate.exists():
            gnd_file = candidate
            break
    
    ground_truth = None
    if gnd_file:
        ground_truth = read_ivecs(gnd_file)
    else:
        print("   ⚠️  Ground truth not found, skipping recall calculation")
    
    # Warmup
    print("\n   🔥 Warming up...")
    for _ in range(10):
        client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vectors[0].tolist()],
            limit=100,
            search_params={"metric_type": "L2", "params": SEARCH_PRESETS["balanced"]}
        )
    
    # Benchmark different search_list values
    print("\n   📊 Search Performance:")
    print("-" * 70)
    print(f"   {'search_list':<15} {'QPS':<12} {'Latency (ms)':<15} {'Recall@100':<12}")
    print("-" * 70)
    
    num_queries = min(1000, len(query_vectors))
    top_k = 100
    
    for name, params in SEARCH_PRESETS.items():
        search_params = {"metric_type": "L2", "params": params}
        query_data = [vec.tolist() for vec in query_vectors[:num_queries]]
        
        # Measure search time
        start = time.time()
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=query_data,
            limit=top_k,
            search_params=search_params
        )
        elapsed = time.time() - start
        
        qps = num_queries / elapsed
        latency_ms = (elapsed / num_queries) * 1000
        
        # Calculate recall if ground truth available
        recall = "N/A"
        if ground_truth is not None:
            recall_count = 0
            for i, hits in enumerate(results):
                result_ids = set([hit["id"] for hit in hits])
                gt_ids = set(ground_truth[i, :top_k])
                recall_count += len(result_ids.intersection(gt_ids))
            recall = f"{recall_count / (num_queries * top_k):.4f}"
        
        sl = params["search_list"]
        print(f"   {sl:<15} {qps:<12.1f} {latency_ms:<15.2f} {recall:<12}")
    
    print("-" * 70)
    
    # 8. Summary
    print("\n" + "=" * 70)
    print("📊 Benchmark Summary")
    print("=" * 70)
    print(f"   Vectors:        {num_vectors:,}")
    print(f"   Dimension:      {DIMENSION}")
    print(f"   Index Type:     DiskANN")
    print(f"   Insert Time:    {insert_time:.1f}s ({insert_time/60:.1f} min)")
    print(f"   Index Time:     {index_time:.1f}s ({index_time/60:.1f} min)")
    print(f"   Total Time:     {(insert_time + index_time)/60:.1f} min")
    print("=" * 70)
    
    # Cleanup
    client.release_collection(COLLECTION_NAME)
    client.close()
    print("\n✅ Benchmark complete!")


def parse_num_vectors(value: str) -> int:
    """Parse vector count from string (supports K, M, B suffixes)."""
    value = value.upper().strip()
    
    if value in PRESETS:
        return PRESETS[value]["vectors"]
    
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    
    for suffix, mult in multipliers.items():
        if value.endswith(suffix):
            return int(float(value[:-1]) * mult)
    
    return int(value)


def main():
    parser = argparse.ArgumentParser(
        description="SIFT1B billion-scale vector search benchmark"
    )
    parser.add_argument(
        "--vectors", "-n",
        type=str,
        default="1B",
        help="Number of vectors to test (e.g., 10M, 100M, 500M, 1B)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100_000,
        help="Batch size for insertion"
    )
    parser.add_argument(
        "--skip-insert",
        action="store_true",
        help="Skip data insertion (use existing collection)"
    )
    
    args = parser.parse_args()
    
    if not check_prerequisites():
        return
    
    num_vectors = parse_num_vectors(args.vectors)
    print(f"🎯 Target: {num_vectors:,} vectors")
    
    benchmark(num_vectors, args.batch_size)


if __name__ == "__main__":
    main()
