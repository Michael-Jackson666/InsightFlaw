"""
DiskANN Index Demo
==================
Demonstrates DiskANN index creation and vector search with Milvus Standalone.

DiskANN is a disk-based approximate nearest neighbor algorithm optimized for
large-scale datasets. It provides high recall while reducing memory usage.

Usage:
    python diskann.py

Prerequisites:
    - Docker running with Milvus Standalone
    - pip install pymilvus numpy
"""

import numpy as np
from pymilvus import (
    MilvusClient,
    DataType,
    FieldSchema,
    CollectionSchema,
)

# ==================== Configuration ====================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "diskann_demo"
DIMENSION = 128
NUM_ENTITIES = 10000


def main():
    """Run DiskANN demo."""
    # ==================== 1. Connect to Milvus ====================
    print("=" * 60)
    print("1. Connecting to Milvus Standalone")
    print("=" * 60)

    try:
        client = MilvusClient(uri=MILVUS_URI)
        print(f"✅ Connected to Milvus!")
        print(f"   Server version: {client.get_server_version()}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("   Make sure Milvus Docker is running")
        return

    # ==================== 2. Create Collection ====================
    print("\n" + "=" * 60)
    print("2. Creating collection with DiskANN support")
    print("=" * 60)

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"⚠️  Dropped existing collection: {COLLECTION_NAME}")

    # Define schema
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=256)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        consistency_level="Strong"
    )
    print(f"✅ Created collection: {COLLECTION_NAME}")

    # ==================== 3. Insert Data ====================
    print("\n" + "=" * 60)
    print("3. Inserting test data")
    print("=" * 60)

    # Generate random vectors
    np.random.seed(42)
    vectors = np.random.random((NUM_ENTITIES, DIMENSION)).astype(np.float32)

    data = [
        {"id": i, "vector": vectors[i].tolist(), "text": f"document_{i}"}
        for i in range(NUM_ENTITIES)
    ]

    res = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"✅ Inserted {NUM_ENTITIES} records")

    # ==================== 4. Create DiskANN Index ====================
    print("\n" + "=" * 60)
    print("4. Creating DiskANN index")
    print("=" * 60)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        index_type="DISKANN",
        metric_type="L2",
        params={}
    )

    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )
    print("✅ DiskANN index created!")

    index_info = client.describe_index(collection_name=COLLECTION_NAME, index_name="vector")
    print(f"   Index info: {index_info}")

    # ==================== 5. Load Collection ====================
    print("\n" + "=" * 60)
    print("5. Loading collection into memory")
    print("=" * 60)

    client.load_collection(collection_name=COLLECTION_NAME)
    print(f"✅ Collection {COLLECTION_NAME} loaded")

    # ==================== 6. Vector Search ====================
    print("\n" + "=" * 60)
    print("6. Executing vector search")
    print("=" * 60)

    query_vectors = np.random.random((1, DIMENSION)).astype(np.float32).tolist()

    search_params = {
        "metric_type": "L2",
        "params": {"search_list": 100}
    }

    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vectors,
        limit=5,
        search_params=search_params,
        output_fields=["id", "text"]
    )

    print("🔍 Search results:")
    print("-" * 40)
    for i, hits in enumerate(search_res):
        print(f"Query {i + 1}:")
        for hit in hits:
            print(f"  ID: {hit['id']}, Distance: {hit['distance']:.4f}, Text: {hit['entity']['text']}")

    # ==================== 7. Self-search Test ====================
    print("\n" + "=" * 60)
    print("7. Self-search test (searching with known vector)")
    print("=" * 60)

    known_vector = vectors[0].tolist()

    search_res2 = client.search(
        collection_name=COLLECTION_NAME,
        data=[known_vector],
        limit=5,
        search_params=search_params,
        output_fields=["id", "text"]
    )

    print("🔍 Searching with ID=0 vector (should return itself):")
    print("-" * 40)
    for hit in search_res2[0]:
        print(f"  ID: {hit['id']}, Distance: {hit['distance']:.4f}, Text: {hit['entity']['text']}")

    # ==================== 8. Statistics ====================
    print("\n" + "=" * 60)
    print("8. Collection statistics")
    print("=" * 60)

    collection_info = client.describe_collection(collection_name=COLLECTION_NAME)
    print(f"   Collection: {collection_info['collection_name']}")
    print(f"   Fields: {len(collection_info['fields'])}")

    stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
    print(f"   Total rows: {stats['row_count']}")

    # ==================== 9. Cleanup ====================
    print("\n" + "=" * 60)
    print("9. Cleanup")
    print("=" * 60)

    client.release_collection(collection_name=COLLECTION_NAME)
    print(f"✅ Collection {COLLECTION_NAME} released from memory")

    # Uncomment to delete collection
    # client.drop_collection(COLLECTION_NAME)
    # print(f"✅ Collection {COLLECTION_NAME} deleted")

    client.close()
    print("\n" + "=" * 60)
    print("🎉 DiskANN demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
