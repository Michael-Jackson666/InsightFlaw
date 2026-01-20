"""
Milvus Lite Quick Test
======================
Demonstrates basic Milvus Lite operations: create, insert, search.

Usage:
    python test-lite.py

This script creates a local database file (milvus_demo.db) and performs
basic vector database operations without requiring a server.
"""

import pymilvus
from pymilvus import MilvusClient

# Configuration
DB_PATH = "./milvus_demo.db"
COLLECTION_NAME = "demo_collection"
VECTOR_DIM = 768  # BERT-like embedding dimension

def main():
    """Run Milvus Lite demo."""
    print("=" * 50)
    print("Milvus Lite Quick Test")
    print("=" * 50)
    print(f"pymilvus version: {pymilvus.__version__}")
    
    # 1. Initialize Milvus Lite
    client = MilvusClient(uri=DB_PATH)
    print(f"\n✅ Milvus Lite initialized")
    print(f"   Database: {DB_PATH}")

    # 2. Create collection
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"   Dropped existing collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=VECTOR_DIM
    )
    print(f"✅ Created collection: {COLLECTION_NAME}")
    print(f"   Dimension: {VECTOR_DIM}")

    # 3. Insert sample data
    data = [
        {"id": i, "vector": [0.1] * VECTOR_DIM, "text": f"document_{i}", "subject": "history"}
        for i in range(10)
    ]
    res = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"\n✅ Inserted {len(data)} records")

    # 4. Vector search
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[[0.1] * VECTOR_DIM],
        limit=3,
        output_fields=["text", "subject"]
    )
    print(f"\n🔍 Search results (top 3):")
    for i, hits in enumerate(search_res):
        for hit in hits:
            print(f"   - {hit}")

    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()