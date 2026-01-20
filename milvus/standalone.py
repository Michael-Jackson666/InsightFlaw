"""
Milvus Standalone Connection Test
=================================
Test script to verify connection to Milvus Docker Standalone instance.

Usage:
    python standalone.py

Prerequisites:
    - Docker running with Milvus Standalone
    - pip install pymilvus
"""

from pymilvus import MilvusClient

# Configuration - modify as needed
MILVUS_URI = "http://localhost:19530"

def main():
    """Test connection to Milvus Standalone."""
    print("=" * 50)
    print("Milvus Standalone Connection Test")
    print("=" * 50)
    
    try:
        # Connect to Milvus Docker Standalone
        client = MilvusClient(uri=MILVUS_URI)
        
        print(f"✅ Successfully connected to Milvus!")
        print(f"   URI: {MILVUS_URI}")
        print(f"   Server Version: {client.get_server_version()}")
        
        # List existing collections
        collections = client.list_collections()
        print(f"\n📦 Existing collections: {collections}")
        
        client.close()
        print("\n✅ Connection test passed!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Make sure Milvus Docker is running:")
        print("   docker-compose up -d")
        return False
    
    return True

if __name__ == "__main__":
    main()