"""
SIFT1B DiskANN 性能测试脚本
============================
环境: macOS + Docker Milvus Standalone
数据集: SIFT1B (10亿条 128维向量)

测试指标:
- QPS (每秒查询数)
- Latency (平均延迟)  
- Recall@K (召回率)

使用方法:
    python diskann-test.py              # 默认使用全部数据
    python diskann-test.py -n 10M       # 使用 1000 万向量测试
    python diskann-test.py -n 100M      # 使用 1 亿向量测试
    python diskann-test.py -n 1B        # 使用 10 亿向量测试
    python diskann-test.py --hdf5 xxx   # 使用 HDF5 格式数据集
"""

import argparse
import numpy as np
import struct
import time
import os

# ================= 配置区域 =================
URI = "http://localhost:19530"  # Milvus Standalone 地址
COLLECTION_NAME = "sift1b_diskann_test"
DIMENSION = 128  # SIFT 数据集是 128 维
BATCH_SIZE = 50000  # 插入批次大小

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# 数据文件路径
BVECS_BASE_PATH = os.path.join(DATA_DIR, "bigann_base.bvecs")
BVECS_QUERY_PATH = os.path.join(DATA_DIR, "bigann_query.bvecs")
GND_DIR = os.path.join(DATA_DIR, "bigann_gnd")

# 向量数量预设
PRESETS = {
    "10M": 10_000_000,
    "100M": 100_000_000,
    "500M": 500_000_000,
    "1B": 1_000_000_000,
}


# ================= 数据加载函数 =================
def load_hdf5_dataset(filepath):
    """从 HDF5 文件加载数据集 (ann-benchmarks 格式)"""
    import h5py
    
    print(f"📂 正在读取 HDF5 数据集: {os.path.basename(filepath)}")
    
    with h5py.File(filepath, 'r') as f:
        train = np.array(f['train'])      # 底库向量
        test = np.array(f['test'])        # 查询向量
        neighbors = np.array(f['neighbors'])  # ground truth
        distances = np.array(f['distances'])  # ground truth 距离
    
    return train, test, neighbors, distances


def read_bvecs_batch(filepath, start_idx, count, dim=128):
    """
    从 bvecs 文件批量读取向量
    
    bvecs 格式: 每个向量 = [dim (4 bytes int)] + [dim * uint8 values]
    """
    vector_size = 4 + dim  # 4 bytes for dimension + dim bytes for vector
    
    with open(filepath, 'rb') as f:
        f.seek(start_idx * vector_size)
        
        vectors = np.zeros((count, dim), dtype=np.float32)
        
        for i in range(count):
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            
            d = struct.unpack('i', dim_bytes)[0]
            assert d == dim, f"Dimension mismatch: expected {dim}, got {d}"
            
            vec_bytes = f.read(dim)
            vectors[i] = np.frombuffer(vec_bytes, dtype=np.uint8).astype(np.float32)
        
        return vectors


def read_bvecs_query(filepath, dim=128):
    """读取全部查询向量"""
    vector_size = 4 + dim
    file_size = os.path.getsize(filepath)
    num_vectors = file_size // vector_size
    
    return read_bvecs_batch(filepath, 0, num_vectors, dim)


def read_ivecs(filepath):
    """读取 ivecs 格式的 ground truth"""
    with open(filepath, 'rb') as f:
        # 读取第一个向量的维度
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        
        # 计算向量数量
        vector_size = 4 + dim * 4
        file_size = os.path.getsize(filepath)
        num_vectors = file_size // vector_size
        
        result = np.zeros((num_vectors, dim), dtype=np.int32)
        
        for i in range(num_vectors):
            d = struct.unpack('i', f.read(4))[0]
            result[i] = np.frombuffer(f.read(d * 4), dtype=np.int32)
        
        return result


def get_ground_truth(num_vectors):
    """根据向量数量选择对应的 ground truth 文件"""
    # BigANN 提供了不同规模的 ground truth
    gnd_files = {
        10_000_000: "idx_10M.ivecs",
        100_000_000: "idx_100M.ivecs",
        500_000_000: "idx_500M.ivecs",
        1_000_000_000: "idx_1000M.ivecs",
    }
    
    # 找到最接近的 ground truth
    for size, filename in sorted(gnd_files.items()):
        if num_vectors <= size:
            filepath = os.path.join(GND_DIR, filename)
            if os.path.exists(filepath):
                return read_ivecs(filepath)
            # 尝试 gnd 子目录
            filepath = os.path.join(GND_DIR, "gnd", filename)
            if os.path.exists(filepath):
                return read_ivecs(filepath)
    
    return None


# ================= 依赖检查 =================
def check_dependencies():
    """检查必要的依赖"""
    try:
        import h5py
        return True
    except ImportError:
        print("⚠️  h5py 未安装，HDF5 格式不可用")
        print("   pip install h5py")
        return False


def check_bvecs_dataset():
    """检查 bvecs 数据集是否存在"""
    if os.path.exists(BVECS_BASE_PATH):
        file_size = os.path.getsize(BVECS_BASE_PATH) / (1024**3)
        print(f"✅ 找到底库文件: bigann_base.bvecs ({file_size:.1f} GB)")
        return True
    return False


def check_query_dataset():
    """检查查询数据集"""
    if os.path.exists(BVECS_QUERY_PATH):
        return True
    return False


# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser(description="SIFT1B DiskANN 性能测试")
    parser.add_argument("-n", "--vectors", type=str, default="1B",
                       help="向量数量 (10M, 100M, 500M, 1B)")
    parser.add_argument("--hdf5", type=str, default=None,
                       help="使用 HDF5 格式数据集")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="插入批次大小")
    args = parser.parse_args()
    
    # 解析向量数量
    num_vectors = PRESETS.get(args.vectors.upper())
    if num_vectors is None:
        try:
            num_vectors = int(args.vectors)
        except ValueError:
            print(f"❌ 无效的向量数量: {args.vectors}")
            print(f"   可用预设: {list(PRESETS.keys())}")
            return
    
    print("=" * 70)
    print(f"🚀 SIFT1B DiskANN 性能测试")
    print(f"   目标向量数: {num_vectors:,}")
    print("=" * 70)
    
    # 加载数据
    if args.hdf5:
        # 使用 HDF5 格式
        if not os.path.exists(args.hdf5):
            print(f"❌ HDF5 文件不存在: {args.hdf5}")
            return
        
        xb, xq, gt, _ = load_hdf5_dataset(args.hdf5)
        num_vectors = min(num_vectors, len(xb))
        xb = xb[:num_vectors]
        
    else:
        # 使用 bvecs 格式
        if not check_bvecs_dataset():
            print(f"❌ 底库文件不存在: {BVECS_BASE_PATH}")
            print("\n请先下载数据集:")
            print(f"   cd {DATA_DIR}")
            print('   wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz')
            print('   gunzip bigann_base.bvecs.gz')
            return
        
        if not check_query_dataset():
            print(f"❌ 查询文件不存在: {BVECS_QUERY_PATH}")
            print("\n请先下载查询向量:")
            print(f"   cd {DATA_DIR}")
            print('   wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz')
            print('   gunzip bigann_query.bvecs.gz')
            return
        
        # 读取查询向量
        print("\n📂 正在读取查询向量...")
        xq = read_bvecs_query(BVECS_QUERY_PATH)
        print(f"   查询向量: {xq.shape}")
        
        # 读取 ground truth
        gt = get_ground_truth(num_vectors)
        if gt is not None:
            print(f"   Ground Truth: {gt.shape}")
        else:
            print("   ⚠️  未找到 Ground Truth，跳过召回率计算")
        
        xb = None  # 延迟加载底库向量
    
    # 导入 pymilvus
    from pymilvus import MilvusClient, DataType
    
    # 连接 Milvus
    print("\n🔌 连接 Milvus Standalone...")
    try:
        client = MilvusClient(uri=URI)
        print(f"   ✅ 已连接: {URI}")
        print(f"   📦 服务器版本: {client.get_server_version()}")
    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
        print("   请确保 Milvus Docker 容器正在运行:")
        print("   docker ps | grep milvus")
        return
    
    # 创建集合
    if client.has_collection(COLLECTION_NAME):
        print(f"\n⚠️  删除已存在的集合: {COLLECTION_NAME}")
        client.drop_collection(COLLECTION_NAME)
    
    print(f"\n📦 创建集合: {COLLECTION_NAME}")
    
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    
    # DiskANN 索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="DISKANN",
        metric_type="L2",
        index_name="vector_index"
    )
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print("   ✅ 集合创建成功 (DiskANN 索引)")
    
    # 插入数据
    print("\n" + "=" * 70)
    print(f"🚀 开始插入数据 ({num_vectors:,} 条)...")
    print("=" * 70)
    
    batch_size = args.batch_size
    insert_start = time.time()
    
    for i in range(0, num_vectors, batch_size):
        end = min(i + batch_size, num_vectors)
        
        # 加载批次数据
        if xb is not None:
            # HDF5 模式：从内存读取
            batch_vectors = xb[i:end]
        else:
            # bvecs 模式：从文件读取
            batch_vectors = read_bvecs_batch(BVECS_BASE_PATH, i, end - i, DIMENSION)
        
        batch_data = [
            {"id": i + j, "vector": batch_vectors[j].tolist()}
            for j in range(len(batch_vectors))
        ]
        
        client.insert(collection_name=COLLECTION_NAME, data=batch_data)
        
        progress = (end / num_vectors) * 100
        elapsed = time.time() - insert_start
        rate = end / elapsed if elapsed > 0 else 0
        eta = (num_vectors - end) / rate if rate > 0 else 0
        
        print(f"   进度: {end:>12,} / {num_vectors:,} ({progress:5.1f}%) | "
              f"{rate:,.0f} vec/s | ETA: {eta/60:.1f} min", end="\r")
    
    insert_time = time.time() - insert_start
    print(f"\n✅ 数据插入完成! 耗时: {insert_time:.1f}s ({insert_time/60:.1f} min)")
    print(f"   插入速度: {num_vectors / insert_time:,.0f} 条/秒")
    
    # 构建索引
    print("\n" + "=" * 70)
    print("🔨 正在构建 DiskANN 索引...")
    print("=" * 70)
    print("   ⏳ 这可能需要较长时间，请耐心等待...")
    
    index_start = time.time()
    client.flush(collection_name=COLLECTION_NAME)
    
    while True:
        info = client.describe_index(collection_name=COLLECTION_NAME, index_name="vector_index")
        indexed = info.get('indexed_rows', 0)
        total = info.get('total_rows', num_vectors)
        
        if indexed >= num_vectors:
            break
        
        print(f"   索引进度: {indexed:,} / {total:,} ({indexed/total*100:.1f}%)", end="\r")
        time.sleep(5)
    
    index_time = time.time() - index_start
    print(f"\n✅ 索引构建完成! 耗时: {index_time:.1f}s ({index_time/60:.1f} min)")
    
    # 加载集合
    print("\n📥 加载集合到内存...")
    client.load_collection(COLLECTION_NAME)
    print("   ✅ 集合已加载")
    
    # 性能测试
    TOP_K = 100
    SEARCH_PARAMS = {"metric_type": "L2", "params": {"search_list": 150}}
    NUM_QUERIES = min(1000, len(xq))
    
    print("\n" + "=" * 70)
    print("⚡ 开始性能测试")
    print("=" * 70)
    print(f"   TopK: {TOP_K}")
    print(f"   search_list: 150")
    print(f"   测试查询数: {NUM_QUERIES}")
    
    # 预热
    print("\n   🔥 预热中...")
    for _ in range(10):
        client.search(
            collection_name=COLLECTION_NAME,
            data=[xq[0].tolist()],
            limit=TOP_K,
            search_params=SEARCH_PARAMS
        )
    print("   ✅ 预热完成")
    
    # 正式测试
    print("\n   🔍 执行搜索测试...")
    query_vectors = [vec.tolist() for vec in xq[:NUM_QUERIES]]
    
    start_time = time.time()
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vectors,
        limit=TOP_K,
        search_params=SEARCH_PARAMS
    )
    end_time = time.time()
    
    # 结果分析
    total_time = end_time - start_time
    qps = NUM_QUERIES / total_time
    avg_latency = (total_time / NUM_QUERIES) * 1000
    
    # 计算召回率
    recall_rate = None
    if gt is not None:
        recall_count = 0
        for i, hits in enumerate(results):
            result_ids = set([hit['id'] for hit in hits])
            gt_ids = set(gt[i, :TOP_K])
            recall_count += len(result_ids.intersection(gt_ids))
        recall_rate = recall_count / (NUM_QUERIES * TOP_K)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("📊 性能测试结果")
    print("=" * 70)
    print(f"   {'指标':<20} {'结果':<20}")
    print("-" * 70)
    print(f"   {'向量总数':<20} {num_vectors:,}")
    print(f"   {'QPS (吞吐量)':<20} {qps:.2f} queries/sec")
    print(f"   {'平均延迟':<20} {avg_latency:.2f} ms/query")
    if recall_rate:
        print(f"   {'Recall@' + str(TOP_K):<20} {recall_rate:.4f} ({recall_rate*100:.2f}%)")
    print("-" * 70)
    print(f"   {'插入耗时':<20} {insert_time/60:.1f} min")
    print(f"   {'索引耗时':<20} {index_time/60:.1f} min")
    print("=" * 70)
    
    # search_list 参数对比
    print("\n📈 search_list 参数对比测试:")
    print("-" * 50)
    print(f"{'search_list':<15} {'QPS':<15} {'Recall@100':<15}")
    print("-" * 50)
    
    for sl in [100, 150, 200, 300]:
        params = {"metric_type": "L2", "params": {"search_list": sl}}
        
        start = time.time()
        res = client.search(
            collection_name=COLLECTION_NAME,
            data=query_vectors[:100],
            limit=TOP_K,
            search_params=params
        )
        elapsed = time.time() - start
        
        recall = "N/A"
        if gt is not None:
            recall_cnt = 0
            for i, hits in enumerate(res):
                result_ids = set([hit['id'] for hit in hits])
                gt_ids = set(gt[i, :TOP_K])
                recall_cnt += len(result_ids.intersection(gt_ids))
            recall = f"{recall_cnt / (100 * TOP_K):.4f}"
        
        qps_test = 100 / elapsed
        print(f"{sl:<15} {qps_test:<15.2f} {recall:<15}")
    
    print("-" * 50)
    print("💡 结论: search_list 越大，召回率越高，但 QPS 会降低")
    
    # 清理
    print("\n🧹 清理资源...")
    client.release_collection(COLLECTION_NAME)
    client.close()
    print("✅ 测试完成!")


if __name__ == "__main__":
    main()
