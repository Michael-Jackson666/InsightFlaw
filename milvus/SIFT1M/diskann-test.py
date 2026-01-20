"""
SIFT1M DiskANN 性能测试脚本
============================
环境: macOS + Docker Milvus Standalone
数据集: SIFT1M (100万条 128维向量)

测试指标:
- QPS (每秒查询数)
- Latency (平均延迟)  
- Recall@K (召回率)
"""

import numpy as np
import time
import os

# ================= 配置区域 =================
URI = "http://localhost:19530"  # Milvus Standalone 地址
COLLECTION_NAME = "sift1m_diskann_test"
DIMENSION = 128  # SIFT 数据集是 128 维

# 获取脚本所在目录，确保相对路径正确
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HDF5_PATH = os.path.join(SCRIPT_DIR, "sift-128-euclidean.hdf5")

# ================= 工具函数 =================
def load_hdf5_dataset(filepath):
    """从 HDF5 文件加载 SIFT 数据集 (ann-benchmarks 格式)"""
    import h5py
    
    with h5py.File(filepath, 'r') as f:
        # ann-benchmarks 格式的 key
        train = np.array(f['train'])      # 底库向量
        test = np.array(f['test'])        # 查询向量
        neighbors = np.array(f['neighbors'])  # ground truth (最近邻 ID)
        distances = np.array(f['distances'])  # ground truth 距离
    
    return train, test, neighbors, distances

def check_dependencies():
    """检查必要的依赖"""
    try:
        import h5py
        return True
    except ImportError:
        print("❌ 缺少 h5py 库，请先安装:")
        print("   pip install h5py")
        return False

def check_dataset():
    """检查数据集是否存在"""
    if not os.path.exists(HDF5_PATH):
        print(f"❌ 数据集文件不存在: {HDF5_PATH}")
        print("\n请先下载数据集:")
        print(f"   cd {SCRIPT_DIR}")
        print('   curl -L -o sift-128-euclidean.hdf5 "https://ann-benchmarks.com/sift-128-euclidean.hdf5"')
        return False
    return True

# ================= 主程序 =================
def main():
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查数据集
    if not check_dataset():
        return
    
    # 导入 pymilvus (放在这里避免在检查失败时报错)
    from pymilvus import MilvusClient
    
    # 1. 加载数据到内存
    print("=" * 60)
    print("📂 正在读取 SIFT1M 数据集 (HDF5 格式)...")
    print("=" * 60)
    
    xb, xq, gt, gt_distances = load_hdf5_dataset(HDF5_PATH)

    print(f"   底库数据: {xb.shape} ({xb.nbytes / 1024 / 1024:.1f} MB)")
    print(f"   查询数据: {xq.shape}")
    print(f"   标准答案: {gt.shape}")

    # 2. 初始化 Milvus
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

    # 重建集合
    if client.has_collection(COLLECTION_NAME):
        print(f"\n⚠️ 删除已存在的集合: {COLLECTION_NAME}")
        client.drop_collection(COLLECTION_NAME)

    # 使用 Schema 方式创建集合，手动指定 ID 以匹配 ground truth
    from pymilvus import DataType
    
    print(f"\n📦 创建集合: {COLLECTION_NAME}")
    
    # auto_id=False: 手动指定 ID，确保与 ground truth 中的行号一致
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    
    # 先准备 DiskANN 索引参数
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="DISKANN", 
        metric_type="L2",
        index_name="vector_index"
    )
    
    # 创建集合时同时指定索引
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params  # 直接使用 DiskANN 索引
    )
    print("   ✅ 集合创建成功 (使用 DiskANN 索引)")
    print("   ℹ️  使用手动 ID (0 ~ 999999) 以匹配 ground truth")

    # 3. 插入数据 (分批插入，包含手动 ID)
    print("\n" + "=" * 60)
    print("🚀 开始插入数据 (1M 条)...")
    print("=" * 60)
    
    batch_size = 10000
    total_count = len(xb)
    insert_start = time.time()

    for i in range(0, total_count, batch_size):
        end = min(i + batch_size, total_count)
        # 转换 numpy 数组为 list，包含手动指定的 ID (行号)
        batch_data = [
            {"id": i + j, "vector": xb[i + j].tolist()} 
            for j in range(end - i)
        ]
        client.insert(collection_name=COLLECTION_NAME, data=batch_data)
        progress = (end / total_count) * 100
        print(f"   进度: {end:>7}/{total_count} ({progress:.1f}%)", end="\r")

    insert_time = time.time() - insert_start
    print(f"\n✅ 数据插入完成! 耗时: {insert_time:.2f} 秒")
    print(f"   插入速度: {total_count / insert_time:.0f} 条/秒")

    # 4. 刷新并等待索引构建
    print("\n" + "=" * 60)
    print("🔨 正在构建 DiskANN 索引...")
    print("=" * 60)
    print("   ⏳ 这可能需要几分钟，请耐心等待...")
    
    start_idx = time.time()
    client.flush(collection_name=COLLECTION_NAME)
    
    # 等待索引构建完成
    import time as t
    while True:
        index_info = client.describe_index(collection_name=COLLECTION_NAME, index_name="vector_index")
        indexed_rows = index_info.get('indexed_rows', 0)
        total_rows = index_info.get('total_rows', total_count)
        if indexed_rows >= total_count:
            break
        print(f"   索引进度: {indexed_rows}/{total_rows}", end="\r")
        t.sleep(2)
    
    index_time = time.time() - start_idx
    print(f"\n✅ 索引构建完成! 耗时: {index_time:.2f} 秒")

    # 5. 加载集合
    print("\n📥 加载集合到内存...")
    client.load_collection(COLLECTION_NAME)
    print("✅ 集合已加载")

    # ================= 性能测试与召回率计算 =================
    # 定义搜索参数
    # 注意: DiskANN 要求 search_list >= limit (TopK)
    TOP_K = 100  # SIFT 标准通常计算 Top 100 的召回率
    SEARCH_PARAMS = {
        "metric_type": "L2", 
        "params": {"search_list": 150}  # 必须 >= TOP_K，越大召回率越高但速度越慢
    }
    NUM_QUERIES = 1000  # 测试查询数量

    print("\n" + "=" * 60)
    print(f"⚡ 开始性能测试")
    print("=" * 60)
    print(f"   TopK: {TOP_K}")
    print(f"   search_list: {SEARCH_PARAMS['params']['search_list']}")
    print(f"   测试查询数: {NUM_QUERIES}")
    print(f"   ⚠️  注意: DiskANN 要求 search_list >= TopK")

    # 预热 (Warmup) - 让系统加载缓存
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

    # ================= 结果分析 =================
    total_queries = len(results)
    total_time = end_time - start_time
    qps = total_queries / total_time
    avg_latency = (total_time / total_queries) * 1000  # 毫秒

    # 计算召回率 (Recall@K)
    recall_count = 0
    for i, hits in enumerate(results):
        # 获取 Milvus 返回的 ID 列表
        result_ids = set([hit['id'] for hit in hits])
        # 获取标准答案的 ID 列表 (取前 TopK)
        ground_truth_ids = set(gt[i, :TOP_K])
        
        # 计算交集
        intersection = result_ids.intersection(ground_truth_ids)
        recall_count += len(intersection)

    recall_rate = recall_count / (total_queries * TOP_K)

    # 打印结果报告
    print("\n" + "=" * 60)
    print("📊 性能测试结果")
    print("=" * 60)
    print(f"   {'指标':<20} {'结果':<20}")
    print("-" * 60)
    print(f"   {'QPS (吞吐量)':<20} {qps:.2f} queries/sec")
    print(f"   {'平均延迟':<20} {avg_latency:.2f} ms/query")
    print(f"   {'Recall@' + str(TOP_K):<20} {recall_rate:.4f} ({recall_rate*100:.2f}%)")
    print("-" * 60)
    print(f"   {'总查询数':<20} {total_queries}")
    print(f"   {'总耗时':<20} {total_time:.2f} s")
    print("=" * 60)

    # 不同 search_list 对比测试
    # 注意: search_list 必须 >= TOP_K (100)
    print("\n📈 search_list 参数对比测试 (search_list >= TopK):")
    print("-" * 50)
    print(f"{'search_list':<15} {'QPS':<15} {'Recall@100':<15}")
    print("-" * 50)
    
    for sl in [100, 150, 200, 300]:  # 所有值都 >= TOP_K (100)
        params = {"metric_type": "L2", "params": {"search_list": sl}}
        
        start = time.time()
        res = client.search(
            collection_name=COLLECTION_NAME,
            data=query_vectors[:100],  # 用 100 个查询快速测试
            limit=TOP_K,
            search_params=params
        )
        elapsed = time.time() - start
        
        # 计算召回率
        recall = 0
        for i, hits in enumerate(res):
            result_ids = set([hit['id'] for hit in hits])
            ground_truth_ids = set(gt[i, :TOP_K])
            recall += len(result_ids.intersection(ground_truth_ids))
        recall_pct = recall / (100 * TOP_K)
        
        qps_test = 100 / elapsed
        print(f"{sl:<15} {qps_test:<15.2f} {recall_pct:<15.4f}")
    
    print("-" * 50)
    print("💡 结论: search_list 越大，召回率越高，但 QPS 会降低")

    # 清理（可选）
    print("\n🧹 清理资源...")
    client.release_collection(COLLECTION_NAME)
    # client.drop_collection(COLLECTION_NAME)  # 取消注释以删除集合
    client.close()
    print("✅ 测试完成!")

if __name__ == "__main__":
    main()