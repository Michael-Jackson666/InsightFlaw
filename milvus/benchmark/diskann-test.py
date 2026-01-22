"""
DiskANN 多数据集性能测试脚本
==============================
支持多种 ann-benchmarks 数据集的统一测试框架

使用方法:
    python diskann-test.py --list                    # 列出所有数据集
    python diskann-test.py --dataset sift            # 测试 SIFT-128
    python diskann-test.py --dataset gist            # 测试 GIST-960
    python diskann-test.py --dataset sift1b -n 10M   # 测试 SIFT1B 子集
    python diskann-test.py --dataset sift --download # 下载数据集
"""

import argparse
import numpy as np
import struct
import subprocess
import time
import os
import sys
from pathlib import Path

# 导入数据集定义
from datasets import (
    DATASETS,
    VECTOR_PRESETS,
    DATA_DIR,
    list_datasets,
    get_dataset_path,
    get_dataset_info,
    check_dataset_exists,
)

# ================= 配置 =================
DEFAULT_URI = "http://localhost:19530"
DEFAULT_BATCH_SIZE = 50000


# ================= 数据加载函数 =================
def load_hdf5_dataset(filepath: Path):
    """从 HDF5 文件加载数据集 (ann-benchmarks 格式)"""
    import h5py
    
    print(f"📂 正在读取 HDF5 数据集: {filepath.name}")
    
    with h5py.File(filepath, 'r') as f:
        train = np.array(f['train'])        # 底库向量
        test = np.array(f['test'])          # 查询向量
        neighbors = np.array(f['neighbors'])  # ground truth
        distances = np.array(f['distances'])  # ground truth 距离
    
    print(f"   底库向量: {train.shape} ({train.nbytes / 1024 / 1024:.1f} MB)")
    print(f"   查询向量: {test.shape}")
    print(f"   Ground Truth: {neighbors.shape}")
    
    return train, test, neighbors, distances


def read_bvecs_batch(filepath: Path, start_idx: int, count: int, dim: int = 128):
    """从 bvecs 文件批量读取向量"""
    vector_size = 4 + dim
    
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


def read_bvecs_all(filepath: Path, dim: int = 128):
    """读取全部 bvecs 向量"""
    vector_size = 4 + dim
    file_size = filepath.stat().st_size
    num_vectors = file_size // vector_size
    
    return read_bvecs_batch(filepath, 0, num_vectors, dim)


def read_ivecs(filepath: Path):
    """读取 ivecs 格式的 ground truth"""
    with open(filepath, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        
        vector_size = 4 + dim * 4
        file_size = filepath.stat().st_size
        num_vectors = file_size // vector_size
        
        result = np.zeros((num_vectors, dim), dtype=np.int32)
        
        for i in range(num_vectors):
            d = struct.unpack('i', f.read(4))[0]
            result[i] = np.frombuffer(f.read(d * 4), dtype=np.int32)
        
        return result


# ================= 下载函数 =================
def download_dataset(dataset_key: str):
    """下载指定数据集"""
    if dataset_key not in DATASETS:
        print(f"❌ 未知数据集: {dataset_key}")
        return False
    
    dataset = DATASETS[dataset_key]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    filepath = get_dataset_path(dataset_key)
    
    if filepath.exists():
        print(f"✅ 数据集已存在: {filepath.name}")
        return True
    
    url = dataset["url"]
    print(f"\n📥 下载 {dataset['name']}...")
    print(f"   URL: {url}")
    print(f"   大小: {dataset['size']}")
    
    try:
        if dataset["format"] == "hdf5":
            # 使用 curl 下载 HDF5
            cmd = ["curl", "-L", "-o", str(filepath), url]
            subprocess.run(cmd, check=True)
            print(f"✅ 下载完成: {filepath.name}")
            return True
        
        elif dataset["format"] == "bvecs":
            # SIFT1B 需要下载多个文件
            print("⚠️  SIFT1B 数据集较大 (~128GB)，请手动下载:")
            print(f"   cd {DATA_DIR}")
            print(f"   wget {url}")
            print(f"   gunzip {dataset['filename']}.gz")
            print(f"   wget {dataset.get('query_url', '')}")
            print(f"   wget {dataset.get('gnd_url', '')}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ curl 未找到，请安装 curl")
        return False


# ================= 主测试函数 =================
def run_benchmark(
    dataset_key: str,
    num_vectors: int = None,
    uri: str = DEFAULT_URI,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hdf5_path: str = None,
):
    """运行基准测试"""
    from pymilvus import MilvusClient, DataType
    
    # 获取数据集信息
    if hdf5_path:
        # 自定义 HDF5 文件
        filepath = Path(hdf5_path)
        xb, xq, gt, _ = load_hdf5_dataset(filepath)
        dimension = xb.shape[1]
        metric_type = "L2"
        collection_name = f"custom_benchmark"
    else:
        dataset = get_dataset_info(dataset_key)
        filepath = get_dataset_path(dataset_key)
        dimension = dataset["dimension"]
        metric_type = dataset["metric"]
        collection_name = f"{dataset_key}_benchmark"
        
        # 检查数据集
        if not filepath.exists():
            print(f"❌ 数据集不存在: {filepath}")
            print(f"   请先下载: python diskann-test.py --dataset {dataset_key} --download")
            return
        
        # 加载数据
        if dataset["format"] == "hdf5":
            xb, xq, gt, _ = load_hdf5_dataset(filepath)
            if num_vectors:
                num_vectors = min(num_vectors, len(xb))
                xb = xb[:num_vectors]
            else:
                num_vectors = len(xb)
        
        elif dataset["format"] == "bvecs":
            # SIFT1B
            query_path = DATA_DIR / "bigann_query.bvecs"
            if not query_path.exists():
                print(f"❌ 查询文件不存在: {query_path}")
                return
            
            xq = read_bvecs_all(query_path, dimension)
            gt = None  # TODO: 加载 ground truth
            xb = None  # 延迟加载
            
            if num_vectors is None:
                num_vectors = 1_000_000_000  # 默认全部
    
    print("\n" + "=" * 70)
    print(f"🚀 DiskANN 性能测试")
    print("=" * 70)
    print(f"   数据集: {dataset_key if not hdf5_path else hdf5_path}")
    print(f"   向量数: {num_vectors:,}")
    print(f"   维度: {dimension}")
    print(f"   距离类型: {metric_type}")
    print("=" * 70)
    
    # 连接 Milvus
    print("\n🔌 连接 Milvus...")
    try:
        client = MilvusClient(uri=uri)
        print(f"   ✅ 已连接: {uri}")
        print(f"   📦 服务器版本: {client.get_server_version()}")
    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
        print("   请确保 Milvus 正在运行")
        return
    
    # 创建集合
    if client.has_collection(collection_name):
        print(f"\n⚠️  删除已存在的集合: {collection_name}")
        client.drop_collection(collection_name)
    
    print(f"\n📦 创建集合: {collection_name}")
    
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    
    # DiskANN 索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="DISKANN",
        metric_type=metric_type,
        index_name="vector_index"
    )
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print(f"   ✅ 集合创建成功 (DiskANN, {metric_type})")
    
    # 插入数据
    print("\n" + "=" * 70)
    print(f"🚀 开始插入数据 ({num_vectors:,} 条)...")
    print("=" * 70)
    
    insert_start = time.time()
    
    for i in range(0, num_vectors, batch_size):
        end = min(i + batch_size, num_vectors)
        
        if xb is not None:
            batch_vectors = xb[i:end]
        else:
            # bvecs 流式读取
            batch_vectors = read_bvecs_batch(filepath, i, end - i, dimension)
        
        batch_data = [
            {"id": i + j, "vector": batch_vectors[j].tolist()}
            for j in range(len(batch_vectors))
        ]
        
        client.insert(collection_name=collection_name, data=batch_data)
        
        progress = (end / num_vectors) * 100
        elapsed = time.time() - insert_start
        rate = end / elapsed if elapsed > 0 else 0
        eta = (num_vectors - end) / rate if rate > 0 else 0
        
        print(f"   进度: {end:>12,} / {num_vectors:,} ({progress:5.1f}%) | "
              f"{rate:,.0f} vec/s | ETA: {eta/60:.1f} min", end="\r")
    
    insert_time = time.time() - insert_start
    print(f"\n✅ 插入完成! 耗时: {insert_time:.1f}s ({insert_time/60:.1f} min)")
    
    # 构建索引
    print("\n" + "=" * 70)
    print("🔨 正在构建 DiskANN 索引...")
    print("=" * 70)
    
    index_start = time.time()
    client.flush(collection_name=collection_name)
    
    while True:
        info = client.describe_index(collection_name=collection_name, index_name="vector_index")
        indexed = info.get('indexed_rows', 0)
        total = info.get('total_rows', num_vectors)
        
        if indexed >= num_vectors:
            break
        
        print(f"   索引进度: {indexed:,} / {total:,} ({indexed/total*100:.1f}%)", end="\r")
        time.sleep(2)
    
    index_time = time.time() - index_start
    print(f"\n✅ 索引完成! 耗时: {index_time:.1f}s ({index_time/60:.1f} min)")
    
    # 加载集合
    print("\n📥 加载集合...")
    client.load_collection(collection_name)
    print("   ✅ 加载完成")
    
    # 性能测试
    TOP_K = 100
    SEARCH_PARAMS = {"metric_type": metric_type, "params": {"search_list": 150}}
    NUM_QUERIES = min(1000, len(xq))
    
    print("\n" + "=" * 70)
    print("⚡ 性能测试")
    print("=" * 70)
    
    # 预热
    print("   🔥 预热...")
    for _ in range(10):
        client.search(
            collection_name=collection_name,
            data=[xq[0].tolist()],
            limit=TOP_K,
            search_params=SEARCH_PARAMS
        )
    
    # 正式测试
    print("   🔍 执行搜索...")
    query_vectors = [vec.tolist() for vec in xq[:NUM_QUERIES]]
    
    start_time = time.time()
    results = client.search(
        collection_name=collection_name,
        data=query_vectors,
        limit=TOP_K,
        search_params=SEARCH_PARAMS
    )
    end_time = time.time()
    
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
    print("📊 测试结果")
    print("=" * 70)
    print(f"   {'数据集':<20} {dataset_key}")
    print(f"   {'向量数':<20} {num_vectors:,}")
    print(f"   {'维度':<20} {dimension}")
    print(f"   {'QPS':<20} {qps:.2f} queries/sec")
    print(f"   {'平均延迟':<20} {avg_latency:.2f} ms")
    if recall_rate:
        print(f"   {'Recall@' + str(TOP_K):<20} {recall_rate:.4f} ({recall_rate*100:.2f}%)")
    print("-" * 70)
    print(f"   {'插入耗时':<20} {insert_time/60:.1f} min")
    print(f"   {'索引耗时':<20} {index_time/60:.1f} min")
    print("=" * 70)
    
    # search_list 对比
    print("\n📈 search_list 参数对比:")
    print("-" * 50)
    print(f"{'search_list':<15} {'QPS':<15} {'Recall@100':<15}")
    print("-" * 50)
    
    for sl in [100, 150, 200, 300]:
        params = {"metric_type": metric_type, "params": {"search_list": sl}}
        
        start = time.time()
        res = client.search(
            collection_name=collection_name,
            data=query_vectors[:100],
            limit=TOP_K,
            search_params=params
        )
        elapsed = time.time() - start
        
        recall = "N/A"
        if gt is not None:
            cnt = 0
            for i, hits in enumerate(res):
                result_ids = set([hit['id'] for hit in hits])
                gt_ids = set(gt[i, :TOP_K])
                cnt += len(result_ids.intersection(gt_ids))
            recall = f"{cnt / (100 * TOP_K):.4f}"
        
        print(f"{sl:<15} {100/elapsed:<15.2f} {recall:<15}")
    
    print("-" * 50)
    
    # 清理
    print("\n🧹 清理...")
    client.release_collection(collection_name)
    client.close()
    print("✅ 测试完成!")


# ================= 入口 =================
def main():
    parser = argparse.ArgumentParser(
        description="DiskANN 多数据集性能测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python diskann-test.py --list                    # 列出数据集
  python diskann-test.py --dataset sift            # 测试 SIFT-128
  python diskann-test.py --dataset gist            # 测试 GIST-960
  python diskann-test.py --dataset sift1b -n 10M   # SIFT1B 子集
  python diskann-test.py --dataset sift --download # 下载数据集
        """
    )
    
    parser.add_argument("--list", "-l", action="store_true",
                       help="列出所有可用数据集")
    parser.add_argument("--dataset", "-d", type=str,
                       help="选择数据集 (sift, gist, glove-100, sift1b 等)")
    parser.add_argument("--download", action="store_true",
                       help="下载指定数据集")
    parser.add_argument("-n", "--vectors", type=str,
                       help="向量数量 (1M, 10M, 100M, 1B)")
    parser.add_argument("--hdf5", type=str,
                       help="使用自定义 HDF5 文件")
    parser.add_argument("--uri", type=str, default=DEFAULT_URI,
                       help="Milvus 服务器地址")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="插入批次大小")
    
    args = parser.parse_args()
    
    # 列出数据集
    if args.list:
        list_datasets()
        return
    
    # 下载数据集
    if args.download:
        if not args.dataset:
            print("❌ 请指定数据集: --dataset <ID>")
            return
        download_dataset(args.dataset)
        return
    
    # 检查参数
    if not args.dataset and not args.hdf5:
        parser.print_help()
        print("\n💡 提示: 使用 --list 查看可用数据集")
        return
    
    # 解析向量数量
    num_vectors = None
    if args.vectors:
        num_vectors = VECTOR_PRESETS.get(args.vectors.upper())
        if num_vectors is None:
            try:
                num_vectors = int(args.vectors)
            except ValueError:
                print(f"❌ 无效的向量数量: {args.vectors}")
                return
    
    # 检查依赖
    try:
        import h5py
    except ImportError:
        print("❌ 缺少 h5py，请安装: pip install h5py")
        return
    
    # 运行测试
    run_benchmark(
        dataset_key=args.dataset,
        num_vectors=num_vectors,
        uri=args.uri,
        batch_size=args.batch_size,
        hdf5_path=args.hdf5,
    )


if __name__ == "__main__":
    main()
