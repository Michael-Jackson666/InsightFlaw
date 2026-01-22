"""
数据集定义
==========
定义所有支持的 ann-benchmarks 数据集
"""

import os
from pathlib import Path

# 数据目录
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"

# ================= 数据集定义 =================
DATASETS = {
    # ===== HDF5 格式数据集 (ann-benchmarks) =====
    "sift": {
        "name": "SIFT-128",
        "filename": "sift-128-euclidean.hdf5",
        "url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "dimension": 128,
        "metric": "L2",
        "vectors": "1M",
        "size": "~500 MB",
        "format": "hdf5",
        "description": "经典 SIFT 特征，128维，100万向量（推荐入门）",
    },
    "gist": {
        "name": "GIST-960",
        "filename": "gist-960-euclidean.hdf5",
        "url": "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "dimension": 960,
        "metric": "L2",
        "vectors": "1M",
        "size": "~3.6 GB",
        "format": "hdf5",
        "description": "GIST 图像描述符，960维，100万向量（高维度测试）",
    },
    "glove-25": {
        "name": "GloVe-25",
        "filename": "glove-25-angular.hdf5",
        "url": "https://ann-benchmarks.com/glove-25-angular.hdf5",
        "dimension": 25,
        "metric": "IP",
        "vectors": "1.2M",
        "size": "~100 MB",
        "format": "hdf5",
        "description": "GloVe 词向量，25维，120万向量（低维度）",
    },
    "glove-100": {
        "name": "GloVe-100",
        "filename": "glove-100-angular.hdf5",
        "url": "https://ann-benchmarks.com/glove-100-angular.hdf5",
        "dimension": 100,
        "metric": "IP",
        "vectors": "1.2M",
        "size": "~460 MB",
        "format": "hdf5",
        "description": "GloVe 词向量，100维，120万向量",
    },
    "fashion-mnist": {
        "name": "Fashion-MNIST-784",
        "filename": "fashion-mnist-784-euclidean.hdf5",
        "url": "https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "dimension": 784,
        "metric": "L2",
        "vectors": "60K",
        "size": "~200 MB",
        "format": "hdf5",
        "description": "Fashion MNIST 图像，784维，6万向量（小规模测试）",
    },
    "nytimes": {
        "name": "NYTimes-256",
        "filename": "nytimes-256-angular.hdf5",
        "url": "https://ann-benchmarks.com/nytimes-256-angular.hdf5",
        "dimension": 256,
        "metric": "IP",
        "vectors": "290K",
        "size": "~280 MB",
        "format": "hdf5",
        "description": "NYTimes 文章向量，256维，29万向量",
    },
    
    # ===== SIFT1B 大规模数据集 (bvecs 格式) =====
    "sift1b": {
        "name": "SIFT1B (BigANN)",
        "filename": "bigann_base.bvecs",
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz",
        "query_url": "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz",
        "gnd_url": "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz",
        "dimension": 128,
        "metric": "L2",
        "vectors": "1B",
        "size": "~128 GB",
        "format": "bvecs",
        "description": "10亿级 SIFT 向量，需要大量存储（生产级测试）",
    },
}

# 向量数量预设 (用于 SIFT1B)
VECTOR_PRESETS = {
    "1M": 1_000_000,
    "10M": 10_000_000,
    "100M": 100_000_000,
    "500M": 500_000_000,
    "1B": 1_000_000_000,
}


def get_dataset_path(dataset_key: str) -> Path:
    """获取数据集文件路径"""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    
    dataset = DATASETS[dataset_key]
    return DATA_DIR / dataset["filename"]


def list_datasets():
    """列出所有可用数据集"""
    print("\n📦 可用数据集:")
    print("=" * 80)
    print(f"{'ID':<15} {'名称':<20} {'维度':<8} {'向量数':<10} {'大小':<12} {'距离':<6}")
    print("-" * 80)
    
    for key, ds in DATASETS.items():
        print(f"{key:<15} {ds['name']:<20} {ds['dimension']:<8} {ds['vectors']:<10} {ds['size']:<12} {ds['metric']:<6}")
    
    print("-" * 80)
    print("\n💡 使用方法: python diskann-test.py --dataset <ID>")
    print("   例如: python diskann-test.py --dataset sift")
    print("         python diskann-test.py --dataset gist")
    print("         python diskann-test.py --dataset sift1b -n 10M")


def check_dataset_exists(dataset_key: str) -> bool:
    """检查数据集是否已下载"""
    filepath = get_dataset_path(dataset_key)
    return filepath.exists()


def get_dataset_info(dataset_key: str) -> dict:
    """获取数据集信息"""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    return DATASETS[dataset_key]
