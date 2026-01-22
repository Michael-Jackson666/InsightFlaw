# ANN Benchmark - 多数据集向量检索测试

统一的向量数据库性能测试框架，支持从小规模到亿级的多种数据集。

## 支持的数据集

| 数据集 | 维度 | 向量数 | 大小 | 距离类型 | 说明 |
|--------|------|--------|------|----------|------|
| sift-128 | 128 | 1M | ~500 MB | L2 | 经典 SIFT 特征 |
| gist-960 | 960 | 1M | ~3.6 GB | L2 | GIST 图像描述符 |
| glove-25 | 25 | 1.2M | ~100 MB | IP | GloVe 词向量 |
| glove-100 | 100 | 1.2M | ~460 MB | IP | GloVe 词向量 |
| fashion-mnist | 784 | 60K | ~200 MB | L2 | Fashion MNIST |
| nytimes-256 | 256 | 290K | ~280 MB | IP | NYTimes 文章 |
| sift1b | 128 | 1B | ~128 GB | L2 | 10亿级 SIFT |

## 快速开始

### 1. 安装依赖

```bash
pip install pymilvus numpy h5py
```

### 2. 下载数据集

```bash
# 列出所有可用数据集
python diskann-test.py --list

# 下载指定数据集
python diskann-test.py --dataset sift --download
python diskann-test.py --dataset gist --download
```

### 3. 运行测试

```bash
# 测试 SIFT-128 (1M 向量，推荐入门)
python diskann-test.py --dataset sift

# 测试 GIST-960 (1M 向量，高维度)
python diskann-test.py --dataset gist

# 测试 SIFT1B 子集 (亿级)
python diskann-test.py --dataset sift1b -n 10M
python diskann-test.py --dataset sift1b -n 100M
```

## 使用示例

```bash
# 基本用法
python diskann-test.py --dataset sift

# 指定 Milvus 地址
python diskann-test.py --dataset gist --uri http://localhost:19530

# 使用自定义 HDF5 文件
python diskann-test.py --hdf5 /path/to/custom.hdf5

# 调整批量大小
python diskann-test.py --dataset sift --batch-size 10000

# SIFT1B 大规模测试
python diskann-test.py --dataset sift1b -n 100M --batch-size 100000
```

## 文件结构

```
benchmark/
├── README.md           # 本文件
├── diskann-test.py     # 统一测试脚本
├── datasets.py         # 数据集定义
└── data/               # 数据存储目录 (gitignored)
    ├── sift-128-euclidean.hdf5
    ├── gist-960-euclidean.hdf5
    └── ...
```

## 测试指标

- **QPS** (Queries Per Second): 每秒查询数
- **Latency**: 平均查询延迟 (ms)
- **Recall@K**: 召回率 (返回结果中真实最近邻的比例)

## 硬件建议

| 数据集规模 | RAM | 磁盘 |
|-----------|-----|------|
| 1M (sift, gist) | 8 GB | 10 GB |
| 10M | 16 GB | 50 GB |
| 100M | 32 GB | 200 GB |
| 1B | 64 GB | 500 GB+ |

## 数据集来源

- HDF5 格式: [ann-benchmarks.com](https://ann-benchmarks.com/)
- SIFT1B: [INRIA BigANN](http://corpus-texmex.irisa.fr/)
