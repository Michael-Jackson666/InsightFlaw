# SIFT1B 亿级向量检索测试

使用 SIFT1B 数据集进行 **10亿** 128维向量的 DiskANN 性能基准测试。

## 数据集概览

| 属性 | 值 |
|------|-----|
| 底库向量 | 1,000,000,000 (10亿) |
| 查询向量 | 10,000 |
| 维度 | 128 |
| 距离度量 | L2 (欧氏距离) |
| 原始数据大小 | ~128 GB |
| 索引大小 (DiskANN) | ~150-200 GB |

## 硬件要求

| 资源 | 最低 | 推荐 |
|------|------|------|
| RAM | 32 GB | 64 GB |
| 磁盘 | 500 GB SSD | 1 TB NVMe SSD |
| CPU | 8 核 | 16+ 核 |

## 文件结构

```
SIFT1B/
├── README.md           # 本文件
├── SIFT1B.md           # 详细测试指南
├── diskann-test.py     # DiskANN 性能测试脚本
└── data/               # 数据存储目录 (gitignored)
    ├── bigann_base.bvecs    # 底库向量 (128 GB)
    ├── bigann_query.bvecs   # 查询向量
    └── bigann_gnd/          # Ground truth
```

## 下载数据集

```bash
# 进入数据目录
cd data

# 下载底库向量 (约 128 GB，需数小时)
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
gunzip bigann_base.bvecs.gz

# 下载查询向量 (约 1 MB)
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
gunzip bigann_query.bvecs.gz

# 下载 ground truth (约 40 MB)
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
tar -xzf bigann_gnd.tar.gz
```

## 快速开始

1. 确保 Milvus Standalone 运行中且内存充足

2. 渐进式测试 (推荐):

```bash
# 安装依赖
pip install pymilvus numpy h5py

# 先测试 1000 万向量
python diskann-test.py -n 10M

# 再测试 1 亿向量
python diskann-test.py -n 100M

# 最后测试 10 亿向量
python diskann-test.py -n 1B
```

## 测试预设

| 预设 | 向量数 | 磁盘占用 | 预计耗时 |
|------|--------|----------|----------|
| 10M | 1000万 | ~1.5 GB | 10-15 min |
| 100M | 1亿 | ~15 GB | 1-2 hours |
| 500M | 5亿 | ~75 GB | 3-5 hours |
| 1B | 10亿 | ~150 GB | 8-12 hours |

## 预期性能 (64GB RAM + NVMe SSD)

| 指标 | 100M | 500M | 1B |
|------|------|------|-----|
| 索引构建 | ~30 min | ~3 hrs | ~8 hrs |
| QPS | ~500 | ~300 | ~200 |
| Recall@100 | 0.95+ | 0.93+ | 0.90+ |

## 详细说明

查看 [SIFT1B.md](SIFT1B.md) 获取完整的测试指南和参数说明。

## 参考资料

- [BigANN Benchmark](http://corpus-texmex.irisa.fr/)
- [DiskANN Paper](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
- [Milvus DiskANN Guide](https://milvus.io/docs/disk_index.md)
