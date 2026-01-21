# SIFT1B 数据集测试指南 (macOS + Milvus Standalone)

本教程指导你在 **macOS** 上使用 **Docker Milvus Standalone** 版本进行**亿级向量**的 **DiskANN** 索引性能测试。

## 前置条件

- ✅ Docker Desktop for Mac 已安装并运行
- ✅ Milvus Standalone 已启动 (需配置足够内存)
- ✅ Python 环境已配置 (pymilvus >= 2.5.0)
- ✅ 硬件要求: **64GB RAM + 1TB NVMe SSD**

---

## 硬件配置说明

| 资源 | 最低要求 | 推荐配置 |
|------|---------|----------|
| RAM | 32 GB | 64 GB |
| 磁盘 | 500 GB SSD | 1 TB NVMe SSD |
| CPU | 8 cores | 16+ cores |

### Milvus Docker 配置调整

修改 `docker-compose.yml` 增加内存限制：

```yaml
services:
  standalone:
    deploy:
      resources:
        limits:
          memory: 48G  # 根据实际内存调整
```

---

## 数据集选择

### 方案一：SIFT1B (BigANN 原始格式 - 推荐)

*   **数据量**：10 亿条 128 维向量
*   **格式**：bvecs (二进制格式)
*   **大小**：约 128 GB (压缩后)
*   **来源**：INRIA BigANN Project

### 方案二：Deep1B (HDF5 子集 - 适合测试)

*   **数据量**：10 亿条 96 维向量
*   **格式**：HDF5 (ann-benchmarks)
*   **优点**：格式与 SIFT1M 一致，代码兼容

### 本指南使用方案

我们提供两种测试方式：
1. **快速测试**：使用 SIFT1M (100万) 验证流程
2. **亿级测试**：使用 SIFT1B 完整数据集

---

## 第一步：下载数据集

### 选项 A：下载 SIFT1B (完整 10 亿向量)

```bash
# 进入 SIFT1B 目录
cd /path/to/milvus/SIFT1B

# 创建数据目录
mkdir -p data && cd data

# 下载底库向量 (约 128 GB，需要数小时)
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
gunzip bigann_base.bvecs.gz

# 下载查询向量 (约 1 MB)
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
gunzip bigann_query.bvecs.gz

# 下载 ground truth (约 40 MB)
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
tar -xzf bigann_gnd.tar.gz
```

### 选项 B：使用下载脚本

```bash
# 仅下载查询向量和 ground truth (用于测试脚本)
python download.py --query-only

# 下载完整数据集
python download.py
```

### 选项 C：使用 HDF5 子集 (ann-benchmarks)

如果无法下载完整数据集，可以使用 ann-benchmarks 的子集：

```bash
# 下载 SIFT1M (100万) - 用于测试流程
curl -L -o sift-128-euclidean.hdf5 "https://ann-benchmarks.com/sift-128-euclidean.hdf5"

# 下载 GloVe-100 (120万) - 100维向量
curl -L -o glove-100-angular.hdf5 "https://ann-benchmarks.com/glove-100-angular.hdf5"
```

### 下载后的文件结构

```
SIFT1B/
├── diskann-test.py          # DiskANN 测试脚本
├── SIFT1B.md                # 本文档
└── data/                    # 数据目录 (gitignored)
    ├── bigann_base.bvecs    # 底库向量 (128 GB)
    ├── bigann_query.bvecs   # 查询向量
    └── bigann_gnd/          # Ground truth
        └── idx_1000M.ivecs  # 10亿向量的标准答案
```

---

## 第二步：运行测试脚本

### 安装依赖

```bash
pip install pymilvus numpy h5py
```

### 渐进式测试 (推荐)

从小规模开始，逐步增加数据量：

```bash
# 1. 先测试 1000 万 (约 10-15 分钟)
python diskann-test.py --vectors 10M

# 2. 再测试 1 亿 (约 1-2 小时)
python diskann-test.py --vectors 100M

# 3. 最后测试 10 亿 (约 8-12 小时)
python diskann-test.py --vectors 1B
```

### 脚本功能说明

脚本会自动完成以下步骤：
1. 读取 SIFT1B 数据集 (bvecs 或 HDF5 格式)
2. 将向量分批插入 Milvus
3. 构建 **DiskANN** 索引
4. 执行查询性能测试
5. 计算 **QPS** (每秒查询数)
6. 计算 **Latency** (平均延迟)
7. 计算 **Recall@100** (召回率)
8. 对比不同 search_list 参数的性能

---

## 测试配置预设

| 预设 | 向量数 | 磁盘占用 | 预计耗时 |
|------|-------|----------|---------|
| 10M | 1000 万 | ~1.5 GB | 10-15 分钟 |
| 100M | 1 亿 | ~15 GB | 1-2 小时 |
| 500M | 5 亿 | ~75 GB | 3-5 小时 |
| 1B | 10 亿 | ~150 GB | 8-12 小时 |

---

## 预期性能 (64GB RAM + NVMe SSD)

| 指标 | 100M | 500M | 1B |
|------|------|------|-----|
| 索引构建时间 | ~30 min | ~3 hrs | ~8 hrs |
| QPS (search_list=100) | ~500 | ~300 | ~200 |
| Recall@100 | 0.95+ | 0.93+ | 0.90+ |
| P99 延迟 | <50ms | <100ms | <150ms |

---

## DiskANN 参数说明

### 索引参数

```python
index_params = {
    "index_type": "DISKANN",
    "metric_type": "L2",      # 欧氏距离
    "params": {}              # DiskANN 自动调优
}
```

### 搜索参数

```python
search_params = {
    "metric_type": "L2",
    "params": {
        "search_list": 100    # 必须 >= limit (TopK)
    }
}
```

**search_list 参数说明：**
- `search_list` 越大，召回率越高，但速度越慢
- 必须 >= `limit` (TopK)
- 推荐值：`limit` 的 1-3 倍

| search_list | 召回率 | 速度 |
|-------------|--------|------|
| 100 | ~95% | 最快 |
| 200 | ~98% | 较快 |
| 500 | ~99% | 适中 |

---

## 注意事项

1. **渐进式测试**：建议先用 10M、100M 验证，再跑完整 1B
2. **磁盘 I/O**：强烈推荐 NVMe SSD，HDD 会慢 10 倍以上
3. **内存**：DiskANN 是磁盘优化算法，但更多内存有助于缓存
4. **索引持久化**：Milvus 会将索引存储到磁盘，重启后数据保留

---

## 参考资料

- [BigANN Benchmark](http://corpus-texmex.irisa.fr/)
- [DiskANN 论文](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
- [Milvus DiskANN 文档](https://milvus.io/docs/disk_index.md)
- [ann-benchmarks](https://ann-benchmarks.com/)
