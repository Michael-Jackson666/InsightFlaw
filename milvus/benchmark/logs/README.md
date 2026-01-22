# Benchmark Logs

该目录存储所有 DiskANN 性能测试日志。

## 日志列表

| 日期 | 数据集 | 向量数 | 维度 | QPS | Recall@100 |
|------|--------|--------|------|-----|------------|
| 2026-01-22 | [GIST-960](gist-960_2026-01-22.md) | 1M | 960 | 5.41 | 99.95% |

## 命名规则

日志文件命名: `<dataset>_<date>.md`

例如: `gist-960_2026-01-22.md`

## 测试环境

- **设备**: MacBook Pro (M4 Pro, 24GB RAM)
- **Milvus**: 2.6.8 Standalone
- **索引**: DiskANN
