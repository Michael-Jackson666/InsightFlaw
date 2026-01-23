# Benchmark 文档

本目录包含 Milvus 向量数据库性能测试的相关文档。

## 文档列表

### [pymilvus-api.md](pymilvus-api.md)
**PyMilvus API 与 DiskANN 参数完整指南**

涵盖内容：
- **PyMilvus 基础 API**: 连接、集合管理、插入、搜索、删除等完整操作
- **DiskANN 索引介绍**: 工作原理、架构图解、适用场景
- **参数详解**: `search_list` 深度解析，包含数学公式和性能关系
- **调优指南**: 性能调优流程、常见问题排查、推荐配置
- **完整示例**: 生产级代码示例和性能测试模板

**适合人群**: 
- 需要使用 Milvus 的开发者
- 想要优化向量检索性能的工程师
- RAG 应用开发者

**核心内容**:
```
search_list 参数详解
├── 定义: 候选向量队列大小
├── 性能公式: Recall ∝ log(search_list), Latency ∝ search_list
├── 实测数据: GIST-960 benchmark 结果
└── 最佳实践: 不同场景的推荐配置
```

---

## 相关资源

- [../README.md](../README.md) - Benchmark 项目主文档
- [../logs/](../logs/) - 性能测试日志
- [../diskann-test.py](../diskann-test.py) - 多数据集测试脚本
- [../datasets.py](../datasets.py) - 数据集定义

## 快速链接

- [Milvus 官方文档](https://milvus.io/docs)
- [PyMilvus API 参考](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md)
- [DiskANN 论文](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
- [ann-benchmarks](https://ann-benchmarks.com/)
