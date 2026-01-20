# InsightFlaw

An intelligent RAG (Retrieval-Augmented Generation) application powered by LLM and Vector Database. Turning fragmented data into actionable insights.

## 项目概述

InsightFlaw 是一个基于 RAG 架构的智能应用程序，结合了以下技术栈：
- **LLM**: 大语言模型用于理解和生成自然语言
- **VectorDB**: Milvus 向量数据库用于高效的向量检索
- **LangChain**: 用于构建和编排 LLM 应用程序的框架

## 技术架构

- **向量数据库**: Milvus (支持 Lite 和 Standalone 模式)
- **AI 框架**: LangChain
- **向量检索算法**: DiskANN (高性能近似最近邻搜索)
- **部署环境**: macOS

## 功能特性

- 文档向量化存储与检索
- 基于语义的智能问答
- 支持多种数据格式导入
- 高效的向量相似度搜索

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Milvus Lite (轻量级部署)
python milvus/lite-tutorial.ipynb

# 或启动 Milvus Standalone (完整功能)
python milvus/standalone.py
```

### 使用示例

详见 `milvus/` 目录下的教程文件：
- `lite-tutorial.ipynb`: Milvus Lite 快速入门
- `diskann-tutorial.ipynb`: DiskANN 索引使用教程
- `Milvus Lite部署与应用-EasyVectordb.md`: 详细部署文档

## 目录结构

```
InsightFlaw/
├── milvus/              # Milvus 相关配置和示例
│   ├── diskann.py       # DiskANN 索引实现
│   ├── standalone.py    # Milvus Standalone 部署
│   ├── test-lite.py     # Milvus Lite 测试
│   └── SIFT1M/          # SIFT1M 数据集测试
└── README.md
```

## 开发计划

- [ ] 集成 LangChain 框架
- [ ] 实现文档解析和向量化pipeline
- [ ] 添加 Web UI 界面
- [ ] 支持多种 LLM 模型
- [ ] 优化检索性能

## License

详见 [LICENSE](LICENSE) 文件
