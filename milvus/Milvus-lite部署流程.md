# Milvus Lite 快速上手教程

本教程将带你快速上手 Milvus Lite，学习如何创建向量数据库、插入数据、进行向量搜索等基本操作。

## 什么是 Milvus Lite？

**Milvus Lite** 是 Milvus 向量数据库的轻量级版本，可以在本地运行，无需启动服务器。它非常适合：
- 🚀 快速原型开发
- 🧪 本地测试和开发
- 📦 小规模应用场景
 
## 环境准备

确保已安装 pymilvus：
```bash
pip install pymilvus
```

## 1. 导入库并检查版本

首先导入 `pymilvus` 库并检查版本，确保环境配置正确。

```python
import pymilvus
from pymilvus import MilvusClient

# 输出当前 pymilvus 版本
print(f"当前 pymilvus 版本: {pymilvus.__version__}")
```

## 2. 配置与初始化

创建 MilvusClient 实例，连接到本地 Milvus Lite 数据库。

**说明**：
- `uri` 参数指定为本地文件路径（如 `./milvus_demo.db`）
- Milvus Lite 会自动在该路径创建数据库文件
- 无需启动任何服务器进程，开箱即用

```python
# 配置与初始化
# 这里的 uri 指定为本地文件路径，Milvus Lite 会自动在该目录下生成数据库文件
client = MilvusClient(uri="./milvus_demo.db")

print("Milvus Lite 初始化成功，数据库文件位于当前目录下 milvus_demo.db")
```

## 3. 创建一个 Collection（集合）

在 Milvus 中，Collection 类似于关系数据库中的表。创建集合时需要指定：
- `collection_name`: 集合名称
- `dimension`: 向量的维度（例如 BERT 模型输出通常是 768 维）

**注意**：如果集合已存在，需要先删除再创建。

```python
# 创建一个 Collection (集合)
if client.has_collection("demo_collection"):
    client.drop_collection("demo_collection")  # 如果集合已存在，先删除

client.create_collection(
    collection_name="demo_collection",  # 集合名称
    dimension=768  # 向量维度，例如 BERT 模型输出
)

# 列出所有集合
print("当前所有集合:", client.list_collections())

# 检查集合是否创建成功
print("demo_collection 是否存在:", client.has_collection("demo_collection"))
```

### 查看集合信息

```python
res = client.describe_collection('demo_collection')
print(res)
# """
# {'collection_name': 'demo_collections', 'auto_id': False, 'num_shards': 0, 'description': '', 'fields': [{'field_id': 100, 'name': 'id', 'description': '', 'type': <DataType.INT64: 5>, 'params': {}, 'is_primary': True}, {'field_id': 101, 'name': 'vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 2048}}], 'functions': [], 'aliases': [], 'collection_id': 0, 'consistency_level': 0, 'properties': {}, 'num_partitions': 0, 'enable_dynamic_field': True}
# """
```

### 字段含义说明

- **`collection_name`**：集合名称，用于唯一标识存储数据的容器，类似于关系型数据库中的表名。命名需简洁且具业务意义，如`demo_collection`
- **`auto_id`**：主键自动生成开关。此处为`False`，表示需手动指定主键值（如插入数据时提供`id`字段）；若为`True`，则Milvus自动生成全局唯一的64位整数主键
- **`num_shards`**：分片数量。此处为`0`，表示使用系统默认分片策略（通常自动按集群规模分配）
- **`description`**：集合描述，此处为空字符串，可补充业务用途说明（如"存储用户行为向量"）
- 字段定义（`fields`)
  - 主键字段 `id`：
    - `type: DataType.INT64`：64位整数类型，唯一标识每条数据。
    - `is_primary`: True：指定为主键，用于高效检索和去重
    - `params: {}`：无额外参数，因主键类型为标量，无需配置向量维度等属性。
  - 向量字段 `vector`：
    - `type: DataType.FLOAT_VECTOR`：浮点向量类型，用于存储高维向量数据（如文本/图像嵌入）。
    - `params: {'dim': 2048}`：向量维度为2048，需与插入数据的实际维度一致

## 4. 插入数据（模拟）

首先创建一个小的向量集合包含详细的字段说明如下所示：

```python
client.create_collection(
    collection_name = "demo_test", # 集合名称
    dimension=5, # 向量维度
    primary_field_name="id", # 主键字段名称
    id_type="int", # 主键字段类型
    vector_field_name="vector", # 向量字段名称
    metric_type="L2", # 向量相似度度量方式
    auto_id=True # 主键ID自动递增
)
```

### 插入单条数据

现在向集合中插入一些模拟数据。每条数据包含：
- `id`: 唯一标识符
- `vector`: 5维的向量（这里用简单的模拟数据）
- `name`: 数据名称
- `text`: 文本内容
- `subject`: 主题分类

**说明**：在实际应用中，向量应该由嵌入模型（如 BERT、Sentence-Transformers）生成。

```python
# 插入一条数据 
res1 = client.insert(
    collection_name="test", # 前面创建的集合名称
    data = {
        "id":0, # 主键ID
        "vector":[0.1, 0.2, 0.3, 0.4, 0.5], # 向量数据
        "name":"test", # 其他字段数据
        "text":"text_0",
        "subject":"history"
    }
)
# 查看插入的数据
result = client.query(collection_name="test", filter="", output_fields=["*"], limit=100)
print("数据条数:", len(result))
print("查询结果:", result)
```

### 批量插入数据

Mivlus Lite 支持批量插入数据，下面示例展示如何插入 10 条模拟数据。

```python
import numpy as np

# 生成随机向量
data = [
    {
        "id": i,
        "vector": np.random.rand(5).tolist(),
        "name": f"name_{i}",
        "text": f"text_{i}",
        "subject": "history"
    }
    for i in range(1, 10)
]

# 批量插入数据到集合
res2 = client.insert(
    collection_name="test",
    data=data
)

client.flush(collection_name="test")  # 确保数据已写入

result = client.query(collection_name="test", filter="", output_fields=["*"], limit=100)
print("数据条数:", len(result))
print("查询结果:", result)
```

## 5. 数据查询（向量搜索）

使用向量搜索功能查找最相似的数据。

**参数说明**：
- `data`: 查询向量（可以是多个）
- `limit`: 返回最相似的前 N 条结果
- `output_fields`: 需要返回的额外字段

```python
# 向量搜索
search_res = client.search(
    collection_name="test",
    data=[[0.1] * 5],  # 查询向量
    limit=3,
    search_params={
        "metric_type": "L2",
        "params": {"nprobe": 10} # 搜索参数, nprobe 越大，搜索越精确，但速度越慢
    },
    output_fields=["name", "text", "subject"]
)
for row in search_res[0]:
    print(row)
```

### 搜索参数说明

- `collection_name`: 指定搜索的集合名称
- `data`: 查询向量, 二维列表格式。可以查询多个向量，但是每个向量的维度必须和集合中定义的向量维度一致。
- `limit`: 返回最相似的前 N 条结果
- `output_fields`: 需要返回的额外字段
- `search_params`: 搜索参数，如使用的索引类型和度量方式
  - `metric_type`: 度量方式，如 'L2'（欧氏距离）、'IP'（内积）
  - `index_type`: 使用的索引类型，如 'FLAT'（暴力搜索）、'IVF_FLAT'（倒排文件索引），其中`nprobe`指定搜索簇数量。

## 总结

通过本教程，你已经学会了 Milvus Lite 的基本操作：
1. ✅ 安装和导入 pymilvus 库
2. ✅ 初始化 Milvus Lite 客户端
3. ✅ 创建和管理 Collection（集合）
4. ✅ 插入向量数据（单条和批量）
5. ✅ 执行向量相似度搜索

## 相关资源

- [Milvus 官方文档](https://milvus.io/docs)
- [PyMilvus API 文档](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)
- [Milvus GitHub](https://github.com/milvus-io/milvus)
