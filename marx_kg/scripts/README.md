# 马克思恩格斯全集文本向量化与搜索工具

这个工具包含两个主要脚本，用于将马克思恩格斯全集的文本块向量化并进行语义搜索。

## 功能特点

- **文本向量化**：使用DMXAPI的text-embedding-3-large模型将文本块转换为向量表示
- **向量数据库存储**：使用LanceDB存储向量和元数据，支持高效的相似度搜索
- **语义搜索**：基于向量相似度进行语义搜索，找出与查询语义相关的文本块
- **上下文获取**：获取相关文本块的上下文，提供更完整的信息
- **断点续传**：支持向量化过程中断后继续处理，避免重复工作
- **并行处理**：使用多线程并行调用API，提高处理效率

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 文本向量化

```bash
python marx_vectorize.py \
  --input_dir "/path/to/marx_collection_works/collection_works_chunks" \
  --db_path "/path/to/marx_vector_db" \
  --api_key "your-dmx-api-key" \
  --batch_size 20 \
  --max_workers 5
```

参数说明：
- `--input_dir`：包含JSON文本块的输入目录
- `--db_path`：向量数据库保存路径
- `--output_dir`：(可选) 保存带向量的JSON文件的输出目录
- `--api_key`：DMXAPI密钥
- `--batch_size`：批处理大小，默认20
- `--max_workers`：最大并行工作线程数，默认5
- `--checkpoint_file`：检查点文件路径，默认"checkpoint.json"

### 2. 文本搜索

```bash
python marx_search.py \
  --query "马克思对自由的看法" \
  --db_path "/path/to/marx_vector_db" \
  --chunks_dir "/path/to/marx_collection_works/collection_works_chunks" \
  --api_key "your-dmx-api-key" \
  --top_k 5 \
  --context_size 1
```

参数说明：
- `--query`：搜索查询文本
- `--db_path`：向量数据库路径
- `--chunks_dir`：原始文本块目录
- `--api_key`：DMXAPI密钥
- `--top_k`：返回结果数量，默认5
- `--context_size`：上下文大小（前后各多少个块），默认1
- `--output`：(可选) 输出文件路径

## 工作流程

1. **向量化流程**：
   - 读取文本块JSON文件
   - 批量调用API生成嵌入向量
   - 将向量和元数据存储到LanceDB
   - 更新检查点，支持断点续传

2. **搜索流程**：
   - 为查询文本生成嵌入向量
   - 在向量数据库中搜索相似向量
   - 获取相关文本块的上下文
   - 格式化并返回搜索结果

## 注意事项

- 确保DMXAPI密钥有效且有足够的配额
- 对于大规模处理，建议在高性能服务器上运行
- 向量化过程可能需要几小时，请确保网络稳定
- 检查点文件会记录已处理的文件，删除它将重新处理所有文件
