# 马克思恩格斯知识图谱项目

本项目使用GraphRAG构建马克思恩格斯全集的知识图谱，并提供基于图的检索增强生成(RAG)功能。

## 项目结构

```
marx_kg/
├── data/               # 数据目录
│   ├── raw/            # 原始文本数据
│   └── processed/      # 处理后的数据
├── scripts/            # 脚本目录
│   ├── process_data.py # 数据处理脚本
│   ├── build_kg.py     # 知识图谱构建脚本
│   └── query_kg.py     # 知识图谱查询脚本
├── models/             # 模型和嵌入
│   ├── cache/          # 缓存目录
│   └── vector_store/   # 向量存储
├── kg/                 # 知识图谱输出
│   └── output/         # 输出目录
├── config/             # 配置文件
│   └── graphrag_config.yaml  # GraphRAG配置
└── README.md           # 项目说明
```

## 使用方法

### 1. 准备数据

将马克思恩格斯全集的文本文件放入`data/raw`目录。文件命名格式建议为：`作者_作品名_年份.txt`，例如：`马克思_资本论_1867.txt`。

### 2. 处理数据

```powershell
cd marx_kg
python scripts/process_data.py
```

这将处理原始文本并生成GraphRAG可用的JSON格式数据。

### 3. 构建知识图谱

```powershell
cd marx_kg
python scripts/build_kg.py
```

这将使用GraphRAG构建知识图谱，包括实体提取、关系提取、图谱嵌入等步骤。

### 4. 查询知识图谱

交互式查询：

```powershell
cd marx_kg
python scripts/query_kg.py
```

单次查询：

```powershell
cd marx_kg
python scripts/query_kg.py "马克思关于资本主义的观点是什么？"
```

## 配置说明

配置文件位于`config/graphrag_config.yaml`，您可以根据需要修改以下配置：

- **模型配置**：指定使用的语言模型和嵌入模型
- **输入输出配置**：指定数据和输出路径
- **图谱提取配置**：指定实体类型和提取参数
- **查询配置**：指定查询参数

## 使用本地模型

本项目默认配置为使用本地运行的Qwen2.5-7B-Instruct模型。确保您的本地模型服务器已启动，并在配置文件中正确设置了`api_base`。

## 依赖

本项目依赖于GraphRAG库，确保GraphRAG已正确安装并可访问。

## 注意事项

- 处理大量文本可能需要较长时间
- 图谱构建过程会消耗较多计算资源
- 使用本地模型时，确保有足够的GPU内存
