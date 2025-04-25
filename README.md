# secondKarlMarx

基于大型语言模型的马克思主义理论助手，支持分布式训练和MCP远程访问。

## 项目概述

secondKarlMarx是一个专门针对马克思主义理论的大语言模型微调项目，通过SFT（监督微调）技术，使模型能够深入理解和解释马克思主义理论，同时保持自然对话能力和RAG工具调用能力。

主要特点：
- 基于主流大语言模型（如Llama 3）进行微调
- 支持多GPU分布式训练
- 使用LoRA高效参数微调方法
- 通过MCP（Model Context Protocol）实现远程访问
- 支持RAG（检索增强生成）工具调用

## 项目结构

```
secondKarlMarx/
├── configs/                # 配置文件
│   ├── training_config.py  # 训练配置
│   └── ds_config.json      # DeepSpeed配置
├── training/               # 训练相关代码
│   ├── data_utils.py       # 数据处理工具
│   └── trainer.py          # 训练器实现
├── model/                  # 模型相关代码
│   └── model_loader.py     # 模型加载器
├── mcp/                    # MCP服务相关
│   ├── server.py           # MCP服务器
│   ├── client.py           # MCP客户端
│   └── mcp_config.json     # MCP配置
├── utils/                  # 工具函数
├── train.py                # 主训练脚本
├── run_distributed_training.sh  # 分布式训练启动脚本
├── start_mcp_server.py     # 启动MCP服务器脚本
└── requirements.txt        # 依赖包列表
```

## 安装指南

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/yourusername/secondKarlMarx.git
cd secondKarlMarx

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. Hugging Face认证（如需访问受限模型）

```bash
# 设置Hugging Face令牌
export HUGGING_FACE_HUB_TOKEN=your_token_here
# 或在Windows上
set HUGGING_FACE_HUB_TOKEN=your_token_here
```

## 使用指南

### 1. 训练模型（在云服务器上）

#### 单GPU训练

```bash
python train.py
```

#### 多GPU分布式训练

```bash
# 修改run_distributed_training.sh中的GPU设置
chmod +x run_distributed_training.sh
./run_distributed_training.sh
```

### 2. 启动MCP服务（在云服务器上）

```bash
# 启动MCP服务器
python start_mcp_server.py --model_path ./results/final_model --host 0.0.0.0 --port 8000
```

### 3. 在本地笔记本上使用模型

#### 配置MCP客户端

1. 编辑`mcp/mcp_config.json`文件，设置服务器IP和路径：
```json
{
  "mcpServers": {
    "secondKarlMarx": {
      "command": "python",
      "args": ["/path/to/your/server.py"],
      "host": "your-server-ip",
      "port": 8000
    }
  }
}
```

2. 启动客户端界面：
```bash
python mcp/client.py
```

## 自定义配置

### 修改训练配置

编辑`configs/training_config.py`文件，可以调整以下参数：
- 基础模型：修改`BASE_MODEL_CONFIG`中的`model_name_or_path`
- 数据集：修改`DATASET_CONFIG`中的`dataset_name`
- 训练参数：修改`TRAINING_CONFIG`中的各项参数
- LoRA配置：修改`LORA_CONFIG`中的参数

### 修改DeepSpeed配置

编辑`configs/ds_config.json`文件，可以调整分布式训练参数。

## 注意事项

1. 确保云服务器有足够的GPU内存
2. 对于大型模型，建议使用8位或4位量化
3. 训练前检查数据集格式是否符合要求
4. MCP服务器需要开放相应端口供外部访问

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

[MIT License](LICENSE)
