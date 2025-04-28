# 使用MCP远程访问微调后的Qwen2.5-7B-Instruct模型

本文档介绍如何使用MCP (Model Control Protocol) 从本地笔记本访问运行在云服务器上的微调后的Qwen2.5-7B-Instruct模型。

## 什么是MCP?

MCP (Model Control Protocol) 是一种用于控制和访问远程模型的协议，它允许您在本地与云端模型进行交互，而无需将模型下载到本地。

## 文件结构

- `server.py` - 在云服务器上运行的MCP服务器脚本
- `client.py` - 在本地运行的MCP客户端脚本
- `model_config.json` - 模型配置文件
- `mcp_config.json` - MCP配置文件

## 使用步骤

### 1. 在云服务器上设置

1. 确保云服务器上已经完成了模型微调，并且模型权重保存在`./outputs_llama_factory_full`目录中
2. 修改`model_config.json`文件，确保路径正确
3. 启动MCP服务器：

```bash
# 安装MCP依赖
pip install mcp-server

# 启动服务器
python server.py
```

### 2. 在本地设置

1. 安装MCP客户端：

```bash
pip install mcp-client
```

2. 运行客户端，连接到云服务器：

```bash
python client.py --host <云服务器IP> --port <端口号>
```

这将启动一个Gradio Web界面，您可以通过它与云端的模型进行交互。

## 配置文件说明

### model_config.json

```json
{
  "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "adapter_name_or_path": "./outputs_llama_factory_full",
  "template": "qwen",
  "finetuning_type": "lora",
  "device": "cuda",
  "max_new_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9,
  "repetition_penalty": 1.1
}
```

- `model_name_or_path`: 基础模型的名称或路径
- `adapter_name_or_path`: LoRA适配器的路径
- `template`: 使用的模板
- `finetuning_type`: 微调类型
- `device`: 使用的设备（cuda或cpu）
- 其他参数用于控制生成过程

### mcp_config.json

```json
{
  "mcpServers": {
    "secondKarlMarx": {
      "command": "python",
      "args": ["path/to/server.py"],
      "host": "your-server-ip",
      "port": 8000
    }
  }
}
```

- `command`: 启动服务器的命令
- `args`: 命令参数
- `host`: 服务器主机地址
- `port`: 服务器端口

## 故障排除

1. 如果无法连接到服务器，请检查：
   - 云服务器的防火墙是否允许指定端口的访问
   - `mcp_config.json`中的IP和端口是否正确
   - 服务器是否正在运行

2. 如果模型加载失败，请检查：
   - GPU内存是否足够
   - 模型路径是否正确
   - 是否安装了所有必要的依赖

3. 如果生成结果不符合预期，可以尝试调整`model_config.json`中的参数，如温度、top_p等。
