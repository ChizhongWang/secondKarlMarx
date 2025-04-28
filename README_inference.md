# 使用微调后的Qwen2.5-7B-Instruct模型

本文档介绍如何从您的笔记本访问运行在云服务器上的微调后的模型。

## 准备工作

1. 确保云服务器上的微调已经完成，并且模型权重保存在`./outputs_llama_factory_full`目录中
2. 确保云服务器有公网IP或者您的笔记本可以通过某种方式（如VPN）连接到云服务器
3. 确保云服务器的防火墙允许访问您将要使用的端口（默认为7860或8000）

## 使用方法

我们提供了两种访问方式：

### 1. 使用Web聊天界面（推荐）

这种方式提供了一个友好的Web界面，您可以直接在浏览器中与模型对话。

```bash
# 在云服务器上运行
bash run_webchat.sh
```

默认情况下，Web界面将在端口7860上启动。您可以通过以下URL访问：

```
http://[服务器IP]:7860
```

如果需要指定其他端口，可以在命令中添加端口号：

```bash
bash run_webchat.sh 8080  # 使用8080端口
```

### 2. 使用API服务

如果您需要通过程序调用模型，可以使用API服务：

```bash
# 在云服务器上运行
bash run_api_server.sh
```

API服务默认在端口8000上启动，您可以通过以下方式调用：

```
POST http://[服务器IP]:8000/v1/chat/completions
```

API格式兼容OpenAI的Chat Completions API，您可以使用以下Python代码示例进行调用：

```python
import requests
import json

url = "http://[服务器IP]:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "Qwen2.5-7B-Instruct-LoRA",
    "messages": [
        {"role": "user", "content": "你好，请介绍一下自己"}
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

## 注意事项

1. 确保云服务器有足够的GPU内存来加载模型
2. 如果使用`huggingface`作为推理后端，第一次加载模型可能需要一些时间
3. 如果需要更快的推理速度，可以在`qwen_inference.yaml`中将`infer_backend`改为`vllm`
4. 如果遇到CUDA内存不足的问题，可以尝试使用8位量化：在`qwen_inference.yaml`中添加`load_in_8bit: true`

## 故障排除

1. 如果无法连接到服务，请检查云服务器的防火墙设置
2. 如果加载模型时出现内存错误，请尝试使用量化或减小批处理大小
3. 如果API调用返回错误，请检查请求格式是否正确
