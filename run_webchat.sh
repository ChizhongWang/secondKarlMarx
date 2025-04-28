#!/bin/bash
# 启动LLaMA Factory Web聊天界面

# 设置端口，默认为7860
PORT=${1:-7860}

echo "启动Web聊天界面，端口: $PORT..."

# 启动Web聊天界面
llamafactory-cli webchat qwen_inference.yaml --port $PORT --host 0.0.0.0

echo "Web聊天界面已启动，可以通过 http://[服务器IP]:$PORT 访问"
