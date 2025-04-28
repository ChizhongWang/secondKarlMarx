#!/bin/bash
# 启动LLaMA Factory API服务器

# 设置端口，默认为8000
PORT=${1:-8000}

echo "启动API服务器，端口: $PORT..."

# 启动API服务器
llamafactory-cli api qwen_inference.yaml --port $PORT --host 0.0.0.0

echo "API服务器已启动，可以通过 http://[服务器IP]:$PORT 访问"
