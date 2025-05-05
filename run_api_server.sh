#!/bin/bash
# 启动LLaMA Factory API服务器

echo "启动API服务器..."

# 启动API服务器（使用环境变量指定端口）
API_PORT=8000 llamafactory-cli api qwen_inference.yaml

echo "API服务器已启动，可以通过 http://[服务器IP]:8000 访问"
