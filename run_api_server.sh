#!/bin/bash
# 启动LLaMA Factory API服务器

echo "启动API服务器..."

# 设置环境变量
export API_HOST=0.0.0.0
export API_PORT=8000

# 启动API服务器
llamafactory-cli api qwen_inference.yaml

echo "API服务器已启动，可以通过 http://[服务器IP]:8000 访问"
