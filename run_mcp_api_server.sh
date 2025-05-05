#!/bin/bash
# 启动MCP API服务器

echo "启动MCP API服务器，端口: 8000..."

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:/home/featurize/secondKarlMarx

# 安装MCP依赖
pip install -q mcp-server peft transformers accelerate

# 启动MCP服务器
cd /home/featurize/secondKarlMarx/mcp
python server.py

echo "MCP API服务器已启动，可以通过 http://[服务器IP]:8000 访问"
