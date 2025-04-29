#!/bin/bash
# 在云服务器上启动MCP服务器

echo "启动MCP服务器..."

# 安装依赖（如果需要）
pip install -q mcp-server peft transformers accelerate

# 启动服务器
python server.py
