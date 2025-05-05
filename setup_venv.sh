#!/bin/bash
# 创建虚拟环境并安装依赖

echo "创建虚拟环境..."
python -m venv karlmarx_venv

# 激活虚拟环境
echo "激活虚拟环境..."
source karlmarx_venv/bin/activate

# 安装特定版本的依赖
echo "安装依赖..."
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.31.0 peft==0.4.0

# 安装其他依赖
pip install fastapi uvicorn pydantic networkx pandas

# 安装MCP
pip install mcp-server

echo "虚拟环境设置完成！"
echo "使用 'source karlmarx_venv/bin/activate' 激活环境"
