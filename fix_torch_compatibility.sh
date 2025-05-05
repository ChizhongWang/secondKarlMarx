#!/bin/bash
# 修复PyTorch和torchvision的兼容性问题

echo "开始修复PyTorch和torchvision的兼容性问题..."

# 卸载当前的PyTorch和torchvision
pip uninstall -y torch torchvision torchaudio

# 安装特定版本的PyTorch和torchvision
echo "安装PyTorch 2.0.0和匹配的torchvision..."
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 安装与PyTorch 2.0.0兼容的transformers版本
echo "安装与PyTorch 2.0.0兼容的transformers..."
pip install transformers==4.31.0

# 安装其他必要的依赖
echo "安装其他必要的依赖..."
pip install peft==0.4.0 accelerate

echo "兼容性修复完成！"
echo "现在尝试运行 bash run_api_server.sh 或 bash run_mcp_api_server.sh"
