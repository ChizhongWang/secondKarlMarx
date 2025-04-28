#!/bin/bash
# 完整的环境修复脚本 - 使用LLaMA Factory官方指定的版本

# 卸载现有包以避免冲突
pip uninstall -y torch torchvision torchaudio
pip uninstall -y transformers
pip uninstall -y peft
pip uninstall -y llama-factory

# 安装CUDA 11.6兼容的PyTorch (最低要求)
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

# 安装官方指定的最低版本依赖
pip install transformers==4.45.0
pip install datasets==2.16.0
pip install accelerate==0.34.0
pip install peft==0.14.0
pip install trl==0.8.6

# 安装可选依赖（最低版本）
pip install deepspeed==0.10.0
pip install bitsandbytes==0.39.0
pip install flash-attn==2.5.6

# 从源码安装最新版本的LLaMA Factory
if [ ! -d "LLaMA-Factory" ]; then
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
# 不指定特定版本，使用最新版本
pip install -e .
cd ..

# 检查安装
echo "检查PyTorch安装:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('PyTorch version:', torch.__version__)"

echo "检查Transformers安装:"
python -c "import transformers; print('Transformers version:', transformers.__version__)"

echo "检查PEFT安装:"
python -c "import peft; print('PEFT version:', peft.__version__)"
