#!/bin/bash
# 替代方案：使用与CUDA 12.6兼容的PyTorch版本，但降级到PyTorch 2.0.1

# 卸载当前的PyTorch
pip uninstall -y torch torchvision

# 安装CUDA 12.1兼容的PyTorch 2.0.1
# 注意：目前没有官方的CUDA 12.6预编译包，使用12.1版本通常也能与12.6兼容
pip install torch==2.0.1+cu121 torchvision==0.15.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 确保transformers版本保持不变
pip install transformers==4.51.3

# 安装其他LLaMA Factory依赖
pip install datasets==3.2.0 accelerate==1.2.1 peft==0.15.1 trl==0.9.6

# 从源码安装LLaMA Factory
if [ ! -d "LLaMA-Factory" ]; then
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e .
cd ..

# 检查安装
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('PyTorch version:', torch.__version__)"
