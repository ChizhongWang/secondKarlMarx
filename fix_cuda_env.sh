#!/bin/bash
# 修复CUDA环境问题

# 检查CUDA版本
nvidia-smi

# 安装CUDA 12.2兼容的PyTorch版本
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu122

# 安装推荐的依赖版本
pip install transformers==4.51.3
pip install datasets==3.2.0
pip install accelerate==1.2.1
pip install peft==0.15.1
pip install trl==0.9.6

# 安装可选依赖
pip install deepspeed==0.16.4
pip install bitsandbytes==0.43.1
pip install flash-attn==2.7.2

# 从源码安装LLaMA Factory
if [ ! -d "LLaMA-Factory" ]; then
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e .
cd ..

# 检查安装
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('PyTorch version:', torch.__version__)"
