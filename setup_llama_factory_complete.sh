#!/bin/bash
# LLaMA Factory 完整环境配置脚本
# 这个脚本整合了所有修复步骤，解决各种依赖问题

echo "===== 开始配置 LLaMA Factory 环境 ====="

# 1. 检查CUDA版本
echo "检查CUDA版本..."
nvidia-smi
echo ""

# 2. 卸载可能冲突的包
echo "卸载可能冲突的包..."
pip uninstall -y torch torchvision torchaudio
pip uninstall -y transformers
pip uninstall -y peft
pip uninstall -y llama-factory
pip uninstall -y flash-attn
pip uninstall -y gradio
echo ""

# 3. 安装PyTorch (CUDA 11.6兼容版本，LLaMA Factory最低要求)
echo "安装PyTorch (CUDA 11.6兼容版本)..."
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
echo ""

# 4. 安装LLaMA Factory官方指定的最低版本依赖
echo "安装核心依赖..."
pip install transformers==4.45.0
pip install datasets==2.16.0
pip install accelerate==0.34.0
pip install peft==0.14.0
pip install trl==0.8.6
echo ""

# 5. 安装可选依赖
echo "安装可选依赖..."
pip install deepspeed==0.10.0
pip install bitsandbytes==0.39.0
# 不安装flash-attn，避免长时间编译
echo ""

# 6. 安装httpx的socks扩展（修复gradio导入错误）
echo "安装httpx的socks扩展..."
pip install "httpx[socks]"
echo ""

# 7. 安装兼容版本的gradio
echo "安装兼容版本的gradio..."
pip install "gradio>=3.38.0,<4.0.0"
echo ""

# 8. 从源码安装LLaMA Factory
echo "从源码安装LLaMA Factory..."
if [ ! -d "LLaMA-Factory" ]; then
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
# 使用最新版本
pip install -e .
cd ..
echo ""

# 9. 检查安装
echo "===== 检查安装 ====="
echo "检查PyTorch安装:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('PyTorch version:', torch.__version__)"
echo ""

echo "检查Transformers安装:"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
echo ""

echo "检查PEFT安装:"
python -c "import peft; print('PEFT version:', peft.__version__)"
echo ""

echo "检查Gradio安装:"
python -c "import gradio; print('Gradio version:', gradio.__version__)"
echo ""

echo "===== 环境配置完成 ====="
echo "现在可以运行: ./run_llama_factory.sh"
