#!/bin/bash
# 安装LLaMA Factory及其依赖

# 卸载当前的llama-factory和相关依赖
pip uninstall -y llama-factory torch torchvision torchaudio transformers peft accelerate datasets trl

# 安装LLaMA Factory推荐的依赖版本
echo "安装LLaMA Factory推荐的依赖版本..."
pip install torch==2.6.0 torchvision torchaudio
pip install transformers==4.50.0
pip install datasets==3.2.0
pip install accelerate==1.2.1
pip install peft==0.15.1
pip install trl==0.9.6

# 直接从GitHub安装LLaMA Factory
echo "从GitHub安装LLaMA Factory..."
pip install git+https://github.com/hiyouga/LLaMA-Factory.git

# 检查安装是否成功
if command -v llamafactory-cli &> /dev/null; then
    echo "LLaMA Factory安装成功！"
    echo "可以使用 'llamafactory-cli' 命令了"
else
    echo "LLaMA Factory安装可能失败，请检查错误信息"
fi
