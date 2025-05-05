#!/bin/bash
# 安装LLaMA Factory及其依赖

# 卸载当前的llama-factory（如果已安装）
pip uninstall -y llama-factory

# 确保安装指定版本的依赖
pip install torch==2.6.0
pip install transformers==4.51.3

# 从源码安装最新的llama-factory
if [ ! -d "LLaMA-Factory" ]; then
    echo "克隆LLaMA-Factory仓库..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi

cd LLaMA-Factory
echo "安装LLaMA-Factory..."
pip install -e .

# 返回原目录
cd ..

echo "LLaMA-Factory安装完成！"
