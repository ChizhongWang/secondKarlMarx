#!/bin/bash
# 安装cuDNN 9.x以支持PyTorch 2.6.0

# 方法1: 使用conda安装cuDNN
conda install -y cudnn=9.0

# 如果上面的方法不起作用，尝试方法2
# 方法2: 使用apt安装cuDNN (需要sudo权限)
# sudo apt-get update
# sudo apt-get install -y libcudnn9

# 方法3: 使用特定版本的PyTorch wheel包，它包含所需的cuDNN
# pip install torch==2.6.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126

# 检查是否成功安装
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('PyTorch version:', torch.__version__)"
