# secondKarlMarx 最小可行性实验 (MVP)

本文档提供了在云服务器上运行最小可行性实验的步骤，以验证训练框架和MCP访问功能。

## 实验目的

1. 验证训练流程能否正常工作
2. 测试DeepSpeed分布式训练功能
3. 确认模型可以通过MCP服务提供
4. 验证客户端能否连接并与模型交互

## 实验步骤

### 1. 克隆仓库到云服务器

```bash
git clone https://github.com/ChizhongWang/secondKarlMarx.git
cd secondKarlMarx
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 设置Hugging Face认证（如需访问私有模型）

```bash
# 设置环境变量
export HUGGING_FACE_HUB_TOKEN=your_token_here

# 或者登录（推荐）
huggingface-cli login
```

### 4. 运行分布式训练

```bash
# 确保run_distributed_training.sh有执行权限
chmod +x run_distributed_training.sh

# 启动训练
./run_distributed_training.sh
```

训练将使用简化配置：
- 仅使用2张GPU
- 仅训练100个样本
- 最大100步训练
- 序列长度限制为1024 tokens

### 5. 启动MCP服务

训练完成后，启动MCP服务以提供模型访问：

```bash
python start_mcp_server.py --model_path ./results/final_model --host 0.0.0.0 --port 8000
```

### 6. 在本地测试MCP连接

在本地笔记本上：

1. 更新`mcp/mcp_config.json`中的服务器信息：
```json
{
  "mcpServers": {
    "secondKarlMarx": {
      "command": "python",
      "args": ["/path/to/your/server.py"],
      "host": "your-server-ip",
      "port": 8000
    }
  }
}
```

2. 运行测试脚本：
```bash
python test_mcp.py
```

如果测试成功，您将看到模型的回复。

## 恢复完整训练

MVP测试成功后，您可以恢复完整训练配置：

1. 编辑`configs/training_config.py`：
   - 将`max_samples`设置为`None`
   - 将`num_train_epochs`增加到`3`
   - 将`max_seq_length`增加到`4096`

2. 编辑`run_distributed_training.sh`：
   - 将GPU数量增加到可用的最大数量
   - 恢复高级网络优化设置

3. 编辑`configs/ds_config.json`：
   - 考虑升级到ZeRO-3以处理更大模型

## 故障排除

如果遇到问题：

1. **OOM错误**：减小批处理大小或序列长度
2. **NCCL错误**：检查网络接口配置
3. **MCP连接失败**：确认防火墙设置和端口开放
