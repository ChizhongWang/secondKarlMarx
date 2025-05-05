# 马克思恩格斯知识图谱云服务器测试指南

本指南提供在云服务器上测试马克思恩格斯知识图谱的完整步骤。

## 1. 准备测试数据

首先，在云服务器上创建一个测试数据目录并添加一些示例文本：

```bash
# 在云服务器上
mkdir -p /path/to/secondKarlMarx/marx_kg/data/raw
cd /path/to/secondKarlMarx/marx_kg/data/raw

# 创建测试文本文件
cat > 马克思_资本论摘录_1867.txt << 'EOF'
第一章 商品和货币
第一节 商品
资本主义生产方式占统治地位的社会的财富，表现为"庞大的商品堆积"，单个的商品表现为这种财富的元素形式。因此，我们的研究就从分析商品开始。
商品首先是一个外界的对象，一个物，它用自己的属性满足人的某种需要。这些需要的性质，例如是由胃还是由想象产生的，并不会使问题发生任何变化。在这里，也不涉及这个物怎样满足人的需要，是作为生活资料，即作为消费品，还是间接地作为生产资料。
EOF

cat > 马克思_共产党宣言摘录_1848.txt << 'EOF'
一个幽灵，共产主义的幽灵，在欧洲游荡。为了对这个幽灵进行神圣的围剿，旧欧洲的一切势力，教皇和沙皇、梅特涅和基佐、法国的激进派和德国的警察，都联合起来了。
有哪一个反对党不被它的当政的敌人骂为共产党呢？有哪一个反对党不拿共产主义这个罪名去回敬更进步的反对党人和自己的反动敌人呢？
从这一事实中可以得出两个结论：
共产主义已经被欧洲的一切势力公认为一种势力；
现在是共产党人向全世界公开说明自己的观点、自己的目的、自己的意图并且拿党自己的宣言来反驳关于共产主义幽灵的神话的时候了。
EOF

cat > 恩格斯_自然辩证法摘录_1883.txt << 'EOF'
辩证法所谓的客观性质，自然界中到处都有。偶然性不断地被必然性所排斥，而必然性又不断地被偶然性所打破。
自然科学已经发展到这样的地步，它再也不能逃避辩证的综合了。但是，如果自然科学家们学会在辩证哲学的范围内进行思考，它就会使自己的工作容易些。
EOF
```

## 2. 安装依赖

确保云服务器上安装了所有必要的依赖：

```bash
# 进入项目目录
cd /path/to/secondKarlMarx

# 安装GraphRAG依赖
cd graphrag
pip install -e .
pip install -r requirements.txt

# 返回项目根目录
cd ..
```

## 3. 启动本地模型服务器

使用vLLM启动一个OpenAI兼容的API服务器：

```bash
# 安装vLLM
pip install vllm

# 启动OpenAI兼容的API服务器
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/qwen3-8b-instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2  # 根据您的GPU数量调整
```

如果您使用的是其他框架（如FastChat），请相应调整命令。

## 4. 处理数据和构建知识图谱

在另一个终端会话中，运行以下命令：

```bash
# 连接到云服务器
ssh username@your-server-ip

# 进入项目目录
cd /path/to/secondKarlMarx/marx_kg

# 处理数据
python scripts/process_data.py

# 构建知识图谱
python scripts/build_kg.py
```

## 5. 测试知识图谱查询

测试知识图谱查询功能：

```bash
# 运行测试脚本
python scripts/test_kg.py

# 或者运行交互式查询
python scripts/query_kg.py
```

## 6. 测试工具调用集成

测试工具调用功能：

```bash
# 测试kg_tool.py
python scripts/kg_tool.py "马克思关于商品的定义是什么？"

# 如果您已经微调了模型，测试MCP集成
python scripts/mcp_server.py --model_path /path/to/secondKarlMarx-model --port 8080
```

## 7. 监控和调试

监控系统资源使用情况和日志：

```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 查看日志
tail -f *.log
```

## 8. 将示例转换为训练数据

如果您需要为模型微调准备训练数据：

```bash
# 进入examples目录
cd /path/to/secondKarlMarx/marx_kg/examples

# 将工具调用示例转换为LLaMA Factory格式
python convert_to_llama_factory.py --format sharegpt
```

## 9. 使用LLaMA Factory进行微调

使用转换后的数据进行微调：

```bash
# 进入LLaMA Factory目录
cd /path/to/LLaMA-Factory

# 运行微调
CUDA_VISIBLE_DEVICES=0,1 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /path/to/qwen3-8b-instruct \
    --dataset /path/to/secondKarlMarx/marx_kg/examples/llama_factory_examples.json \
    --template default \
    --finetuning_type lora \
    --lora_target all \
    --output_dir /path/to/output/secondKarlMarx-kg-tool \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --fp16
```

## 10. 完整工作流程测试

最后，测试完整的工作流程：

1. 启动微调后的模型服务器
2. 启动集成了知识图谱工具的MCP服务器
3. 从本地笔记本通过MCP客户端连接并测试查询

```bash
# 在云服务器上启动MCP服务器
python /path/to/secondKarlMarx/marx_kg/scripts/mcp_server.py \
    --model_path /path/to/secondKarlMarx-kg-tool \
    --port 8080

# 在本地笔记本上
python mcp_client.py --server your-server-ip:8080 --query "请解释马克思的剩余价值理论"
```

这个测试指南涵盖了从数据准备到最终部署的完整流程，帮助您在云服务器上验证马克思恩格斯知识图谱的功能。
