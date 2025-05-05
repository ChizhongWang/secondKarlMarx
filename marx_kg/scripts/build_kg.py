"""
马克思恩格斯知识图谱构建脚本
使用GraphRAG构建知识图谱
"""
import os
import sys
import json
import logging
import pandas as pd
import networkx as nx
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
GRAPHRAG_ROOT = PROJECT_ROOT.parent / "graphrag"
sys.path.append(str(GRAPHRAG_ROOT))

# 导入GraphRAG
from graphrag.config.load_config import load_config
from graphrag.config.enums import AsyncType
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
from graphrag.cache.noop_pipeline_cache import NoopPipelineCache
from graphrag.cache.memory_pipeline_cache import InMemoryCache

# 导入操作函数 - 使用正确的导入路径
from graphrag.index.operations.extract_graph.extract_graph import extract_graph
from graphrag.index.operations.chunk_text.chunk_text import chunk_text
from graphrag.index.operations.embed_text.embed_text import embed_text
from graphrag.index.operations.prune_graph import prune_graph
from graphrag.index.operations.embed_graph.embed_graph import embed_graph
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import summarize_descriptions

# 导入常量定义
from graphrag.config.defaults import DEFAULT_CHAT_MODEL_ID, DEFAULT_EMBEDDING_MODEL_ID

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("build_kg.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def build_knowledge_graph():
    """构建知识图谱"""
    logger.info("开始构建马克思恩格斯知识图谱")
    
    # 加载配置
    config_path = PROJECT_ROOT / "config" / "graphrag_config.yaml"
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(PROJECT_ROOT, config_path)
    
    # 创建回调和缓存
    callbacks = ConsoleWorkflowCallbacks()  # 使用控制台回调，它会在控制台显示进度
    cache = InMemoryCache()  # 使用内存缓存，不需要传递参数
    
    # 加载处理后的文档
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "marx_engels_documents.json"
    logger.info(f"加载处理后的文档: {processed_data_path}")
    
    with open(processed_data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    # 将文档转换为DataFrame
    text_units = pd.DataFrame(documents)
    logger.info(f"加载了 {len(text_units)} 个文档")
    
    # 提取图谱
    logger.info("步骤1: 提取实体和关系")
    extract_config = config.extract_graph if hasattr(config, 'extract_graph') else None
    
    # 默认的实体类型
    default_entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "WORK_OF_ART", "EVENT", "DATE", "CONCEPT"]
    
    # 如果extract_config存在且有entity_types属性，则使用它，否则使用默认值
    entity_types = extract_config.entity_types if extract_config and hasattr(extract_config, 'entity_types') else default_entity_types
    
    # 获取默认聊天模型配置
    default_chat_model = config.models[DEFAULT_CHAT_MODEL_ID] if DEFAULT_CHAT_MODEL_ID in config.models else None
    
    # 提取策略
    strategy = None
    if extract_config and hasattr(extract_config, 'resolved_strategy') and default_chat_model:
        strategy = extract_config.resolved_strategy(str(PROJECT_ROOT), default_chat_model)
    
    entities, relationships = await extract_graph(
        text_units=text_units,
        callbacks=callbacks,
        cache=cache,
        text_column="content",
        id_column="id",
        strategy=strategy,
        async_mode=AsyncType.AsyncIO,
        entity_types=entity_types,
        num_threads=4
    )
    
    logger.info(f"提取了 {len(entities)} 个实体和 {len(relationships)} 个关系")
    
    # 创建图谱
    logger.info("步骤2: 创建图谱")
    G = nx.Graph()
    
    # 添加节点
    for _, row in entities.iterrows():
        G.add_node(
            row['title'],
            type=row['type'],
            frequency=row['frequency'],
            description=row['description']
        )
    
    # 添加边
    for _, row in relationships.iterrows():
        G.add_edge(
            row['source'],
            row['target'],
            weight=row['weight'],
            description=row['description']
        )
    
    logger.info(f"创建了包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边的图谱")
    
    # 修剪图谱
    logger.info("步骤3: 修剪图谱")
    prune_config = config.prune_graph if hasattr(config, 'prune_graph') else None
    
    # 默认的修剪配置
    default_min_node_freq = 1
    default_min_node_degree = 1
    default_min_edge_weight_pct = 40
    
    # 如果prune_config存在且有对应属性，则使用它，否则使用默认值
    min_node_freq = prune_config.min_node_freq if prune_config and hasattr(prune_config, 'min_node_freq') else default_min_node_freq
    min_node_degree = prune_config.min_node_degree if prune_config and hasattr(prune_config, 'min_node_degree') else default_min_node_degree
    min_edge_weight_pct = prune_config.min_edge_weight_pct if prune_config and hasattr(prune_config, 'min_edge_weight_pct') else default_min_edge_weight_pct
    
    pruned_graph = prune_graph(
        graph=G,
        min_node_freq=min_node_freq,
        min_node_degree=min_node_degree,
        min_edge_weight_pct=min_edge_weight_pct
    )
    
    logger.info(f"修剪后保留 {len(pruned_graph.nodes)} 个节点和 {len(pruned_graph.edges)} 条边")
    
    # 保存图谱
    logger.info("步骤4: 保存图谱")
    kg_output_dir = PROJECT_ROOT / "kg" / "output"
    kg_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为GraphML格式
    graph_path = kg_output_dir / "marx_engels_kg.graphml"
    nx.write_graphml(pruned_graph, graph_path)
    
    # 保存节点和边的信息
    nodes_path = kg_output_dir / "nodes.json"
    edges_path = kg_output_dir / "edges.json"
    
    nodes_data = [{"id": n, **pruned_graph.nodes[n]} for n in pruned_graph.nodes]
    edges_data = [{"source": u, "target": v, **pruned_graph.edges[u, v]} for u, v in pruned_graph.edges]
    
    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump(nodes_data, f, ensure_ascii=False, indent=2)
    
    with open(edges_path, "w", encoding="utf-8") as f:
        json.dump(edges_data, f, ensure_ascii=False, indent=2)
    
    logger.info("知识图谱构建完成")
    logger.info(f"图谱文件保存在: {kg_output_dir}")
    
    return kg_output_dir

async def main():
    """主函数"""
    try:
        kg_dir = await build_knowledge_graph()
        logger.info(f"知识图谱已保存到: {kg_dir}")
        print(f"\n知识图谱构建成功！\n图谱文件保存在: {kg_dir}")
    except Exception as e:
        logger.error(f"构建知识图谱时出错: {str(e)}", exc_info=True)
        print(f"\n构建知识图谱时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
