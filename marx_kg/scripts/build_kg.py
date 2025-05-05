"""
马克思恩格斯知识图谱构建脚本
使用GraphRAG构建知识图谱
"""
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
GRAPHRAG_ROOT = PROJECT_ROOT.parent / "graphrag"
sys.path.append(str(GRAPHRAG_ROOT))

# 导入GraphRAG
from graphrag.config.load_config import load_config
from graphrag.index.operations.extract_graph import ExtractGraph
from graphrag.index.operations.embed_text import EmbedText
from graphrag.index.operations.prune_graph import PruneGraph
from graphrag.index.operations.embed_graph import EmbedGraph
from graphrag.index.operations.summarize_descriptions import SummarizeDescriptions

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

def build_knowledge_graph():
    """构建知识图谱"""
    logger.info("开始构建马克思恩格斯知识图谱")
    
    # 加载配置
    config_path = PROJECT_ROOT / "config" / "graphrag_config.yaml"
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(PROJECT_ROOT, config_path)
    
    # 提取图谱
    logger.info("步骤1: 提取实体和关系")
    extract_graph = ExtractGraph(config)
    graph = extract_graph.run()
    logger.info(f"提取了 {len(graph.nodes)} 个实体和 {len(graph.edges)} 个关系")
    
    # 修剪图谱
    logger.info("步骤2: 修剪图谱")
    prune_graph = PruneGraph(config)
    pruned_graph = prune_graph.run()
    logger.info(f"修剪后保留 {len(pruned_graph.nodes)} 个实体和 {len(pruned_graph.edges)} 个关系")
    
    # 生成嵌入
    logger.info("步骤3: 生成文本嵌入")
    embed_text = EmbedText(config)
    embed_text.run()
    
    # 嵌入图谱
    logger.info("步骤4: 嵌入图谱")
    embed_graph = EmbedGraph(config)
    embed_graph.run()
    
    # 生成摘要描述
    logger.info("步骤5: 生成实体摘要描述")
    summarize = SummarizeDescriptions(config)
    summarize.run()
    
    logger.info("知识图谱构建完成")
    
    # 返回图谱路径
    kg_output_dir = PROJECT_ROOT / "kg" / "output"
    return kg_output_dir

def main():
    """主函数"""
    try:
        kg_dir = build_knowledge_graph()
        logger.info(f"知识图谱已保存到: {kg_dir}")
        print(f"\n知识图谱构建成功！\n图谱文件保存在: {kg_dir}")
    except Exception as e:
        logger.error(f"构建知识图谱时出错: {str(e)}", exc_info=True)
        print(f"\n构建知识图谱时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
