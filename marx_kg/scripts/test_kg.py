"""
测试马克思恩格斯知识图谱
"""
import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入知识图谱查询工具
from scripts.kg_tool import query_marx_kg, test_query

def run_test_queries():
    """运行一系列测试查询"""
    test_queries = [
        "马克思关于资本主义的观点是什么？",
        "资本论中对商品的定义是什么？",
        "马克思和恩格斯的合作关系如何？",
        "马克思主义对历史的看法是什么？",
        "马克思的主要著作有哪些？"
    ]
    
    print("\n=== 马克思恩格斯知识图谱测试 ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n测试 {i}/{len(test_queries)}: {query}")
        print("-" * 50)
        
        # 执行查询
        result = query_marx_kg(query)
        
        # 打印结果
        print(f"回答: {result['answer']}")
        
        if result['entities']:
            print("\n相关实体:")
            for entity in result['entities']:
                print(f"  - {entity.get('text', '')}: {entity.get('type', '')}")
        
        if result['relations']:
            print("\n相关关系:")
            for relation in result['relations']:
                print(f"  - {relation.get('head_text', '')} {relation.get('relation', '')} {relation.get('tail_text', '')}")
        
        print("\n" + "=" * 50)

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 如果提供了命令行参数，将其作为查询
        test_query(' '.join(sys.argv[1:]))
    else:
        # 否则运行测试查询集
        run_test_queries()

if __name__ == "__main__":
    main()
