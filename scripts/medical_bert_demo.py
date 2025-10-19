#!/usr/bin/env python3
"""
医疗BERT模型演示脚本
文件路径: scripts/medical_bert_demo.py
作者: AI Assistant
功能: 演示如何使用和评估不同的医疗BERT模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge.dialogue_memory_manager import DialogueMemoryManager
import json
import time

def demo_medical_bert_models():
    """演示医疗BERT模型的使用"""
    
    print("🏥 医疗BERT模型演示")
    print("=" * 50)
    
    # 1. 查看可用模型
    print("\n📋 1. 查看可用的医疗模型")
    manager = DialogueMemoryManager(memory_db_path="demo_medical_bert_db")
    available_models = manager.get_available_medical_models()
    
    print(f"当前模型: {available_models['current_model']['name']}")
    print(f"模型类型: {available_models['current_model']['type']}")
    print(f"推荐: {available_models['recommendation']}")
    
    # 2. 评估当前模型性能
    print("\n📊 2. 评估当前模型性能")
    performance = manager.evaluate_model_performance()
    
    if "error" not in performance:
        print(f"模型维度: {performance['model_info']['dimension']}")
        print(f"平均向量范数: {performance['embedding_stats']['mean_norm']:.4f}")
        print(f"语义相似度均值: {performance['semantic_quality']['avg_similarity']:.4f}")
    else:
        print(f"评估失败: {performance['error']}")
    
    # 3. 测试医疗文本理解
    print("\n🧠 3. 测试医疗文本理解能力")
    test_medical_texts = [
        "患者主诉胸痛3天，伴有呼吸困难",
        "建议进行冠状动脉造影检查",
        "血常规显示白细胞计数升高",
        "术后患者恢复良好，无并发症",
        "化疗方案需要根据患者耐受性调整"
    ]
    
    print("测试文本:")
    for i, text in enumerate(test_medical_texts, 1):
        print(f"  {i}. {text}")
    
    # 生成嵌入向量并分析
    embeddings = manager.embedding_model.encode(test_medical_texts)
    print(f"\n生成嵌入向量: {embeddings.shape}")
    
    # 计算文本间相似度
    import numpy as np
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    print("\n文本相似度矩阵 (前3x3):")
    for i in range(min(3, len(similarity_matrix))):
        row = " ".join([f"{similarity_matrix[i][j]:.3f}" for j in range(min(3, len(similarity_matrix[i])))])
        print(f"  [{row}]")
    
    # 4. 尝试切换到医疗专业模型
    print("\n🔄 4. 尝试切换到医疗专业模型")
    
    medical_models_to_try = [
        "auto",  # 自动选择
        "biobert",  # BioBERT
        "clinical-bert"  # ClinicalBERT
    ]
    
    for model_name in medical_models_to_try:
        print(f"\n尝试切换到: {model_name}")
        
        # 创建新的管理器实例来测试
        try:
            test_manager = DialogueMemoryManager(
                memory_db_path=f"demo_test_{model_name}_db",
                embedding_model_name=model_name
            )
            
            print(f"✅ 成功加载: {test_manager.model_name}")
            print(f"   模型类型: {test_manager._get_model_type()}")
            print(f"   嵌入维度: {test_manager.embedding_dim}")
            
            # 快速性能测试
            start_time = time.time()
            test_embeddings = test_manager.embedding_model.encode(test_medical_texts[:2])
            end_time = time.time()
            
            print(f"   处理速度: {(end_time - start_time)*1000:.2f}ms (2个文本)")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
    
    # 5. 医疗文本相似性搜索演示
    print("\n🔍 5. 医疗文本相似性搜索演示")
    
    # 添加一些医疗对话记录
    sample_dialogues = [
        {
            "patient_id": "P001",
            "user_query": "我最近总是感到胸闷气短，这是什么原因？",
            "agent_response": "胸闷气短可能与心血管疾病、呼吸系统疾病或焦虑等因素有关，建议进行心电图和胸部X光检查。"
        },
        {
            "patient_id": "P002", 
            "user_query": "化疗后出现恶心呕吐，有什么缓解方法？",
            "agent_response": "化疗引起的恶心呕吐是常见副作用，可以使用止吐药物，同时注意饮食调理，少量多餐。"
        },
        {
            "patient_id": "P003",
            "user_query": "血压一直控制不好，需要换药吗？",
            "agent_response": "血压控制不佳需要评估当前用药方案，可能需要调整剂量或更换降压药物，建议咨询心内科医生。"
        }
    ]
    
    # 保存对话记录
    for dialogue in sample_dialogues:
        manager.save_dialogue_turn(
            patient_id=dialogue["patient_id"],
            user_query=dialogue["user_query"],
            agent_response=dialogue["agent_response"]
        )
    
    # 测试相似性搜索
    search_query = "我胸口疼痛，呼吸有点困难"
    print(f"\n搜索查询: '{search_query}'")
    
    similar_dialogues = manager.search_similar_dialogues(
        query=search_query,
        k=3,
        similarity_threshold=0.3
    )
    
    print(f"找到 {len(similar_dialogues)} 条相似对话:")
    for i, dialogue in enumerate(similar_dialogues, 1):
        print(f"  {i}. 相似度: {dialogue.get('similarity_score', 0):.3f}")
        print(f"     患者查询: {dialogue.get('user_query', '')[:50]}...")
        print(f"     系统回复: {dialogue.get('agent_response', '')[:50]}...")
        print()
    
    # 6. 模型使用建议
    print("\n📈 6. 模型使用建议")
    print("=" * 50)
    
    recommendations = {
        "英文医疗文本": "推荐使用 BioBERT 或 ClinicalBERT",
        "中文医疗文本": "推荐使用中文医疗BERT或多语言模型", 
        "通用场景": "可以使用 sentence-transformers 通用模型",
        "高精度需求": "推荐使用领域专业的BERT模型",
        "快速响应": "推荐使用轻量级模型如 MiniLM"
    }
    
    for scenario, recommendation in recommendations.items():
        print(f"• {scenario}: {recommendation}")
    
    print(f"\n✅ 演示完成！")
    print(f"💡 提示: 可以通过 manager.switch_embedding_model() 动态切换模型")

def compare_model_performance():
    """比较不同模型的性能"""
    
    print("\n🏆 医疗BERT模型性能比较")
    print("=" * 50)
    
    models_to_compare = [
        ("通用模型", "sentence-transformers/all-MiniLM-L6-v2"),
        ("自动选择", "auto"),
        ("BioBERT", "biobert"),
        ("ClinicalBERT", "clinical-bert")
    ]
    
    test_queries = [
        "患者出现急性心肌梗死症状",
        "肿瘤标志物检查结果异常",
        "术后感染风险评估",
        "化疗药物副作用管理",
        "影像学检查显示肺部阴影"
    ]
    
    results = []
    
    for model_desc, model_name in models_to_compare:
        print(f"\n测试 {model_desc} ({model_name})...")
        
        try:
            manager = DialogueMemoryManager(
                memory_db_path=f"compare_{model_name.replace('/', '_')}_db",
                embedding_model_name=model_name
            )
            
            # 性能评估
            start_time = time.time()
            performance = manager.evaluate_model_performance(test_queries)
            end_time = time.time()
            
            if "error" not in performance:
                result = {
                    "model_desc": model_desc,
                    "model_name": manager.model_name,
                    "dimension": performance['model_info']['dimension'],
                    "avg_similarity": performance['semantic_quality']['avg_similarity'],
                    "evaluation_time": end_time - start_time,
                    "status": "成功"
                }
            else:
                result = {
                    "model_desc": model_desc,
                    "model_name": model_name,
                    "status": "失败",
                    "error": performance['error']
                }
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "model_desc": model_desc,
                "model_name": model_name,
                "status": "异常",
                "error": str(e)
            })
    
    # 显示比较结果
    print("\n📊 性能比较结果:")
    print("-" * 80)
    print(f"{'模型':<15} {'维度':<8} {'语义相似度':<12} {'评估时间':<10} {'状态':<8}")
    print("-" * 80)
    
    for result in results:
        if result['status'] == '成功':
            print(f"{result['model_desc']:<15} {result['dimension']:<8} "
                  f"{result['avg_similarity']:<12.4f} {result['evaluation_time']:<10.3f}s {result['status']:<8}")
        else:
            print(f"{result['model_desc']:<15} {'N/A':<8} {'N/A':<12} {'N/A':<10} {result['status']:<8}")
    
    print("-" * 80)

if __name__ == "__main__":
    print("🚀 启动医疗BERT模型演示")
    
    try:
        demo_medical_bert_models()
        compare_model_performance()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 演示结束")