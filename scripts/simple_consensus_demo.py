#!/usr/bin/env python3
"""
简化版MDT共识演示
展示从患者数据到共识达成的关键步骤
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.llm_interface import LLMConfig, LLMInterface
from src.knowledge.enhanced_faiss_integration import EnhancedFAISSManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.consensus.dialogue_manager import MultiAgentDialogueManager

def demonstrate_consensus_flow():
    """演示共识达成流程"""
    
    print("🏥 MDT医疗AI系统 - 共识达成流程演示")
    print("=" * 60)
    
    # 步骤1: 患者数据输入
    print("\n📋 步骤1: 患者数据输入")
    print("-" * 30)
    patient_data = {
        "patient_id": "DEMO_001",
        "age": 58,
        "diagnosis": "乳腺癌",
        "stage": "IIIA期",
        "symptoms": ["乳房肿块", "疲劳", "轻微疼痛"],
        "comorbidities": ["高血压", "2型糖尿病"],
        "lab_results": {
            "血红蛋白": "11.2 g/dL",
            "白细胞": "6800/μL", 
            "血小板": "280,000/μL",
            "肌酐": "0.9 mg/dL"
        }
    }
    
    print(f"👤 患者: {patient_data['patient_id']}")
    print(f"📊 基本信息: {patient_data['age']}岁, {patient_data['diagnosis']}, {patient_data['stage']}")
    print(f"🔬 主要症状: {', '.join(patient_data['symptoms'])}")
    print(f"⚕️  合并症: {', '.join(patient_data['comorbidities'])}")
    
    # 步骤2: 知识检索 // 这里可以先不管
    print("\n🔍 步骤2: 医学知识检索")
    print("-" * 30)
    print("📚 检索查询: 'IIIA期乳腺癌治疗方案'")
    print("🔎 检索到相关知识:")
    print("   • NCCN乳腺癌治疗指南")
    print("   • 新辅助化疗临床研究")
    print("   • 保乳手术适应症")
    print("   • 放疗剂量分割方案")
    print("✅ 知识检索完成")
    
    # 步骤3: 智能体初始化
    print("\n👥 步骤3: MDT智能体初始化")
    print("-" * 30)
    agents = [
        "🩺 肿瘤科医生",
        "🔪 外科医生", 
        "📡 放射科医生",
        "🔬 病理科医生",
        "👩 专科护士"
    ]
    
    for agent in agents:
        print(f"✅ {agent} 初始化完成")
    
    # 步骤4: 初始意见生成
    print("\n💭 步骤4: 生成初始专业意见")
    print("-" * 30)
    
    initial_opinions = {
        "肿瘤科医生": {
            "化疗": {"评分": 0.85, "推理": "IIIA期乳腺癌标准治疗，新辅助化疗可缩小肿瘤"},
            "手术": {"评分": 0.75, "推理": "化疗后可考虑手术，需评估手术风险"},
            "放疗": {"评分": 0.70, "推理": "术后辅助放疗，降低局部复发风险"}
        },
        "外科医生": {
            "手术": {"评分": 0.80, "推理": "根治性手术是关键，但需考虑患者合并症"},
            "化疗": {"评分": 0.75, "推理": "新辅助化疗有助于手术切除"},
            "放疗": {"评分": 0.65, "推理": "术后放疗作为辅助治疗"}
        },
        "放射科医生": {
            "放疗": {"评分": 0.78, "推理": "精确放疗技术可有效控制局部病灶"},
            "化疗": {"评分": 0.70, "推理": "化疗联合放疗有协同效应"},
            "手术": {"评分": 0.68, "推理": "影像评估显示手术可行性"}
        }
    }
    
    for doctor, opinions in initial_opinions.items():
        print(f"\n🔬 {doctor} 的初始意见:")
        for treatment, data in opinions.items():
            print(f"   {treatment}: {data['评分']:.2f} - {data['推理']}")
    
    # 步骤5: 多轮对话协商
    print("\n💬 步骤5: 多轮对话协商")
    print("-" * 30)
    
    dialogue_rounds = [
        {
            "轮次": 1,
            "主题": "治疗方案优先级讨论",
            "发言": [
                "肿瘤科医生: 建议先行新辅助化疗，评估肿瘤反应后决定手术方案",
                "外科医生: 同意化疗优先，但需密切监测患者耐受性",
                "放射科医生: 支持多学科综合治疗，放疗时机需要协调",
                "病理科医生: 建议获取更多分子标志物信息指导治疗",
                "专科护士: 关注患者心理状态，需要充分的患者教育"
            ],
            "收敛度": 0.65
        },
        {
            "轮次": 2, 
            "主题": "治疗时序和剂量调整",
            "发言": [
                "肿瘤科医生: 考虑到糖尿病，化疗剂量需要适当调整",
                "外科医生: 化疗4-6周期后评估，如果反应良好可行保乳手术",
                "放射科医生: 术后放疗剂量50Gy/25次，需要心脏保护",
                "病理科医生: 化疗后病理完全缓解率约30%，需要动态评估",
                "专科护士: 制定个性化护理计划，重点关注血糖管理"
            ],
            "收敛度": 0.78
        },
        {
            "轮次": 3,
            "主题": "最终方案确认",
            "发言": [
                "肿瘤科医生: 最终方案：新辅助化疗→手术→辅助放疗",
                "外科医生: 同意该方案，手术方式根据化疗反应决定",
                "放射科医生: 放疗计划已制定，与手术时机协调",
                "病理科医生: 支持该综合治疗方案",
                "专科护士: 全程护理方案已准备就绪"
            ],
            "收敛度": 0.92
        }
    ]
    
    for round_data in dialogue_rounds:
        print(f"\n🔄 第 {round_data['轮次']} 轮对话 - {round_data['主题']}")
        for statement in round_data['发言']:
            print(f"   💬 {statement}")
        print(f"   📊 收敛度: {round_data['收敛度']:.2f}")
        
        if round_data['收敛度'] > 0.9:
            print("   🎉 达成高度共识！")
    
    # 步骤6: 共识评估
    print("\n🎯 步骤6: 共识评估")
    print("-" * 30)
    
    final_scores = {
        "新辅助化疗": 0.88,
        "手术治疗": 0.82, 
        "辅助放疗": 0.79
    }
    
    print("📊 最终治疗方案评分:")
    for treatment, score in final_scores.items():
        print(f"   {treatment}: {score:.2f}")
    
    recommended_treatment = max(final_scores.items(), key=lambda x: x[1])
    print(f"\n🎯 推荐治疗方案: {recommended_treatment[0]}")
    print(f"💪 共识强度: {recommended_treatment[1]:.2f}")
    print(f"🔄 对话轮数: 3轮")
    print(f"✅ 共识状态: 已达成")
    
    # 步骤7: RL优化反馈
    print("\n🤖 步骤7: 强化学习优化反馈")
    print("-" * 30)
    
    rl_feedback = {
        "奖励值": 0.85,
        "动作": "新辅助化疗",
        "环境反馈": "治疗方案与最佳实践高度匹配",
        "优化建议": [
            "当前治疗方案获得高度认可",
            "建议密切监测治疗反应",
            "可考虑个性化剂量调整"
        ]
    }
    
    print(f"🏆 RL奖励: {rl_feedback['奖励值']:.2f}")
    print(f"🎯 执行动作: {rl_feedback['动作']}")
    print(f"📊 环境反馈: {rl_feedback['环境反馈']}")
    print("💡 优化建议:")
    for suggestion in rl_feedback['优化建议']:
        print(f"   • {suggestion}")
    
    # 步骤8: 最终报告
    print("\n📄 步骤8: 生成最终报告")
    print("-" * 30)
    
    final_report = {
        "报告ID": f"MDT_DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "患者信息": patient_data,
        "推荐方案": {
            "主要治疗": recommended_treatment[0],
            "共识强度": recommended_treatment[1],
            "治疗序列": "新辅助化疗 → 手术 → 辅助放疗"
        },
        "MDT讨论": {
            "参与专家": len(agents),
            "对话轮数": 3,
            "最终收敛度": 0.92
        },
        "RL优化": rl_feedback
    }
    
    # 保存报告
    report_filename = f"demo_consensus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"💾 报告已保存: {report_filename}")
    
    # 流程总结
    print("\n" + "=" * 60)
    print("🎉 MDT共识达成流程演示完成！")
    print("=" * 60)
    
    print("\n📋 流程总结:")
    print("1. 📊 患者数据输入 → 结构化患者信息")
    print("2. 🔍 知识检索 → 获取相关医学证据")  
    print("3. 👥 智能体初始化 → 5个专业角色就位")
    print("4. 💭 初始意见生成 → 各专业独立评估")
    print("5. 💬 多轮对话协商 → 3轮讨论达成共识")
    print("6. 🎯 共识评估 → 确定最终治疗方案")
    print("7. 🤖 RL优化反馈 → 强化学习验证")
    print("8. 📄 报告生成 → 完整决策记录")
    
    print(f"\n🎯 最终结果:")
    print(f"   推荐治疗: {recommended_treatment[0]}")
    print(f"   共识强度: {recommended_treatment[1]:.2f}")
    print(f"   RL奖励: {rl_feedback['奖励值']:.2f}")
    
    return final_report

if __name__ == "__main__":
    demonstrate_consensus_flow()
    cfg = LLMConfig()
    llm = LLMInterface(cfg)
    faiss_mgr = EnhancedFAISSManager(db_path="clinical_memory_db")
    rag = MedicalKnowledgeRAG(faiss_manager=faiss_mgr)
    manager = MultiAgentDialogueManager(rag_system=rag, llm_interface=llm)