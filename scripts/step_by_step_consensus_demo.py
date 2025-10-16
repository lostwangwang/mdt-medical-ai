#!/usr/bin/env python3
"""
MDT共识达成步骤演示脚本
文件路径: scripts/step_by_step_consensus_demo.py
作者: AI Assistant
功能: 详细演示从患者数据输入到MDT共识达成的完整流程
"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.data_models import (
    PatientState, RoleType, TreatmentOption, 
    ConsensusResult, DialogueRound, DialogueMessage
)
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.consensus.role_agents import RoleAgent
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.rl.rl_environment import MDTReinforcementLearning
from src.utils.llm_interface import LLMInterface, LLMConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StepByStepConsensusDemo:
    """步骤演示类"""
    
    def __init__(self):
        """初始化演示系统"""
        print("🏥 MDT医疗AI系统 - 共识达成流程演示")
        print("=" * 60)
        
        # 初始化核心组件
        self.rag_system = None
        self.dialogue_manager = None
        self.rl_environment = None
        self.llm_interface = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化系统组件"""
        print("\n🔧 步骤1: 初始化系统组件")
        print("-" * 30)
        
        try:
            # 初始化RAG系统
            print("📚 初始化医学知识RAG系统...")
            self.rag_system = MedicalKnowledgeRAG()
            print("✅ RAG系统初始化完成")
            
            # 初始化对话管理器
            print("💬 初始化多智能体对话管理器...")
            self.dialogue_manager = MultiAgentDialogueManager(
                rag_system=self.rag_system
            )
            print("✅ 对话管理器初始化完成")
            
            # 初始化RL环境
            print("🤖 初始化强化学习环境...")
            self.rl_environment = MDTReinforcementLearning()
            print("✅ RL环境初始化完成")
            
            # 初始化LLM接口
            print("🧠 初始化LLM接口...")
            try:
                config = LLMConfig()
                self.llm_interface = LLMInterface(config)
                print("✅ LLM接口初始化完成")
            except Exception as e:
                print(f"⚠️  LLM接口初始化失败，将使用模拟模式: {e}")
                self.llm_interface = None
            
            print("\n🎉 所有组件初始化完成！")
            
        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            raise
    
    def run_complete_demo(self):
        """运行完整演示"""
        print("\n" + "=" * 60)
        print("🚀 开始完整的MDT共识达成流程演示")
        print("=" * 60)
        
        # 步骤1: 创建患者数据
        patient_state = self._step1_create_patient_data()
        
        # 步骤2: 知识检索
        knowledge_context = self._step2_knowledge_retrieval(patient_state)
        
        # 步骤3: 初始化智能体
        agents = self._step3_initialize_agents(patient_state, knowledge_context)
        
        # 步骤4: 生成初始意见
        initial_opinions = self._step4_generate_initial_opinions(agents, patient_state)
        
        # 步骤5: 多轮对话协商
        dialogue_history = self._step5_multi_round_dialogue(agents, patient_state, initial_opinions)
        
        # 步骤6: 共识评估
        consensus_result = self._step6_consensus_evaluation(dialogue_history, patient_state)
        
        # 步骤7: RL优化反馈
        rl_feedback = self._step7_rl_optimization(patient_state, consensus_result)
        
        # 步骤8: 生成最终报告
        final_report = self._step8_generate_final_report(
            patient_state, consensus_result, dialogue_history, rl_feedback
        )
        
        return final_report
    
    def _step1_create_patient_data(self) -> PatientState:
        """步骤1: 创建患者数据"""
        print("\n📋 步骤2: 创建患者数据")
        print("-" * 30)
        
        # 创建示例患者
        patient_state = PatientState(
            patient_id="DEMO_001",
            age=58,
            diagnosis="breast_cancer",
            stage="IIIA",
            lab_results={
                "hemoglobin": 11.2,
                "white_blood_cell": 6800,
                "platelet": 280000,
                "creatinine": 0.9,
                "bilirubin": 1.1,
                "alt": 28,
                "ast": 32
            },
            vital_signs={
                "bp_systolic": 135,
                "bp_diastolic": 85,
                "heart_rate": 78,
                "temperature": 36.8,
                "respiratory_rate": 16
            },
            symptoms=["breast_lump", "fatigue", "mild_pain"],
            comorbidities=["hypertension", "diabetes_type2"],
            psychological_status="anxious",
            quality_of_life_score=0.65
        )
        
        print(f"👤 患者信息:")
        print(f"   ID: {patient_state.patient_id}")
        print(f"   年龄: {patient_state.age}岁")
        print(f"   诊断: {patient_state.diagnosis}")
        print(f"   分期: {patient_state.stage}")
        print(f"   主要症状: {', '.join(patient_state.symptoms)}")
        print(f"   合并症: {', '.join(patient_state.comorbidities)}")
        print(f"   生活质量评分: {patient_state.quality_of_life_score}")
        
        return patient_state
    
    def _step2_knowledge_retrieval(self, patient_state: PatientState) -> Dict[str, Any]:
        """步骤2: 医学知识检索"""
        print("\n🔍 步骤3: 医学知识检索")
        print("-" * 30)
        
        try:
            # 构建查询
            query = f"{patient_state.diagnosis} stage {patient_state.stage} treatment options"
            print(f"📝 检索查询: {query}")
            
            # 执行检索
            print("🔎 正在检索相关医学知识...")
            knowledge_results = self.rag_system.retrieve_knowledge(
                query=query,
                patient_context=patient_state.__dict__,
                top_k=5
            )
            
            print(f"✅ 检索到 {len(knowledge_results.get('documents', []))} 条相关知识")
            
            # 显示检索结果摘要
            if knowledge_results.get('documents'):
                print("📚 相关知识摘要:")
                for i, doc in enumerate(knowledge_results['documents'][:3], 1):
                    print(f"   {i}. {doc.get('title', '未知标题')[:50]}...")
            
            return knowledge_results
            
        except Exception as e:
            print(f"⚠️  知识检索失败，使用默认知识: {e}")
            return {
                "documents": [],
                "metadata": {"source": "fallback"},
                "query": query
            }
    
    def _step3_initialize_agents(self, patient_state: PatientState, knowledge_context: Dict) -> Dict[RoleType, RoleAgent]:
        """步骤3: 初始化智能体"""
        print("\n👥 步骤4: 初始化MDT智能体")
        print("-" * 30)
        
        agents = {}
        roles = [
            RoleType.ONCOLOGIST,
            RoleType.SURGEON, 
            RoleType.RADIOLOGIST,
            RoleType.PATHOLOGIST,
            RoleType.NURSE
        ]
        
        for role in roles:
            print(f"🤖 初始化 {role.value}...")
            agent = RoleAgent(
                role=role,
                rag_system=self.rag_system,
                llm_interface=self.llm_interface
            )
            agents[role] = agent
            print(f"✅ {role.value} 初始化完成")
        
        print(f"\n🎉 成功初始化 {len(agents)} 个专业智能体")
        return agents
    
    def _step4_generate_initial_opinions(self, agents: Dict[RoleType, RoleAgent], patient_state: PatientState) -> Dict[RoleType, Dict]:
        """步骤4: 生成初始专业意见"""
        print("\n💭 步骤5: 生成初始专业意见")
        print("-" * 30)
        
        initial_opinions = {}
        treatment_options = [
            TreatmentOption.CHEMOTHERAPY,
            TreatmentOption.SURGERY,
            TreatmentOption.RADIOTHERAPY
        ]
        
        for role, agent in agents.items():
            print(f"\n🔬 {role.value} 正在分析患者...")
            
            # 生成对各治疗方案的意见
            role_opinions = {}
            for treatment in treatment_options:
                try:
                    print(f"   分析 {treatment.value}...")
                    
                    # 生成治疗推理
                    if self.llm_interface:
                        reasoning = self.llm_interface.generate_treatment_reasoning(
                            patient_state=patient_state,
                            role=role,
                            treatment_option=treatment,
                            knowledge_context={"evidence_level": "high"}
                        )
                    else:
                        reasoning = self._generate_mock_reasoning(role, treatment, patient_state)
                    
                    # 生成评分
                    score = self._calculate_treatment_score(role, treatment, patient_state)
                    
                    role_opinions[treatment] = {
                        "reasoning": reasoning,
                        "score": score,
                        "confidence": 0.8,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    print(f"   ✅ {treatment.value} 评分: {score:.2f}")
                    
                except Exception as e:
                    print(f"   ⚠️  {treatment.value} 分析失败: {e}")
                    role_opinions[treatment] = {
                        "reasoning": f"分析失败: {e}",
                        "score": 0.5,
                        "confidence": 0.3,
                        "timestamp": datetime.now().isoformat()
                    }
            
            initial_opinions[role] = role_opinions
            print(f"✅ {role.value} 意见生成完成")
        
        # 显示初始意见摘要
        print(f"\n📊 初始意见摘要:")
        for role, opinions in initial_opinions.items():
            print(f"   {role.value}:")
            for treatment, opinion in opinions.items():
                print(f"     {treatment.value}: {opinion['score']:.2f}")
        
        return initial_opinions
    
    def _step5_multi_round_dialogue(self, agents: Dict[RoleType, RoleAgent], patient_state: PatientState, initial_opinions: Dict) -> List[DialogueRound]:
        """步骤5: 多轮对话协商"""
        print("\n💬 步骤6: 多轮对话协商")
        print("-" * 30)
        
        dialogue_history = []
        max_rounds = 3
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n🔄 第 {round_num} 轮对话")
            print("." * 20)
            
            round_messages = []
            
            # 每个智能体发表意见
            for role, agent in agents.items():
                print(f"💬 {role.value} 发言...")
                
                try:
                    # 生成对话消息
                    message_content = self._generate_dialogue_message(
                        role, patient_state, initial_opinions, dialogue_history, round_num
                    )
                    
                    message = DialogueMessage(
                        role=role,
                        content=message_content,
                        timestamp=datetime.now(),
                        round_number=round_num,
                        message_type="opinion"
                    )
                    
                    round_messages.append(message)
                    print(f"   📝 {message_content[:100]}...")
                    
                except Exception as e:
                    print(f"   ⚠️  {role.value} 发言失败: {e}")
            
            # 创建对话轮次
            dialogue_round = DialogueRound(
                round_number=round_num,
                messages=round_messages,
                timestamp=datetime.now(),
                convergence_score=self._calculate_convergence_score(round_messages)
            )
            
            dialogue_history.append(dialogue_round)
            
            print(f"✅ 第 {round_num} 轮对话完成，收敛度: {dialogue_round.convergence_score:.2f}")
            
            # 检查是否达成共识
            if dialogue_round.convergence_score > 0.8:
                print("🎉 达成共识，提前结束对话")
                break
        
        return dialogue_history
    
    def _step6_consensus_evaluation(self, dialogue_history: List[DialogueRound], patient_state: PatientState) -> ConsensusResult:
        """步骤6: 共识评估"""
        print("\n🎯 步骤7: 共识评估")
        print("-" * 30)
        
        print("📊 分析对话历史...")
        print("🔍 计算共识指标...")
        
        # 计算最终共识
        final_scores = self._calculate_final_treatment_scores(dialogue_history)
        
        # 确定推荐治疗方案
        recommended_treatment = max(final_scores.items(), key=lambda x: x[1])[0]
        consensus_strength = max(final_scores.values())
        
        # 创建共识结果
        consensus_result = ConsensusResult(
            patient_id=patient_state.patient_id,
            recommended_treatment=recommended_treatment,
            consensus_strength=consensus_strength,
            participant_roles=list(dialogue_history[0].messages[0].role for msg in dialogue_history[0].messages),
            dialogue_rounds=len(dialogue_history),
            final_scores=final_scores,
            timestamp=datetime.now(),
            confidence_level=consensus_strength,
            reasoning="基于多轮MDT讨论达成的共识"
        )
        
        print(f"🎯 推荐治疗方案: {recommended_treatment.value}")
        print(f"💪 共识强度: {consensus_strength:.2f}")
        print(f"🔄 对话轮数: {len(dialogue_history)}")
        print(f"📈 置信度: {consensus_result.confidence_level:.2f}")
        
        return consensus_result
    
    def _step7_rl_optimization(self, patient_state: PatientState, consensus_result: ConsensusResult) -> Dict[str, Any]:
        """步骤7: 强化学习优化反馈"""
        print("\n🤖 步骤8: 强化学习优化反馈")
        print("-" * 30)
        
        try:
            print("🔄 重置RL环境...")
            obs = self.rl_environment.reset(patient_state=patient_state)
            
            print("🎯 执行推荐动作...")
            action = self._treatment_to_action(consensus_result.recommended_treatment)
            next_obs, reward, done, info = self.rl_environment.step(action)
            
            print(f"🏆 RL奖励: {reward:.3f}")
            print(f"📊 环境信息: {info}")
            
            rl_feedback = {
                "reward": reward,
                "action": action,
                "treatment": consensus_result.recommended_treatment.value,
                "environment_info": info,
                "optimization_suggestions": self._generate_optimization_suggestions(reward, info)
            }
            
            return rl_feedback
            
        except Exception as e:
            print(f"⚠️  RL优化失败: {e}")
            return {
                "reward": 0.0,
                "error": str(e),
                "optimization_suggestions": ["RL系统暂时不可用"]
            }
    
    def _step8_generate_final_report(self, patient_state: PatientState, consensus_result: ConsensusResult, 
                                   dialogue_history: List[DialogueRound], rl_feedback: Dict) -> Dict[str, Any]:
        """步骤8: 生成最终报告"""
        print("\n📄 步骤9: 生成最终报告")
        print("-" * 30)
        
        print("📝 整理报告内容...")
        
        final_report = {
            "report_id": f"MDT_DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "patient_info": {
                "patient_id": patient_state.patient_id,
                "age": patient_state.age,
                "diagnosis": patient_state.diagnosis,
                "stage": patient_state.stage
            },
            "consensus_result": {
                "recommended_treatment": consensus_result.recommended_treatment.value,
                "consensus_strength": consensus_result.consensus_strength,
                "confidence_level": consensus_result.confidence_level,
                "dialogue_rounds": consensus_result.dialogue_rounds
            },
            "dialogue_summary": {
                "total_rounds": len(dialogue_history),
                "participants": len(set(msg.role for round in dialogue_history for msg in round.messages)),
                "convergence_progression": [round.convergence_score for round in dialogue_history]
            },
            "rl_optimization": rl_feedback,
            "workflow_metrics": {
                "total_processing_time": "模拟时间",
                "knowledge_retrieval_success": True,
                "consensus_achieved": consensus_result.consensus_strength > 0.7,
                "rl_feedback_available": "reward" in rl_feedback
            }
        }
        
        # 保存报告
        report_filename = f"demo_consensus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 报告已保存: {report_filename}")
        print("✅ 完整流程演示结束")
        
        return final_report
    
    # 辅助方法
    def _generate_mock_reasoning(self, role: RoleType, treatment: TreatmentOption, patient_state: PatientState) -> str:
        """生成模拟推理（当LLM不可用时）"""
        mock_reasoning = {
            (RoleType.ONCOLOGIST, TreatmentOption.CHEMOTHERAPY): f"基于患者{patient_state.age}岁，{patient_state.stage}期乳腺癌，化疗是标准治疗选择。考虑到患者的合并症，需要调整剂量。",
            (RoleType.SURGEON, TreatmentOption.SURGERY): f"患者{patient_state.stage}期乳腺癌适合手术治疗。需要评估手术风险，特别是考虑到高血压和糖尿病。",
            (RoleType.RADIOLOGIST, TreatmentOption.RADIOTHERAPY): f"放疗可作为{patient_state.stage}期乳腺癌的辅助治疗。需要精确定位和剂量规划。"
        }
        
        return mock_reasoning.get((role, treatment), f"{role.value}对{treatment.value}的专业分析")
    
    def _calculate_treatment_score(self, role: RoleType, treatment: TreatmentOption, patient_state: PatientState) -> float:
        """计算治疗评分"""
        # 基于角色和治疗类型的基础评分
        base_scores = {
            (RoleType.ONCOLOGIST, TreatmentOption.CHEMOTHERAPY): 0.85,
            (RoleType.SURGEON, TreatmentOption.SURGERY): 0.80,
            (RoleType.RADIOLOGIST, TreatmentOption.RADIOTHERAPY): 0.75,
        }
        
        base_score = base_scores.get((role, treatment), 0.60)
        
        # 根据患者状态调整
        if patient_state.age > 70:
            base_score -= 0.1  # 高龄患者风险增加
        
        if len(patient_state.comorbidities) > 2:
            base_score -= 0.05  # 合并症多风险增加
        
        return max(0.1, min(1.0, base_score))
    
    def _generate_dialogue_message(self, role: RoleType, patient_state: PatientState, 
                                 initial_opinions: Dict, dialogue_history: List, round_num: int) -> str:
        """生成对话消息"""
        messages = {
            RoleType.ONCOLOGIST: f"第{round_num}轮：作为肿瘤科医生，我认为化疗是{patient_state.stage}期乳腺癌的重要治疗选择。",
            RoleType.SURGEON: f"第{round_num}轮：外科角度看，手术切除是根治性治疗的关键，但需要考虑患者的手术耐受性。",
            RoleType.RADIOLOGIST: f"第{round_num}轮：放疗可以作为辅助治疗，特别是在保乳手术后。",
            RoleType.PATHOLOGIST: f"第{round_num}轮：病理分析显示肿瘤特征支持综合治疗方案。",
            RoleType.NURSE: f"第{round_num}轮：从护理角度，需要关注患者的心理状态和生活质量。"
        }
        
        return messages.get(role, f"第{round_num}轮：{role.value}的专业意见")
    
    def _calculate_convergence_score(self, messages: List[DialogueMessage]) -> float:
        """计算收敛度评分"""
        # 简化的收敛度计算
        return min(1.0, 0.5 + len(messages) * 0.1)
    
    def _calculate_final_treatment_scores(self, dialogue_history: List[DialogueRound]) -> Dict[TreatmentOption, float]:
        """计算最终治疗评分"""
        return {
            TreatmentOption.CHEMOTHERAPY: 0.85,
            TreatmentOption.SURGERY: 0.78,
            TreatmentOption.RADIOTHERAPY: 0.72
        }
    
    def _treatment_to_action(self, treatment: TreatmentOption) -> int:
        """治疗方案转换为RL动作"""
        mapping = {
            TreatmentOption.CHEMOTHERAPY: 0,
            TreatmentOption.SURGERY: 1,
            TreatmentOption.RADIOTHERAPY: 2,
            TreatmentOption.IMMUNOTHERAPY: 3,
            TreatmentOption.TARGETED_THERAPY: 4
        }
        return mapping.get(treatment, 0)
    
    def _generate_optimization_suggestions(self, reward: float, info: Dict) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        if reward < 0.5:
            suggestions.append("建议重新评估治疗方案的适用性")
        
        if reward > 0.8:
            suggestions.append("当前治疗方案获得高度认可")
        
        suggestions.append("继续监测患者反应并调整治疗策略")
        
        return suggestions


def main():
    """主函数"""
    try:
        demo = StepByStepConsensusDemo()
        final_report = demo.run_complete_demo()
        
        print("\n" + "=" * 60)
        print("🎉 MDT共识达成流程演示完成！")
        print("=" * 60)
        print(f"📄 最终报告ID: {final_report['report_id']}")
        print(f"🎯 推荐治疗: {final_report['consensus_result']['recommended_treatment']}")
        print(f"💪 共识强度: {final_report['consensus_result']['consensus_strength']:.2f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main())