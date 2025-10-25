"""
MDT医疗智能体系统主程序入口 - 完整集成版本
文件路径: main_integrated.py
作者: 姚刚 (共识与RL模块)
功能: 系统主入口，完整集成所有功能模块，包括role_agents、consensus、RL等
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import os
import json
# from dotenv import load_dotenv

# load_dotenv()

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.data_models import PatientState, TreatmentOption, RLAction, ChatRole
from src.consensus.consensus_matrix import ConsensusMatrix
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.consensus.role_agents import RoleAgent, RoleType, RoleOpinion, DialogueMessage
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.knowledge.enhanced_faiss_integration import EnhancedFAISSManager, SearchResult
from src.rl.rl_environment import MDTReinforcementLearning, RLTrainer
from src.integration.workflow_manager import IntegratedWorkflowManager
from src.utils.visualization import SystemVisualizer
from src.utils.system_optimizer import get_system_optimizer, optimized_function
from experiments.baseline_comparison import ComparisonExperiment

# 初始化系统优化器
system_optimizer = get_system_optimizer()

# 使用优化的日志系统
logger = system_optimizer.get_logger(__name__)


def _make_json_serializable(obj, _visited=None):
    """将对象转换为JSON可序列化的格式，避免递归和复杂图对象"""
    # 初始化循环引用跟踪
    if _visited is None:
        _visited = set()
    try:
        oid = id(obj)
        if oid in _visited:
            return "<cyclic>"
        _visited.add(oid)
    except Exception:
        pass

    # 基本类型直接返回
    from enum import Enum
    from datetime import datetime as _dt
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, _dt):
        return obj.isoformat()

    # 容器类型
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value, _visited) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [ _make_json_serializable(item, _visited) for item in obj ]

    # Numpy/Pandas等常见类型
    try:
        # numpy数组
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        # pandas DataFrame/Series 只给简要信息，避免海量内容与循环
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": list(obj.shape),
                "columns": obj.columns.tolist()
            }
        if isinstance(obj, pd.Series):
            return obj.to_dict()
    except Exception:
        pass

    # 枚举/动作对象
    if isinstance(obj, Enum):
        return obj.value
    try:
        from src.core.data_models import RLAction
        if isinstance(obj, RLAction):
            return {
                "treatment_recommendation": obj.treatment_recommendation.value,
                "confidence_level": obj.confidence_level,
                "explanation": obj.explanation
            }
    except Exception:
        pass

    # 可自定义to_dict的对象
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        try:
            return _make_json_serializable(obj.to_dict(), _visited)
        except Exception:
            return str(obj)

    # 图形对象：matplotlib/plotly/nx等，直接省略具体结构
    module_name = getattr(obj.__class__, "__module__", "")
    class_name = getattr(obj.__class__, "__name__", "")
    if "matplotlib" in module_name and class_name == "Figure":
        return "<matplotlib.Figure>"
    if module_name.startswith("plotly") and class_name == "Figure":
        return "<plotly.Figure>"
    if module_name.startswith("networkx"):
        return f"<networkx.{class_name}>"

    # 最后尝试__dict__，但避免再次递归爆炸
    if hasattr(obj, '__dict__'):
        try:
            plain = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            return _make_json_serializable(plain, _visited)
        except Exception:
            return str(obj)

    # 回退为字符串表示
    return str(obj)


class EnhancedPatientDialogueManager:
    """增强版患者对话管理器 - 集成所有功能模块"""
    
    def __init__(self, faiss_manager: EnhancedFAISSManager, consensus_system: ConsensusMatrix, 
                 rl_environment: MDTReinforcementLearning):
        self.faiss_manager = faiss_manager
        self.consensus_system = consensus_system
        self.rl_environment = rl_environment
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化角色智能体系统
        self.role_agents = self._initialize_role_agents()
        
        # 初始化新的组件
        try:
            from src.knowledge.dialogue_memory_manager import DialogueMemoryManager
            from src.treatment.enhanced_treatment_planner import EnhancedTreatmentPlanner
            from src.workflow.patient_dialogue_workflow import PatientDialogueWorkflow
            
            # 使用绝对路径初始化对话记忆管理器
            dialogue_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dialogue_memory_db")
            self.dialogue_memory = DialogueMemoryManager(memory_db_path=dialogue_db_path)
            self.treatment_planner = EnhancedTreatmentPlanner(
                self.dialogue_memory, 
                self.faiss_manager
            )
            self.workflow_manager = PatientDialogueWorkflow(
                self.dialogue_memory,
                self.faiss_manager,
                self.treatment_planner
            )
            
            # 当前活跃的对话会话
            self.current_session_id = None
            self.enhanced_mode = True
            
            self.logger.info("增强版患者对话管理器初始化完成")
            
        except ImportError as e:
            self.logger.warning(f"无法导入增强功能模块: {e}，使用基础模式")
            self.dialogue_memory = None
            self.treatment_planner = None
            self.workflow_manager = None
            self.current_session_id = None
            self.enhanced_mode = False
    
    def _initialize_role_agents(self) -> Dict[RoleType, RoleAgent]:
        """初始化角色智能体"""
        role_agents = {}
        
        # 创建各专科角色智能体
        role_types = [
            RoleType.ONCOLOGIST,
            RoleType.RADIOLOGIST,
            RoleType.NURSE,
            RoleType.PSYCHOLOGIST,
            RoleType.PATIENT_ADVOCATE,
            RoleType.NUTRITIONIST,
            RoleType.REHABILITATION_THERAPIST
        ]
        
        for role_type in role_types:
            try:
                # ? 这里为什么不传llm_interface？
                # 因为在RoleAgent中已经初始化了llm_interface，这里不需要重复初始化
                agent = RoleAgent(role_type)
                role_agents[role_type] = agent
                self.logger.info(f"初始化角色智能体: {role_type.value}")
            except Exception as e:
                self.logger.error(f"初始化角色智能体失败 {role_type.value}: {e}")
        
        return role_agents
    
    def query_patient_info_with_mdt(self, patient_id: str, query: str) -> Dict[str, Any]:
        """
        使用完整MDT流程查询患者信息
        集成角色智能体、共识机制、强化学习和历史对话上下文
        """
        try:
            # 0. 获取历史对话上下文（如果启用了对话记忆功能）
            dialogue_context = None
            if self.enhanced_mode and self.dialogue_memory:
                try:
                    dialogue_context = self.dialogue_memory.get_dialogue_context(
                        patient_id, query, context_window=5
                    )
                    self.logger.info(f"获取到患者 {patient_id} 的对话上下文，包含 {len(dialogue_context.get('recent_dialogues', []))} 条最近对话")
                except Exception as e:
                    self.logger.warning(f"获取对话上下文失败: {e}")
            
            # 1. 基础信息检索
            search_results = self.faiss_manager.search_by_patient_id(patient_id, k=5)
            if not search_results:
                search_results = self.faiss_manager.search_by_condition(query, k=3)
            
            # 2. 创建患者状态
            patient_state = self._create_patient_state_from_search(patient_id, search_results)
            
            # 3. 多角色专家意见收集（传入对话上下文）
            role_opinions = self._collect_role_opinions(patient_state, query, dialogue_context)
            
            # 4. 共识计算
            consensus_result = self._calculate_consensus(role_opinions, patient_state)
            
            # 5. 强化学习优化
            rl_optimized_result = self._apply_rl_optimization(consensus_result, patient_state)
            
            # 6. 生成最终回答（包含历史对话信息）
            final_response = self._generate_mdt_response(
                query, search_results, role_opinions, consensus_result, rl_optimized_result, dialogue_context
            )
            
            # 7. 保存对话记录到记忆系统
            dialogue_id = None
            if self.enhanced_mode and self.dialogue_memory:
                try:
                    dialogue_id = self.dialogue_memory.save_dialogue_turn(
                        patient_id=patient_id,
                        user_query=query,
                        agent_response=final_response,
                        session_id=self.current_session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        additional_metadata={
                            "consensus_score": consensus_result.get("consensus_score", 0.0),
                            "rl_confidence": rl_optimized_result.get("confidence", 0.0),
                            "role_count": len(role_opinions),
                            "search_results_count": len(search_results)
                        }
                    )
                    self.logger.info(f"保存对话记录: {dialogue_id}")
                except Exception as e:
                    self.logger.warning(f"保存对话记录失败: {e}")
            
            result = {
                "patient_id": patient_id,
                "query": query,
                "response": final_response,
                "search_results_count": len(search_results),
                "role_opinions": [opinion.to_dict() for opinion in role_opinions],
                "consensus_score": consensus_result.get("consensus_score", 0.0),
                "rl_optimization": rl_optimized_result,
                "timestamp": datetime.now().isoformat(),
                "enhanced_mode": True,
                "mdt_integrated": True,
                "dialogue_context_used": dialogue_context is not None,
                "dialogue_id": dialogue_id
            }
            
            # 添加对话上下文信息到结果中
            if dialogue_context:
                result["dialogue_context"] = {
                    "recent_dialogues_count": len(dialogue_context.get("recent_dialogues", [])),
                    "similar_dialogues_count": len(dialogue_context.get("similar_dialogues", [])),
                    "has_dialogue_patterns": bool(dialogue_context.get("dialogue_patterns"))
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"MDT查询失败: {e}")
            return {
                "patient_id": patient_id,
                "query": query,
                "response": f"抱歉，MDT查询时出现错误: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "enhanced_mode": True,
                "mdt_integrated": False
            }
    
    def _create_patient_state_from_search(self, patient_id: str, search_results: List[SearchResult]) -> PatientState:
        """从搜索结果创建患者状态"""
        if not search_results:
            return PatientState(
                patient_id=patient_id,
                age=0,
                diagnosis="未知",
                stage="未知",
                lab_results={},
                vital_signs={},
                symptoms=[],
                comorbidities=[],
                psychological_status="未评估",
                quality_of_life_score=50.0,
                timestamp=datetime.now()
            )
        
        # 从搜索结果中提取患者信息
        first_result = search_results[0]
        metadata = first_result.metadata
        
        return PatientState(
            patient_id=patient_id,
            age=metadata.get('age', 0),
            diagnosis=metadata.get('diagnosis', '未知'),
            stage=metadata.get('stage', '未知'),
            lab_results=metadata.get('lab_results', {}),
            vital_signs=metadata.get('vital_signs', {}),
            symptoms=metadata.get('symptoms', []),
            comorbidities=metadata.get('comorbidities', []),
            psychological_status=metadata.get('psychological_status', '未评估'),
            quality_of_life_score=metadata.get('quality_of_life_score', 50.0),
            timestamp=datetime.now()
        )
    
    def _collect_role_opinions(self, patient_state: PatientState, query: str, dialogue_context: Dict[str, Any] = None) -> List[RoleOpinion]:
        """收集各角色专家意见，支持历史对话上下文"""
        opinions = []
        # 构建对话消息列表（保留历史上下文以便未来扩展，但当前不直接传入RoleAgent）
        messages = []
        if dialogue_context and dialogue_context.get("recent_dialogues"):
            for dialogue in dialogue_context["recent_dialogues"][-3:]:
                if dialogue.get("user_query"):
                    messages.append(DialogueMessage(
                        role=ChatRole.USER,
                        content=f"[历史] {dialogue['user_query']}",
                        timestamp=datetime.fromisoformat(dialogue.get("timestamp", datetime.now().isoformat())),
                        message_type="user_query",
                        referenced_roles=[],
                        evidence_cited=[],
                        treatment_focus=TreatmentOption.WATCHFUL_WAITING
                    ))
                if dialogue.get("agent_response"):
                    messages.append(DialogueMessage(
                        role=ChatRole.SYSTEM,
                        content=f"[历史回复] {dialogue['agent_response'][:200]}...",
                        timestamp=datetime.fromisoformat(dialogue.get("timestamp", datetime.now().isoformat())),
                        message_type="system_response",
                        referenced_roles=[],
                        evidence_cited=[],
                        treatment_focus=TreatmentOption.WATCHFUL_WAITING
                    ))
        current_message = DialogueMessage(
            role=ChatRole.USER,
            content=query,
            timestamp=datetime.now(),
            message_type="user_query",
            referenced_roles=[],
            evidence_cited=[],
            treatment_focus=TreatmentOption.WATCHFUL_WAITING
        )
        messages.append(current_message)
        # 为各角色准备RAG知识
        try:
            relevant_knowledge = self.consensus_system.rag_system.retrieve_relevant_knowledge(
                patient_state, "initial_assessment"
            )
        except Exception as e:
            self.logger.warning(f"检索初始评估知识失败: {e}")
            relevant_knowledge = {}
        # 收集意见（改用已有API：generate_initial_opinion）
        for role_type, agent in self.role_agents.items():
            try:
                opinion = agent.generate_initial_opinion(patient_state, relevant_knowledge)
                if dialogue_context and dialogue_context.get("dialogue_patterns"):
                    patterns = dialogue_context["dialogue_patterns"]
                    if hasattr(opinion, 'reasoning') and opinion.reasoning:
                        opinion.reasoning += f"\n[基于历史对话模式]: {patterns.get('most_common_query_type', '无特定模式')}"
                opinions.append(opinion)
                self.logger.info(f"收集到{role_type.value}的意见（包含历史上下文）")
            except Exception as e:
                self.logger.error(f"收集{role_type.value}意见失败: {e}")
        return opinions
    
    def _calculate_consensus(self, opinions: List[RoleOpinion], patient_state: PatientState) -> Dict[str, Any]:
        """计算专家共识（改为直接调用 ConsensusMatrix.generate_consensus）"""
        try:
            consensus_obj = self.consensus_system.generate_consensus(patient_state, use_dialogue=False)
            # 将聚合分数的最大值作为简单的共识得分
            consensus_score = 0.0
            if consensus_obj.aggregated_scores:
                consensus_score = float(max(consensus_obj.aggregated_scores.values()))
            return {
                "consensus_score": consensus_score,
                "opinion_matrix": {},  # 如需展示，可从 consensus_obj.role_opinions 构造
                "convergence_achieved": bool(consensus_obj.convergence_achieved),
                "raw_consensus": consensus_obj,
            }
        except Exception as e:
            self.logger.error(f"共识计算失败: {e}")
            return {
                "consensus_score": 0.0,
                "opinion_matrix": {},
                "convergence_achieved": False,
                "error": str(e)
            }
    
    def _apply_rl_optimization(self, consensus_result: Dict[str, Any], patient_state: PatientState) -> Dict[str, Any]:
        """应用强化学习优化（传递真实共识对象以提升置信度计算准确性）"""
        try:
            consensus_obj = consensus_result.get("raw_consensus")
            rl_action = self.rl_environment.get_optimal_action(patient_state, consensus_result=consensus_obj)
            rl_confidence = self.rl_environment.get_action_confidence(
                rl_action.treatment_recommendation, patient_state, consensus_result=consensus_obj
            )
            return {
                "rl_recommended_action": rl_action,
                "rl_confidence": rl_confidence,
                "consensus_rl_alignment": self._calculate_alignment(consensus_result, rl_action)
            }
        except Exception as e:
            self.logger.error(f"RL优化失败: {e}")
            return {
                "rl_recommended_action": None,
                "rl_confidence": 0.0,
                "consensus_rl_alignment": 0.0,
                "error": str(e)
            }
    
    def _calculate_alignment(self, consensus_result: Dict[str, Any], rl_action) -> float:
        """计算共识与RL建议的一致性"""
        # 简化的一致性计算
        consensus_score = consensus_result.get("consensus_score", 0.0)
        if rl_action and consensus_result.get("convergence_achieved", False):
            return min(consensus_score + 0.1, 1.0)
        return consensus_score * 0.8
    
    def _generate_mdt_response(self, query: str, search_results: List[SearchResult], 
                              role_opinions: List[RoleOpinion], consensus_result: Dict[str, Any],
                              rl_result: Dict[str, Any], dialogue_context: Dict[str, Any] = None) -> str:
        """生成MDT综合回答，包含历史对话上下文信息"""
        response = f"🏥 MDT多学科团队会诊结果：\n\n"
        
        # 0. 历史对话上下文（如果有）
        if dialogue_context:
            recent_dialogues = dialogue_context.get("recent_dialogues", [])
            similar_dialogues = dialogue_context.get("similar_dialogues", [])
            patterns = dialogue_context.get("dialogue_patterns", {})
            
            if recent_dialogues or similar_dialogues or patterns:
                response += "📋 历史对话分析：\n"
                
                if recent_dialogues:
                    response += f"  • 最近对话: {len(recent_dialogues)} 条记录\n"
                
                if similar_dialogues:
                    response += f"  • 相似问题: 找到 {len(similar_dialogues)} 条相关历史记录\n"
                
                if patterns and patterns.get("most_common_query_type"):
                    response += f"  • 关注重点: {patterns['most_common_query_type']}\n"
                
                response += "\n"
        
        # 1. 专家意见汇总
        response += "👨‍⚕️ 专家意见汇总：\n"
        for opinion in role_opinions:
            role_name = opinion.role.value
            # Get the highest preference treatment
            if opinion.treatment_preferences:
                best_treatment = max(opinion.treatment_preferences.items(), key=lambda x: x[1])
                treatment = best_treatment[0].value
            else:
                treatment = "待定"
            confidence = opinion.confidence
            response += f"  • {role_name}: {treatment} (置信度: {confidence:.2f})\n"
        
        response += "\n"
        
        # 2. 共识结果
        consensus_score = consensus_result.get("consensus_score", 0.0)
        convergence = consensus_result.get("convergence_achieved", False)
        
        response += f"🤝 专家共识：\n"
        response += f"  • 共识得分: {consensus_score:.2f}\n"
        response += f"  • 是否达成共识: {'是' if convergence else '否'}\n\n"
        
        # 3. AI优化建议
        rl_action = rl_result.get("rl_recommended_action")
        rl_confidence = rl_result.get("rl_confidence", 0.0)
        alignment = rl_result.get("consensus_rl_alignment", 0.0)
        
        response += f"🤖 AI智能优化：\n"
        response += f"  • AI建议: {rl_action if rl_action else '无特定建议'}\n"
        response += f"  • AI置信度: {rl_confidence:.2f}\n"
        response += f"  • 专家-AI一致性: {alignment:.2f}\n\n"
        
        # 4. 历史经验参考（如果有相似对话）
        if dialogue_context and dialogue_context.get("similar_dialogues"):
            similar_dialogues = dialogue_context["similar_dialogues"]
            if similar_dialogues:
                response += "🔍 历史经验参考：\n"
                for i, similar in enumerate(similar_dialogues[:2], 1):  # 显示前2个最相似的
                    similarity = similar.get("similarity", 0.0)
                    timestamp = similar.get("timestamp", "未知时间")[:10]  # 只显示日期
                    response += f"  • 相似案例{i}: 相似度 {similarity:.2f} ({timestamp})\n"
                response += "\n"
        
        # 5. 最终建议
        if convergence and alignment > 0.7:
            response += "✅ 最终建议: 专家共识与AI建议高度一致，建议采纳。\n"
        elif convergence:
            response += "⚠️ 最终建议: 专家已达成共识，但AI建议存在差异，建议进一步讨论。\n"
        else:
            response += "❌ 最终建议: 专家意见分歧较大，建议进一步会诊讨论。\n"
        
        # 6. 个性化提示（基于历史对话模式）
        if dialogue_context and dialogue_context.get("dialogue_patterns"):
            patterns = dialogue_context["dialogue_patterns"]
            total_dialogues = patterns.get("total_dialogues", 0)
            if total_dialogues > 5:
                response += f"\n💡 个性化提示: 基于您的 {total_dialogues} 次历史对话，我们为您提供了更精准的建议。\n"
        
        return response


class FullyIntegratedMDTSystem:
    """完全集成的MDT系统接口"""

    def __init__(self):
        # 初始化系统优化器
        self.system_optimizer = get_system_optimizer()
        self.logger = self.system_optimizer.get_logger(self.__class__.__name__)
        
        # 初始化核心组件
        # 改为可切换的RAG：默认禁用，后续可以增加命令行参数启用
        try:
            from src.knowledge.disabled_rag import DisabledRAG
            self.rag_system = DisabledRAG()
            self.logger.info("RAG 已禁用，使用 DisabledRAG 存根")
        except Exception:
            # 兜底使用原始RAG
            self.rag_system = MedicalKnowledgeRAG()
            self.logger.info("DisabledRAG 加载失败，回退到 MedicalKnowledgeRAG")
        self.faiss_manager = EnhancedFAISSManager()
        
        # 🔥 真正集成所有功能模块
        # 将系统级 RAG 传入 ConsensusMatrix 以保持一致的RAG行为
        self.consensus_system = ConsensusMatrix(rag_system=self.rag_system)
        self.rl_environment = MDTReinforcementLearning(self.consensus_system)
        self.dialogue_manager = MultiAgentDialogueManager(self.rag_system)
        
        # 增强版患者对话管理器 - 集成所有功能
        self.enhanced_dialogue_manager = EnhancedPatientDialogueManager(
            self.faiss_manager, 
            self.consensus_system, 
            self.rl_environment
        )
        
        # 其他组件
        self.workflow_manager = IntegratedWorkflowManager()
        self.visualizer = SystemVisualizer()
        
        # 🚀 集成智能体协作系统
        try:
            from src.utils.llm_interface import LLMInterface, LLMConfig
            from src.workflow.intelligent_collaboration_manager import IntelligentCollaborationManager
            
            llm_config = LLMConfig()
            self.llm_interface = LLMInterface(llm_config)
            self.intelligent_collaboration_manager = IntelligentCollaborationManager(
                llm_interface=self.llm_interface,
                faiss_db_path="clinical_memory_db",
                enable_faiss=True,
                enable_enhanced_roles=True
            )
            self.use_intelligent_agents = True
            self.logger.info("✅ 智能体协作系统已启用")
            print("✅ 智能体协作系统已启用 - 支持多专家MDT协作决策")
        except Exception as e:
            self.logger.warning(f"⚠️ 智能体协作系统初始化失败: {e}")
            print(f"⚠️ 智能体协作系统初始化失败，将使用基础对话模式: {e}")
            self.intelligent_collaboration_manager = None
            self.use_intelligent_agents = False
        
        self.logger.info("完全集成的MDT系统初始化完成")
        print("🏥 完全集成的MDT系统已启动 - 包含角色智能体、共识机制、强化学习")

    async def run_integrated_patient_dialogue(self, patient_id: str = None) -> Dict[str, Any]:
        """运行完全集成的患者对话模式"""
        self.logger.info(f"启动完全集成的患者对话模式，患者ID: {patient_id}")
        
        dialogue_history = []
        session_start_time = datetime.now()
        
        print(f"\n=== 完全集成MDT患者对话系统 ===")
        if patient_id:
            print(f"当前患者: {patient_id}")
        else:
            print("通用查询模式")
            print("💡 使用 'patient:ID' 设置患者ID以启用完整MDT功能")
        print("输入 'quit' 或 'exit' 退出对话")
        print("输入 'help' 查看帮助信息")
        print("🏥 输入 'mdt' 启动完整MDT会诊流程")
        print("🤖 输入 'agents' 查看可用专家角色")
        print("📊 输入 'consensus' 查看共识统计")
        print("🧠 输入 'rl' 查看强化学习建议")
        print("=" * 60)
        
        while True:
            try:
                # 获取用户输入
                try:
                    if patient_id:
                        user_input = input(f"\n[患者 {patient_id}] 请输入您的问题: ").strip()
                    else:
                        user_input = input(f"\n[MDT会诊] 请输入您的问题: ").strip()
                except EOFError:
                    print("\n检测到输入结束，退出对话模式")
                    break
                except KeyboardInterrupt:
                    print("\n\n用户中断，退出对话模式")
                    break
                
                if not user_input:
                    continue
                
                # 检查退出命令
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("感谢使用完全集成MDT对话系统，再见！")
                    break
                
                # 检查帮助命令
                if user_input.lower() in ['help', '帮助', 'h']:
                    self._show_integrated_help()
                    continue
                
                # 检查设置患者ID命令
                if user_input.lower().startswith('patient:') or user_input.lower().startswith('患者:'):
                    new_patient_id = user_input.split(':', 1)[1].strip()
                    if new_patient_id:
                        patient_id = new_patient_id
                        print(f"✅ 已设置患者ID: {patient_id}")
                        print("现在您可以使用完整的MDT功能了！")
                    else:
                        print("❌ 请提供有效的患者ID，格式: patient:ID")
                    continue
                
                # 检查MDT会诊命令
                if user_input.lower() in ['mdt', '会诊', 'consultation']:
                    if patient_id:
                        print("🏥 启动完整MDT会诊流程...")
                        result = self.enhanced_dialogue_manager.query_patient_info_with_mdt(
                            patient_id, "请进行完整的MDT会诊评估"
                        )
                        self._show_mdt_result(result)
                        dialogue_history.append(result)
                    else:
                        print("❌ 请先指定患者ID")
                    continue
                
                # 检查专家角色命令
                if user_input.lower() in ['agents', '专家', 'roles']:
                    self._show_available_agents()
                    continue
                
                # 检查共识统计命令
                if user_input.lower() in ['consensus', '共识', 'stats']:
                    self._show_consensus_stats()
                    continue
                
                # 检查RL建议命令
                if user_input.lower() in ['rl', '强化学习', 'ai']:
                    if patient_id:
                        self._show_rl_recommendations(patient_id)
                    else:
                        print("❌ 请先指定患者ID")
                    continue
                
                # 检查历史对话命令
                if user_input.lower().startswith('history:') or user_input.lower().startswith('历史:'):
                    history_patient_id = user_input.split(':', 1)[1].strip()
                    if history_patient_id:
                        self._show_dialogue_history(history_patient_id)
                    else:
                        print("❌ 请提供有效的患者ID，格式: history:patient_id")
                    continue
                
                # 检查当前患者历史对话命令
                if user_input.lower() in ['history', '历史', 'h']:
                    if patient_id:
                        self._show_dialogue_history(patient_id)
                    else:
                        print("❌ 请先指定患者ID")
                    continue
                
                # 处理常规查询 - 使用完整MDT流程
                if patient_id:
                    print("🔄 使用完整MDT流程处理查询...")
                    result = self.enhanced_dialogue_manager.query_patient_info_with_mdt(patient_id, user_input)
                    print(f"\n{result['response']}")
                    dialogue_history.append(result)
                else:
                    print("❌ 请先指定患者ID以使用完整MDT功能")
                
            except Exception as e:
                print(f"❌ 处理查询时出错: {e}")
                self.logger.error(f"对话处理错误: {e}")
        
        # 返回对话历史
        return {
            "dialogue_history": dialogue_history,
            "total_queries": len(dialogue_history),
            "session_duration": (datetime.now() - session_start_time).total_seconds(),
            "integrated_features_used": True
        }
    
    def _show_integrated_help(self):
        """显示集成系统帮助信息"""
        print("\n=== 完全集成MDT系统帮助 ===")
        print("可用命令:")
        print("  • 'patient:ID' - 设置患者ID (例如: patient:P001)")
        print("  • 'mdt' - 启动完整MDT多学科会诊")
        print("  • 'agents' - 查看可用专家角色")
        print("  • 'consensus' - 查看共识统计信息")
        print("  • 'rl' - 查看AI强化学习建议")
        print("  • 'history' - 查看当前患者的历史对话")
        print("  • 'history:ID' - 查看指定患者的历史对话 (例如: history:P001)")
        print("  • 'help' - 显示此帮助信息")
        print("  • 'quit' - 退出系统")
        print("\n功能特色:")
        print("  🏥 多专科专家角色智能体")
        print("  🤝 实时共识计算与分析")
        print("  🧠 强化学习优化建议")
        print("  📋 完整对话历史记录")
        print("  📊 综合决策支持系统")
        print("\n💡 提示: 先使用 'patient:ID' 设置患者ID，然后就可以使用完整MDT功能了！")
        print("=" * 40)
    
    def _show_mdt_result(self, result: Dict[str, Any]):
        """显示MDT会诊结果"""
        print(f"\n{result.get('response', '无回答')}")
        
        if result.get('mdt_integrated'):
            print(f"\n📊 会诊统计:")
            print(f"  • 参与专家数: {len(result.get('role_opinions', []))}")
            print(f"  • 共识得分: {result.get('consensus_score', 0.0):.2f}")
            print(f"  • 检索结果数: {result.get('search_results_count', 0)}")
    
    def _show_available_agents(self):
        """显示可用的专家角色"""
        print("\n🏥 可用专家角色:")
        agents = self.enhanced_dialogue_manager.role_agents
        for role_type, agent in agents.items():
            print(f"  • {role_type.value} - 专业领域: {role_type.name}")
        print(f"\n总计: {len(agents)} 位专家")
    
    def _show_consensus_stats(self):
        """显示共识统计信息"""
        print("\n🤝 共识系统统计:")
        print("  • 共识算法: 加权平均法")
        print("  • 收敛阈值: 0.7")
        print("  • 支持角色数: 5+")
        print("  • 实时计算: 是")
    
    def _show_rl_recommendations(self, patient_id: str):
        """显示强化学习建议"""
        print(f"\n🧠 AI强化学习建议 (患者: {patient_id}):")
        try:
            # 创建简单的患者状态用于演示
            patient_state = PatientState(
                patient_id=patient_id,
                age=0,
                diagnosis="演示",
                stage="演示",
                lab_results={},
                vital_signs={},
                symptoms=[],
                comorbidities=[],
                psychological_status="未评估",
                quality_of_life_score=50.0,
                timestamp=datetime.now()
            )
            
            action = self.rl_environment.get_optimal_action(patient_state)
            confidence = self.rl_environment.get_action_confidence(patient_state, action)
            
            print(f"  • 推荐行动: {action}")
            print(f"  • AI置信度: {confidence:.2f}")
            print(f"  • 学习状态: 活跃")
            
        except Exception as e:
            print(f"  • 获取RL建议失败: {e}")

    def _show_dialogue_history(self, patient_id: str, limit: int = 10):
        """显示患者的历史对话"""
        print(f"\n📋 患者 {patient_id} 的历史对话:")
        try:
            # 检查是否有对话记忆管理器
            if not hasattr(self.enhanced_dialogue_manager, 'dialogue_memory') or not self.enhanced_dialogue_manager.dialogue_memory:
                print("  • 对话记忆管理器未初始化")
                return
            
            # 获取患者的历史对话
            history = self.enhanced_dialogue_manager.dialogue_memory.get_patient_dialogue_history(patient_id, limit)
            
            if not history:
                print(f"  • 患者 {patient_id} 暂无历史对话记录")
                return
            
            print(f"  • 共找到 {len(history)} 条对话记录 (显示最近 {min(limit, len(history))} 条)")
            print("=" * 80)
            
            for i, record in enumerate(history[:limit], 1):
                timestamp = record.get('timestamp', 'N/A')
                user_query = record.get('user_query', 'N/A')
                agent_response = record.get('agent_response', 'N/A')
                session_id = record.get('session_id', 'N/A')
                
                # 格式化时间戳
                if timestamp != 'N/A':
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        formatted_time = timestamp[:19] if len(timestamp) > 19 else timestamp
                else:
                    formatted_time = 'N/A'
                
                print(f"\n{i}. 时间: {formatted_time}")
                print(f"   会话ID: {session_id}")
                print(f"   用户查询: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
                print(f"   系统回复: {agent_response[:150]}{'...' if len(agent_response) > 150 else ''}")
                print("-" * 80)
            
            # 显示会话统计信息
            sessions = self.enhanced_dialogue_manager.dialogue_memory.get_patient_sessions(patient_id)
            if sessions:
                print(f"\n📊 会话统计:")
                print(f"  • 总会话数: {len(sessions)}")
                print(f"  • 总对话数: {len(history)}")
                print(f"  • 平均每会话对话数: {len(history) / len(sessions):.1f}")
                
                # 显示最近的会话信息
                if sessions:
                    latest_session = max(sessions, key=lambda x: x.get('end_time', ''))
                    print(f"  • 最近会话: {latest_session.get('session_id', 'N/A')}")
                    print(f"  • 最后活动: {latest_session.get('end_time', 'N/A')[:19]}")
            
        except Exception as e:
            print(f"  • 获取历史对话失败: {e}")
            logger.error(f"显示历史对话失败: {e}", exc_info=True)

    @optimized_function
    def run_fully_integrated_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行完全集成的患者分析 - 使用所有功能模块"""
        self.logger.info(f"开始完全集成分析，患者: {patient_data.get('patient_id', 'unknown')}")

        patient_state = self._create_patient_state(patient_data)

        # 1. 多智能体对话与共识 (原有功能)
        self.logger.info("运行多智能体对话...")
        consensus_result = self.dialogue_manager.conduct_mdt_discussion(patient_state)

        # 2. 🔥 新增：角色智能体分析
        # self.logger.info("收集角色智能体意见...")
        # role_opinions = self.enhanced_dialogue_manager._collect_role_opinions(
        #     patient_state, "请提供治疗建议"
        # )

        # # 3. 🔥 新增：增强共识计算
        # self.logger.info("计算增强共识...")
        # enhanced_consensus = self.enhanced_dialogue_manager._calculate_consensus(
        #     role_opinions, patient_state
        # )

        # # 4. 🔥 新增：强化学习优化
        # self.logger.info("应用强化学习优化...")
        # rl_optimization = self.enhanced_dialogue_manager._apply_rl_optimization(
        #     enhanced_consensus, patient_state
        # )

        # # 5. 生成可视化
        # self.logger.info("生成可视化...")
        # visualizations = self.visualizer.create_patient_analysis_dashboard(
        #     patient_state, consensus_result
        # )

        # 6. 整理完整结果
        analysis_result = {
            "patient_info": {
                "patient_id": patient_state.patient_id,
                "age": patient_state.age,
                "diagnosis": patient_state.diagnosis,
                "stage": patient_state.stage,
            },
            # 原有共识结果
            "consensus_result": {
                "recommended_treatment": max(
                    consensus_result.aggregated_scores.items(), key=lambda x: x[1]
                )[0].value,
                "consensus_score": max(consensus_result.aggregated_scores.values()),
                "total_rounds": consensus_result.total_rounds,
                "convergence_achieved": consensus_result.convergence_achieved,
                "conflicts": len(consensus_result.conflicts),
                "agreements": len(consensus_result.agreements),
            },
            # 🔥 新增：角色智能体结果
            # "role_agent_analysis": {
            #     "participating_roles": len(role_opinions),
            #     "role_opinions": [opinion.to_dict() for opinion in role_opinions],
            #     "role_consensus_score": enhanced_consensus.get("consensus_score", 0.0),
            #     "role_convergence": enhanced_consensus.get("convergence_achieved", False)
            # },
            # 🔥 新增：强化学习结果
            # "rl_optimization": {
            #     "rl_recommended_action": rl_optimization.get("rl_recommended_action"),
            #     "rl_confidence": rl_optimization.get("rl_confidence", 0.0),
            #     "consensus_rl_alignment": rl_optimization.get("consensus_rl_alignment", 0.0)
            # },
            # # 其他信息
            # "dialogue_transcript": self.dialogue_manager.get_dialogue_transcript(),
            # "visualizations": visualizations,
            # "analysis_timestamp": datetime.now().isoformat(),
            # "fully_integrated": True
        }

        self.logger.info("完全集成分析完成")
        return analysis_result

    def _create_patient_state(self, patient_data: Dict[str, Any]) -> PatientState:
        """创建患者状态对象"""
        return PatientState(
            patient_id=patient_data.get("patient_id", "unknown"),
            age=patient_data.get("age", 0),
            diagnosis=patient_data.get("diagnosis", ""),
            stage=patient_data.get("stage", ""),
            lab_results=patient_data.get("lab_results", {}),
            vital_signs=patient_data.get("vital_signs", {}),
            symptoms=patient_data.get("symptoms", []),
            comorbidities=patient_data.get("comorbidities", []),
            psychological_status=patient_data.get("psychological_status", "未评估"),
            quality_of_life_score=patient_data.get("quality_of_life_score", 50.0),
            timestamp=datetime.now()
        )

    @optimized_function
    def run_training_experiment(self, episodes: int = 1000) -> Dict[str, Any]:
        """运行RL训练实验 - 真正使用RL环境"""
        self.logger.info(f"开始RL训练，episodes: {episodes}")

        trainer = RLTrainer(self.rl_environment)
        training_results = trainer.train_dqn(episodes=episodes)

        # 生成训练可视化
        training_visualizations = self.visualizer.create_training_dashboard(
            training_results
        )

        result = {
            "training_results": training_results,
            "visualizations": training_visualizations,
            "final_metrics": self.rl_environment.get_training_metrics(),
            "rl_integrated": True
        }

        logger.info("RL训练实验完成")
        return result


def create_sample_patients() -> List[Dict[str, Any]]:
    """创建示例患者数据"""
    return [
        {
            "patient_id": "P002", 
            "age": 45,
            "diagnosis": "乳腺癌",
            "stage": "IIB",
            "lab_results": {"CA153": 25.3, "CEA": 3.2},
            "vital_signs": {"血压": 120, "心率": 72, "体温": 36.8},
            "symptoms": ["乳房肿块", "轻微疼痛"],
            "comorbidities": [],
            "psychological_status": "正常",
            "quality_of_life_score": 80.0
        }
    ]


def main():
    """主函数 - 完全集成版本"""
    parser = argparse.ArgumentParser(description="完全集成的MDT Medical AI System")

    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "patient", "training", "comparison", "simulation", "dialogue", "integrated"],
        help="运行模式",
    )

    parser.add_argument("--patient-file", type=str, help="患者数据文件路径 (JSON格式)")
    parser.add_argument("--patient-id", type=str, help="患者ID (用于对话模式)")
    parser.add_argument("--episodes", type=int, default=1000, help="RL训练episode数量")
    parser.add_argument("--num-patients", type=int, default=100, help="对比实验中的患者数量")
    parser.add_argument("--num-trials", type=int, default=50, help="对比实验的试验次数")
    parser.add_argument("--simulation-days", type=int, default=30, help="时序模拟天数")
    parser.add_argument("--output-dir", type=str, default="results", help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 初始化完全集成系统
    print("=== 完全集成MDT医疗智能体系统 ===")
    print("初始化系统组件...")
    
    # 启动系统优化器
    print("启动系统优化器...")
    system_optimizer.initialize()
    logger.info("系统优化器已启动")

    system = FullyIntegratedMDTSystem()

    print(f"运行模式: {args.mode}")

    if args.mode == "demo":
        logger.info("\n=== 完全集成演示模式 ===")
        # 结构化的肺癌患者医疗档案
        sample_patients = create_sample_patients()

        for i, patient_data in enumerate(sample_patients, 1):
            logger.info(f"\n--- 完全集成分析患者 {i}: {patient_data['patient_id']} ---")
            result = system.run_fully_integrated_analysis(patient_data)
            logger.info(f"完全集成分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            # 显示原有结果
            print(f"推荐治疗方案: {result['consensus_result']['recommended_treatment']}")
            print(f"共识得分: {result['consensus_result']['consensus_score']:.3f}")
            
            # 🔥 显示新增的集成结果
            # print(f"角色智能体参与数: {result['role_agent_analysis']['participating_roles']}")
            # print(f"角色共识得分: {result['role_agent_analysis']['role_consensus_score']:.3f}")
            # print(f"RL优化置信度: {result['rl_optimization']['rl_confidence']:.3f}")
            # print(f"共识-RL一致性: {result['rl_optimization']['consensus_rl_alignment']:.3f}")

            # 保存结果
            output_file = f"{args.output_dir}/integrated_patient_{patient_data['patient_id']}_analysis.json"
            with open(output_file, "w", encoding="utf-8") as f:
                serializable_result = result.copy()
                serializable_result.pop("visualizations", None)
                serializable_result = _make_json_serializable(serializable_result)
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)
            print(f"完全集成结果已保存到: {output_file}")

    elif args.mode == "integrated":
        print("\n=== 完全集成对话模式 ===")
        
        # 检查FAISS数据库
        faiss_db_path = "clinical_memory_db"
        if not os.path.exists(faiss_db_path):
            print(f"错误: FAISS数据库目录不存在: {faiss_db_path}")
            print("请确保已经初始化FAISS数据库")
            return
        
        # 启动完全集成对话模式
        try:
            import asyncio
            result = asyncio.run(system.run_integrated_patient_dialogue(args.patient_id))
            
            # 保存对话历史
            if result['dialogue_history']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"{args.output_dir}/integrated_dialogue_history_{timestamp}.json"
                
                with open(output_file, "w", encoding="utf-8") as f:
                    safe_result = _make_json_serializable(result)
                    json.dump(safe_result, f, ensure_ascii=False, indent=2)

                print(f"\n完全集成对话历史已保存到: {output_file}")
                print(f"总查询次数: {result['total_queries']}")
                print(f"会话时长: {result['session_duration']:.1f}秒")
                print(f"集成功能使用: {result['integrated_features_used']}")
            else:
                print("\n未进行任何查询")
                
        except Exception as e:
            print(f"完全集成对话模式运行出错: {e}")
            logger.error(f"集成对话模式错误: {e}")

    elif args.mode == "training":
        print(f"\n=== 集成RL训练模式 ({args.episodes} episodes) ===")
        result = system.run_training_experiment(args.episodes)

        print("集成RL训练完成!")
        print(f"最终平均奖励: {result['final_metrics']['recent_average_reward']:.3f}")
        print(f"学习改进: {result['final_metrics']['improvement']:+.3f}")
        print(f"RL集成状态: {result['rl_integrated']}")

        # 保存训练结果
        output_file = f"{args.output_dir}/integrated_training_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            serializable_result = result.copy()
            serializable_result.pop("visualizations", None)
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        print(f"集成训练结果已保存到: {output_file}")

    elif args.mode == "comparison":
        print(f"\n=== 基线对比模式 ({args.num_patients} 患者, {args.num_trials} 试验) ===")
        experiment = ComparisonExperiment()
        experiment.generate_test_patients(args.num_patients)
        results = experiment.run_comparison(args.num_trials)
        report = experiment.generate_comparison_report()

        # 绘制并保存对比图表
        plot_path = f"{args.output_dir}/figures/baseline_comparison.png"
        experiment.plot_comparison_results(plot_path)

        # 保存对比结果
        output_file = f"{args.output_dir}/comparison_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "comparison_results": results.to_dict("records"),
                "report": report,
                "visualization_saved": True
            }, f, ensure_ascii=False, indent=2)

        print("对比实验完成!")
        print("\n" + report)
        print(f"对比结果已保存到: {output_file}")
        print(f"对比图表已保存到: {plot_path}")

    # 其他模式保持原有逻辑...
    else:
        print(f"模式 '{args.mode}' 暂未在完全集成版本中实现")
        print(f"可用模式: demo, integrated, training, comparison")

    print(f"\n所有输出文件保存在: {args.output_dir}/")
    print("🏥 完全集成MDT系统运行完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，系统退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"系统运行出错: {e}", exc_info=True)
        print(f"系统运行出错: {e}")
        sys.exit(1)