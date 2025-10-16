"""
MDT医疗智能体系统主程序入口
文件路径: main.py
作者: Tianyu (系统集成) / 姚刚 (共识与RL模块)
功能: 系统主入口，提供命令行界面和演示功能
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.data_models import PatientState, TreatmentOption
from src.consensus.consensus_matrix import ConsensusMatrix
from src.consensus.dialogue_manager import MultiAgentDialogueManager
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


class PatientDialogueManager:
    """患者对话管理器 - 增强版本，集成记忆系统和治疗方案生成"""
    
    def __init__(self, faiss_manager: EnhancedFAISSManager):
        self.faiss_manager = faiss_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化新的组件
        try:
            from src.knowledge.dialogue_memory_manager import DialogueMemoryManager
            from src.treatment.enhanced_treatment_planner import EnhancedTreatmentPlanner
            from src.workflow.patient_dialogue_workflow import PatientDialogueWorkflow
            
            self.dialogue_memory = DialogueMemoryManager()
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
        
    def query_patient_info(self, patient_id: str, query: str) -> Dict[str, Any]:
        """查询患者信息并生成回答 - 增强版本，支持对话记忆和治疗方案生成"""
        try:
            # 如果启用了增强模式，使用新的工作流
            if self.enhanced_mode and self.workflow_manager:
                return self._query_with_enhanced_workflow(patient_id, query)
            else:
                return self._query_with_basic_workflow(patient_id, query)
                
        except Exception as e:
            self.logger.error(f"查询患者信息失败: {e}")
            return {
                "patient_id": patient_id,
                "query": query,
                "response": f"抱歉，查询患者信息时出现错误: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "enhanced_mode": self.enhanced_mode
            }
    
    def _query_with_enhanced_workflow(self, patient_id: str, query: str) -> Dict[str, Any]:
        """使用增强工作流处理查询"""
        # 如果没有活跃会话，创建新会话
        if not self.current_session_id:
            self.current_session_id, welcome_msg = self.workflow_manager.start_dialogue_session(
                patient_id=patient_id,
                session_type="consultation"
            )
            self.logger.info(f"创建新对话会话: {self.current_session_id}")
        
        # 处理对话轮次
        agent_response, turn_data = self.workflow_manager.process_dialogue_turn(
            session_id=self.current_session_id,
            user_input=query,
            include_treatment_planning=True
        )
        
        # 构建搜索查询（保持兼容性）
        search_results = self.faiss_manager.search_by_patient_id(patient_id, k=5)
        if not search_results:
            search_results = self.faiss_manager.search_by_condition(query, k=3)
        
        # 准备返回数据
        return {
            "patient_id": patient_id,
            "query": query,
            "response": agent_response,
            "search_results_count": len(search_results),
            "timestamp": datetime.now().isoformat(),
            # 新增的增强功能数据
            "session_id": self.current_session_id,
            "turn_id": turn_data.get("turn_id"),
            "dialogue_id": turn_data.get("dialogue_id"),
            "response_type": turn_data.get("response_type"),
            "confidence_score": turn_data.get("confidence_score"),
            "processing_time": turn_data.get("processing_time"),
            "treatment_plan_id": turn_data.get("treatment_plan_id"),
            "session_info": turn_data.get("session_info"),
            "enhanced_mode": True
        }
    
    def _query_with_basic_workflow(self, patient_id: str, query: str) -> Dict[str, Any]:
        """使用基础工作流处理查询（向后兼容）"""
        # 从FAISS数据库搜索相关信息
        search_results = self.faiss_manager.search_by_patient_id(patient_id, k=5)
        
        if not search_results:
            # 如果没有找到患者信息，尝试通过条件搜索
            search_results = self.faiss_manager.search_by_condition(query, k=3)
        
        # 生成智能回答
        response = self._generate_response(query, search_results, patient_id)
        
        return {
            "patient_id": patient_id,
            "query": query,
            "response": response,
            "search_results_count": len(search_results),
            "timestamp": datetime.now().isoformat(),
            "enhanced_mode": False
        }
    
    def _generate_response(self, query: str, search_results: List[SearchResult], patient_id: str) -> str:
        """基于搜索结果生成智能回答"""
        if not search_results:
            return f"抱歉，没有找到患者 {patient_id} 的相关信息。请检查患者ID是否正确。"
        
        # 分析查询类型
        query_lower = query.lower()
        
        if "诊断" in query or "diagnosis" in query_lower:
            return self._generate_diagnosis_response(search_results, patient_id)
        elif "治疗" in query or "treatment" in query_lower:
            return self._generate_treatment_response(search_results, patient_id)
        elif "药物" in query or "medication" in query_lower or "drug" in query_lower:
            return self._generate_medication_response(search_results, patient_id)
        elif "检查" in query or "lab" in query_lower or "test" in query_lower:
            return self._generate_lab_response(search_results, patient_id)
        elif "病史" in query or "history" in query_lower:
            return self._generate_history_response(search_results, patient_id)
        else:
            return self._generate_general_response(search_results, patient_id, query)
    
    def _generate_diagnosis_response(self, search_results: List[SearchResult], patient_id: str) -> str:
        """生成诊断相关回答"""
        response = f"患者 {patient_id} 的诊断信息：\n\n"
        
        for i, result in enumerate(search_results[:3], 1):
            metadata = result.metadata
            content = result.content
            
            if 'diagnosis' in metadata:
                response += f"{i}. 主要诊断: {metadata['diagnosis']}\n"
            
            if 'stage' in metadata and metadata['stage']:
                response += f"   分期: {metadata['stage']}\n"
            
            if 'comorbidities' in metadata and metadata['comorbidities']:
                response += f"   合并症: {', '.join(metadata['comorbidities'])}\n"
            
            response += "\n"
        
        return response.strip()
    
    def _generate_treatment_response(self, search_results: List[SearchResult], patient_id: str) -> str:
        """生成治疗相关回答"""
        response = f"患者 {patient_id} 的治疗信息：\n\n"
        
        for i, result in enumerate(search_results[:3], 1):
            content = result.content
            
            # 从内容中提取治疗相关信息
            if "治疗" in content or "手术" in content or "化疗" in content:
                lines = content.split('\n')
                treatment_lines = [line for line in lines if any(keyword in line for keyword in ["治疗", "手术", "化疗", "放疗", "药物"])]
                
                if treatment_lines:
                    response += f"{i}. 治疗方案:\n"
                    for line in treatment_lines[:3]:
                        response += f"   - {line.strip()}\n"
                    response += "\n"
        
        return response.strip() if response.strip() != f"患者 {patient_id} 的治疗信息：" else f"暂未找到患者 {patient_id} 的具体治疗信息。"
    
    def _generate_medication_response(self, search_results: List[SearchResult], patient_id: str) -> str:
        """生成药物相关回答"""
        response = f"患者 {patient_id} 的用药信息：\n\n"
        
        for i, result in enumerate(search_results[:3], 1):
            content = result.content
            
            # 从内容中提取药物信息
            if "药物" in content or "medication" in content.lower():
                lines = content.split('\n')
                med_lines = [line for line in lines if any(keyword in line.lower() for keyword in ["药物", "medication", "drug", "剂量", "dose"])]
                
                if med_lines:
                    response += f"{i}. 用药记录:\n"
                    for line in med_lines[:5]:
                        response += f"   - {line.strip()}\n"
                    response += "\n"
        
        return response.strip() if response.strip() != f"患者 {patient_id} 的用药信息：" else f"暂未找到患者 {patient_id} 的具体用药信息。"
    
    def _generate_lab_response(self, search_results: List[SearchResult], patient_id: str) -> str:
        """生成检查结果相关回答"""
        response = f"患者 {patient_id} 的检查结果：\n\n"
        
        for i, result in enumerate(search_results[:3], 1):
            content = result.content
            
            # 从内容中提取检查信息
            if "检查" in content or "lab" in content.lower() or "结果" in content:
                lines = content.split('\n')
                lab_lines = [line for line in lines if any(keyword in line for keyword in ["检查", "结果", "指标", "数值"])]
                
                if lab_lines:
                    response += f"{i}. 检查记录:\n"
                    for line in lab_lines[:5]:
                        response += f"   - {line.strip()}\n"
                    response += "\n"
        
        return response.strip() if response.strip() != f"患者 {patient_id} 的检查结果：" else f"暂未找到患者 {patient_id} 的具体检查信息。"
    
    def _generate_history_response(self, search_results: List[SearchResult], patient_id: str) -> str:
        """生成病史相关回答"""
        response = f"患者 {patient_id} 的病史信息：\n\n"
        
        for i, result in enumerate(search_results[:3], 1):
            metadata = result.metadata
            content = result.content
            
            if 'age' in metadata:
                response += f"年龄: {metadata['age']}岁\n"
            
            if 'comorbidities' in metadata and metadata['comorbidities']:
                response += f"既往病史: {', '.join(metadata['comorbidities'])}\n"
            
            # 从内容中提取病史信息
            if "病史" in content or "history" in content.lower():
                lines = content.split('\n')
                history_lines = [line for line in lines if any(keyword in line for keyword in ["病史", "既往", "家族史"])]
                
                if history_lines:
                    response += f"详细病史:\n"
                    for line in history_lines[:3]:
                        response += f"   - {line.strip()}\n"
            
            response += "\n"
        
        return response.strip()
    
    def _generate_general_response(self, search_results: List[SearchResult], patient_id: str, query: str) -> str:
        """生成通用回答"""
        response = f"关于患者 {patient_id} 的 '{query}' 相关信息：\n\n"
        
        for i, result in enumerate(search_results[:3], 1):
            content = result.content
            metadata = result.metadata
            
            # 提取相关内容片段
            lines = content.split('\n')
            relevant_lines = []
            
            for line in lines:
                if any(keyword in line.lower() for keyword in query.lower().split()):
                    relevant_lines.append(line.strip())
            
            if relevant_lines:
                response += f"{i}. 相关信息:\n"
                for line in relevant_lines[:3]:
                    if line:
                        response += f"   - {line}\n"
                response += "\n"
            elif metadata:
                response += f"{i}. 基本信息:\n"
                if 'diagnosis' in metadata:
                    response += f"   - 诊断: {metadata['diagnosis']}\n"
                if 'age' in metadata:
                    response += f"   - 年龄: {metadata['age']}岁\n"
                response += "\n"
        
        return response.strip()
    
    def end_current_session(self, reason: str = "normal") -> Dict[str, Any]:
        """结束当前对话会话"""
        try:
            if not self.enhanced_mode or not self.workflow_manager:
                return {"message": "增强模式未启用"}
                
            if not self.current_session_id:
                return {"message": "没有活跃的对话会话"}
            
            session_summary = self.workflow_manager.end_dialogue_session(
                self.current_session_id, reason
            )
            
            self.current_session_id = None
            self.logger.info("对话会话已结束")
            
            return session_summary
            
        except Exception as e:
            self.logger.error(f"结束对话会话失败: {e}")
            return {"error": str(e)}
    
    def get_dialogue_history(self, patient_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取患者对话历史"""
        try:
            if not self.enhanced_mode or not self.dialogue_memory:
                return []
            return self.dialogue_memory.get_patient_dialogue_history(patient_id, limit)
        except Exception as e:
            self.logger.error(f"获取对话历史失败: {e}")
            return []
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        try:
            if not self.enhanced_mode or not self.dialogue_memory:
                return {"enhanced_mode": False, "message": "记忆系统未启用"}
            return self.dialogue_memory.get_memory_statistics()
        except Exception as e:
            self.logger.error(f"获取记忆统计失败: {e}")
            return {"error": str(e)}
    
    def generate_treatment_plan(self, patient_id: str, query: str = None) -> Dict[str, Any]:
        """为患者生成治疗方案"""
        try:
            if not self.enhanced_mode or not self.treatment_planner:
                return {
                    "success": False,
                    "error": "治疗方案生成功能未启用"
                }
                
            treatment_plan = self.treatment_planner.generate_comprehensive_treatment_plan(
                patient_id=patient_id,
                current_query=query,
                include_dialogue_context=True
            )
            
            return {
                "success": True,
                "treatment_plan": treatment_plan,
                "plan_id": treatment_plan.plan_id
            }
            
        except Exception as e:
            self.logger.error(f"生成治疗方案失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class MDTSystemInterface:
    """MDT系统主接口"""

    def __init__(self):
        # 初始化系统优化器, 这个不重要
        self.system_optimizer = get_system_optimizer()
        self.logger = self.system_optimizer.get_logger(self.__class__.__name__)
        
        # 初始化系统组件
        # 知识库RAG系统
        self.rag_system = MedicalKnowledgeRAG()
        # FAISS数据库管理器
        self.faiss_manager = EnhancedFAISSManager()
        # 患者对话管理器
        self.dialogue_manager_patient = PatientDialogueManager(self.faiss_manager)
        # 共识矩阵系统
        self.consensus_system = ConsensusMatrix()
        # 多智能体对话管理系统
        self.dialogue_manager = MultiAgentDialogueManager(self.rag_system)
        # 强化学习环境系统
        self.rl_environment = MDTReinforcementLearning(self.consensus_system)
        # 集成工作流管理系统
        self.workflow_manager = IntegratedWorkflowManager()
        # 系统可视化工具
        self.visualizer = SystemVisualizer()
        
        self.logger.info("MDT系统接口初始化完成")

        logger.info("MDT System initialized successfully")

    def run_patient_dialogue(self, patient_id: str = None) -> Dict[str, Any]:
        """运行患者对话模式 - 增强版本，支持对话记忆和治疗方案生成"""
        self.logger.info(f"启动患者对话模式，患者ID: {patient_id}")
        
        dialogue_history = []
        session_start_time = datetime.now()
        
        # 检查是否启用增强模式
        enhanced_mode = getattr(self.dialogue_manager_patient, 'enhanced_mode', False)
        
        print(f"\n=== 患者对话系统 {'(增强模式)' if enhanced_mode else '(基础模式)'} ===")
        if patient_id:
            print(f"当前患者: {patient_id}")
        else:
            print("通用查询模式")
        print("输入 'quit' 或 'exit' 退出对话")
        print("输入 'help' 查看帮助信息")
        if enhanced_mode:
            print("输入 'history' 查看对话历史")
            print("输入 'treatment' 生成治疗方案")
            print("输入 'stats' 查看记忆统计")
        print("=" * 50)
        
        # 显示患者历史对话（如果启用增强模式）
        if enhanced_mode and patient_id:
            history = self.dialogue_manager_patient.get_dialogue_history(patient_id, limit=3)
            if history:
                print(f"\n📋 最近对话记录 (共{len(history)}条):")
                for i, record in enumerate(history[-3:], 1):
                    print(f"  {i}. {record.get('timestamp', 'N/A')[:19]}: {record.get('user_input', 'N/A')[:50]}...")
                print("-" * 50)
        
        while True:
            try:
                # 获取用户输入
                if patient_id:
                    user_input = input(f"\n[患者 {patient_id}] 请输入您的问题: ").strip()
                else:
                    user_input = input(f"\n[通用查询] 请输入您的问题: ").strip()
                
                if not user_input:
                    continue
                
                # 检查退出命令
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    # 结束当前会话（如果启用增强模式）
                    if enhanced_mode:
                        session_summary = self.dialogue_manager_patient.end_current_session("user_quit")
                        if session_summary.get('session_id'):
                            print(f"✅ 对话会话已保存 (ID: {session_summary.get('session_id')})")
                    print("感谢使用患者对话系统，再见！")
                    break
                
                # 检查帮助命令
                if user_input.lower() in ['help', '帮助', 'h']:
                    self._show_dialogue_help(enhanced_mode)
                    continue
                
                # 检查历史命令（增强模式）
                if enhanced_mode and user_input.lower() in ['history', '历史', 'hist']:
                    if patient_id:
                        history = self.dialogue_manager_patient.get_dialogue_history(patient_id, limit=10)
                        self._show_dialogue_history(history)
                    else:
                        print("❌ 请先指定患者ID")
                    continue
                
                # 检查治疗方案命令（增强模式）
                if enhanced_mode and user_input.lower() in ['treatment', '治疗', 'plan']:
                    if patient_id:
                        treatment_result = self.dialogue_manager_patient.generate_treatment_plan(patient_id)
                        self._show_treatment_plan(treatment_result)
                    else:
                        print("❌ 请先指定患者ID")
                    continue
                
                # 检查统计命令（增强模式）
                if enhanced_mode and user_input.lower() in ['stats', '统计', 'statistics']:
                    stats = self.dialogue_manager_patient.get_memory_statistics()
                    self._show_memory_statistics(stats)
                    continue
                
                # 检查切换患者命令
                if user_input.startswith('patient:') or user_input.startswith('患者:'):
                    new_patient_id = user_input.split(':', 1)[1].strip()
                    if new_patient_id:
                        # 结束当前会话
                        if enhanced_mode and patient_id:
                            self.dialogue_manager_patient.end_current_session("patient_switch")
                        patient_id = new_patient_id
                        print(f"已切换到患者: {patient_id}")
                        # 显示新患者的历史对话
                        if enhanced_mode:
                            history = self.dialogue_manager_patient.get_dialogue_history(patient_id, limit=3)
                            if history:
                                print(f"📋 患者 {patient_id} 最近对话:")
                                for record in history[-3:]:
                                    print(f"  • {record.get('timestamp', 'N/A')[:19]}: {record.get('user_input', 'N/A')[:50]}...")
                        continue
                
                # 处理查询
                if not patient_id:
                    # 尝试从输入中提取患者ID
                    words = user_input.split()
                    for word in words:
                        if word.isdigit() and len(word) >= 6:  # 假设患者ID是6位以上数字
                            patient_id = word
                            print(f"检测到患者ID: {patient_id}")
                            break
                
                # 查询患者信息
                result = self.dialogue_manager_patient.query_patient_info(
                    patient_id or "unknown", user_input
                )
                
                # 显示回答
                print(f"\n🤖 系统回答:")
                print("-" * 40)
                print(result['response'])
                print("-" * 40)
                print(f"查询时间: {result['timestamp']}")
                print(f"搜索结果数量: {result['search_results_count']}")
                
                # 显示增强功能信息
                if enhanced_mode and result.get('enhanced_mode'):
                    print(f"会话ID: {result.get('session_id', 'N/A')}")
                    print(f"响应类型: {result.get('response_type', 'N/A')}")
                    if result.get('confidence_score'):
                        print(f"置信度: {result.get('confidence_score'):.2f}")
                    if result.get('treatment_plan_id'):
                        print(f"治疗方案ID: {result.get('treatment_plan_id')}")
                
                # 记录对话历史
                dialogue_history.append({
                    "timestamp": result['timestamp'],
                    "patient_id": result.get('patient_id'),
                    "query": result['query'],
                    "response": result['response'],
                    "search_results_count": result['search_results_count'],
                    "enhanced_mode": result.get('enhanced_mode', False),
                    "session_id": result.get('session_id'),
                    "response_type": result.get('response_type'),
                    "confidence_score": result.get('confidence_score'),
                    "treatment_plan_id": result.get('treatment_plan_id')
                })
                
            except KeyboardInterrupt:
                print("\n\n用户中断，退出对话系统")
                if enhanced_mode:
                    self.dialogue_manager_patient.end_current_session("user_interrupt")
                break
            except Exception as e:
                print(f"\n❌ 处理查询时出现错误: {e}")
                self.logger.error(f"对话系统错误: {e}")
        
        # 计算会话统计
        session_duration = (datetime.now() - session_start_time).total_seconds()
        
        # 返回对话历史
        return {
            "dialogue_history": dialogue_history,
            "total_queries": len(dialogue_history),
            "session_start_time": session_start_time.isoformat(),
            "session_end_time": datetime.now().isoformat(),
            "session_duration_seconds": session_duration,
            "enhanced_mode": enhanced_mode,
            "patient_id": patient_id
        }
    
    def _show_dialogue_help(self, enhanced_mode: bool = False):
        """显示对话系统帮助信息"""
        help_text = f"""
🔍 患者对话系统帮助 {'(增强模式)' if enhanced_mode else '(基础模式)'}

支持的查询类型:
• 诊断相关: "患者的诊断是什么？", "诊断信息"
• 治疗相关: "治疗方案", "手术情况", "化疗方案"
• 药物相关: "用药情况", "药物清单", "剂量信息"
• 检查相关: "检查结果", "实验室指标", "影像学检查"
• 病史相关: "既往病史", "家族史", "病史信息"

基础命令:
• patient:患者ID - 切换到指定患者
• help 或 帮助 - 显示此帮助信息
• quit 或 exit - 退出对话系统
"""
        
        if enhanced_mode:
            help_text += """
增强功能命令:
• history 或 历史 - 查看患者对话历史
• treatment 或 治疗 - 生成智能治疗方案
• stats 或 统计 - 查看记忆系统统计信息

增强功能特性:
✅ 对话记忆保存到FAISS向量数据库
✅ 基于历史对话的智能分析
✅ 共识矩阵优化的治疗方案生成
✅ 强化学习决策优化
✅ 持续学习和改进
"""
        
        help_text += """
示例查询:
• "10037928的诊断是什么？"
• "这个患者的用药情况如何？"
• "检查结果显示什么？"
• "有什么治疗建议？"
"""
        print(help_text)
    
    def _show_dialogue_history(self, history: List[Dict[str, Any]]):
        """显示对话历史"""
        if not history:
            print("📋 暂无对话历史记录")
            return
        
        print(f"\n📋 对话历史记录 (共{len(history)}条):")
        print("=" * 60)
        
        for i, record in enumerate(history, 1):
            timestamp = record.get('timestamp', 'N/A')
            user_input = record.get('user_input', 'N/A')
            agent_response = record.get('agent_response', 'N/A')
            response_type = record.get('response_type', 'general')
            
            print(f"{i}. 时间: {timestamp[:19] if timestamp != 'N/A' else 'N/A'}")
            print(f"   用户: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
            print(f"   系统: {agent_response[:100]}{'...' if len(agent_response) > 100 else ''}")
            print(f"   类型: {response_type}")
            print("-" * 60)
    
    def _show_treatment_plan(self, treatment_result: Dict[str, Any]):
        """显示治疗方案"""
        if not treatment_result.get('success'):
            print(f"❌ 治疗方案生成失败: {treatment_result.get('error', '未知错误')}")
            return
        
        treatment_plan = treatment_result.get('treatment_plan')
        if not treatment_plan:
            print("❌ 未获取到治疗方案数据")
            return
        
        print(f"\n🏥 智能治疗方案 (ID: {treatment_plan.plan_id})")
        print("=" * 60)
        print(f"患者ID: {treatment_plan.patient_id}")
        print(f"生成时间: {treatment_plan.created_at}")
        print(f"置信度: {treatment_plan.confidence_score:.2f}")
        print(f"优先级: {treatment_plan.priority}")
        
        if treatment_plan.primary_options:
            print(f"\n🎯 主要治疗选项 (共{len(treatment_plan.primary_options)}项):")
            for i, option in enumerate(treatment_plan.primary_options, 1):
                print(f"  {i}. {option.name}")
                print(f"     描述: {option.description}")
                print(f"     置信度: {option.confidence:.2f}")
                print(f"     预期效果: {option.expected_outcome}")
        
        if treatment_plan.alternative_options:
            print(f"\n🔄 备选治疗选项 (共{len(treatment_plan.alternative_options)}项):")
            for i, option in enumerate(treatment_plan.alternative_options, 1):
                print(f"  {i}. {option.name} (置信度: {option.confidence:.2f})")
        
        if treatment_plan.monitoring_plan:
            print(f"\n📊 监测计划:")
            for item in treatment_plan.monitoring_plan:
                print(f"  • {item}")
        
        if treatment_plan.follow_up_schedule:
            print(f"\n📅 随访安排:")
            for item in treatment_plan.follow_up_schedule:
                print(f"  • {item}")
        
        print("=" * 60)
    
    def _show_memory_statistics(self, stats: Dict[str, Any]):
        """显示记忆系统统计信息"""
        if stats.get('error'):
            print(f"❌ 获取统计信息失败: {stats.get('error')}")
            return
        
        if not stats.get('enhanced_mode', True):
            print("📊 记忆系统未启用")
            return
        
        print(f"\n📊 记忆系统统计信息")
        print("=" * 50)
        print(f"总对话数量: {stats.get('total_dialogues', 0)}")
        print(f"活跃患者数: {stats.get('active_patients', 0)}")
        print(f"向量数据库大小: {stats.get('vector_db_size', 0)}")
        print(f"平均对话长度: {stats.get('avg_dialogue_length', 0):.1f}")
        print(f"最后更新时间: {stats.get('last_update', 'N/A')}")
        
        if stats.get('top_patients'):
            print(f"\n🔥 最活跃患者:")
            for patient_id, count in stats.get('top_patients', []):
                print(f"  • 患者 {patient_id}: {count} 次对话")
        
        print("=" * 50)

    @optimized_function
    def run_single_patient_analysis(
        self, patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """运行单个患者的完整分析"""
        self.logger.info(
            f"Starting analysis for patient {patient_data.get('patient_id', 'unknown')}"
        )

        # 创建患者状态对象
        patient_state = self._create_patient_state(patient_data)

        # 运行多智能体对话与共识
        self.logger.info("Running multi-agent dialogue...")
        consensus_result = self.dialogue_manager.conduct_mdt_discussion(patient_state)

        # 生成可视化
        self.logger.info("Generating visualizations...")
        visualizations = self.visualizer.create_patient_analysis_dashboard(
            patient_state, consensus_result
        )

        # 整理结果
        analysis_result = {
            "patient_info": {
                "patient_id": patient_state.patient_id,
                "age": patient_state.age,
                "diagnosis": patient_state.diagnosis,
                "stage": patient_state.stage,
            },
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
            "dialogue_transcript": self.dialogue_manager.get_dialogue_transcript(),
            "visualizations": visualizations,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        self.logger.info("Single patient analysis completed successfully")
        return analysis_result

    @optimized_function
    def run_training_experiment(self, episodes: int = 1000) -> Dict[str, Any]:
        """运行RL训练实验"""
        self.logger.info(f"Starting RL training with {episodes} episodes")

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
        }

        logger.info("RL training experiment completed")
        return result

    def run_baseline_comparison(
        self, num_patients: int = 100, num_trials: int = 50
    ) -> Dict[str, Any]:
        """运行基线模型对比实验"""
        logger.info(
            f"Starting baseline comparison with {num_patients} patients, {num_trials} trials"
        )

        experiment = ComparisonExperiment()
        experiment.generate_test_patients(num_patients)
        results = experiment.run_comparison(num_trials)

        # 生成对比报告和可视化
        report = experiment.generate_comparison_report()
        experiment.plot_comparison_results("results/figures/baseline_comparison.png")

        comparison_result = {
            "comparison_results": results.to_dict("records"),
            "report": report,
            "visualization_saved": True,
        }

        logger.info("Baseline comparison completed")
        return comparison_result

    def run_integrated_simulation(
        self, patient_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """运行集成时序模拟"""
        logger.info(f"Starting integrated simulation for {patient_id}, {days} days")

        simulation_result = self.workflow_manager.run_temporal_simulation(
            patient_id, days
        )

        # 生成时序可视化
        temporal_visualizations = self.visualizer.create_temporal_analysis_dashboard(
            simulation_result
        )

        result = {
            "simulation_result": simulation_result,
            "visualizations": temporal_visualizations,
        }

        logger.info("Integrated simulation completed")
        return result

    def _create_patient_state(self, patient_data: Dict[str, Any]) -> PatientState:
        """从输入数据创建患者状态对象"""
        return PatientState(
            patient_id=patient_data.get("patient_id", "DEMO_001"),
            age=patient_data.get("age", 65),
            diagnosis=patient_data.get("diagnosis", "breast_cancer"),
            stage=patient_data.get("stage", "II"),
            lab_results=patient_data.get(
                "lab_results", {"creatinine": 1.2, "hemoglobin": 11.5}
            ),
            vital_signs=patient_data.get(
                "vital_signs", {"bp_systolic": 140, "heart_rate": 78}
            ),
            symptoms=patient_data.get("symptoms", ["fatigue", "pain"]),
            comorbidities=patient_data.get(
                "comorbidities", ["diabetes", "hypertension"]
            ),
            psychological_status=patient_data.get("psychological_status", "anxious"),
            quality_of_life_score=patient_data.get("quality_of_life_score", 0.7),
            timestamp=datetime.now(),
        )


def create_sample_patients() -> List[Dict[str, Any]]:
    """创建示例患者数据"""
    return [
        {
            "patient_id": "DEMO_001",
            "age": 65,
            "diagnosis": "breast_cancer",
            "stage": "II",
            "lab_results": {"creatinine": 1.2, "hemoglobin": 11.5},
            "vital_signs": {"bp_systolic": 140, "heart_rate": 78},
            "symptoms": ["fatigue", "pain"],
            "comorbidities": ["diabetes", "hypertension"],
            "psychological_status": "anxious",
            "quality_of_life_score": 0.7,
        },
        {
            "patient_id": "DEMO_002",
            "age": 45,
            "diagnosis": "breast_cancer",
            "stage": "I",
            "lab_results": {"creatinine": 0.9, "hemoglobin": 12.8},
            "vital_signs": {"bp_systolic": 120, "heart_rate": 72},
            "symptoms": ["mild_fatigue"],
            "comorbidities": [],
            "psychological_status": "stable",
            "quality_of_life_score": 0.85,
        },
        {
            "patient_id": "DEMO_003",
            "age": 78,
            "diagnosis": "breast_cancer",
            "stage": "III",
            "lab_results": {"creatinine": 1.8, "hemoglobin": 9.2},
            "vital_signs": {"bp_systolic": 160, "heart_rate": 85},
            "symptoms": ["fatigue", "pain", "shortness_of_breath"],
            "comorbidities": ["diabetes", "hypertension", "cardiac_dysfunction"],
            "psychological_status": "depressed",
            "quality_of_life_score": 0.4,
        },
    ]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MDT Medical AI System")

    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "patient", "training", "comparison", "simulation", "dialogue"],
        help="运行模式",
    )

    parser.add_argument("--patient-file", type=str, help="患者数据文件路径 (JSON格式)")

    parser.add_argument("--patient-id", type=str, help="患者ID (用于对话模式)")

    parser.add_argument("--episodes", type=int, default=1000, help="RL训练episode数量")

    parser.add_argument(
        "--num-patients", type=int, default=100, help="对比实验中的患者数量"
    )

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

    # 初始化系统
    print("=== MDT医疗智能体系统 ===")
    print("初始化系统组件...")
    
    # 启动系统优化器
    print("启动系统优化器...")
    system_optimizer.initialize()
    logger.info("系统优化器已启动")

    system = MDTSystemInterface()

    print(f"运行模式: {args.mode}")

    if args.mode == "demo":
        print("\n=== 演示模式 ===")
        sample_patients = create_sample_patients()

        for i, patient_data in enumerate(sample_patients, 1):
            print(f"\n--- 分析患者 {i}: {patient_data['patient_id']} ---")
            result = system.run_single_patient_analysis(patient_data)

            print(
                f"推荐治疗方案: {result['consensus_result']['recommended_treatment']}"
            )
            print(f"共识得分: {result['consensus_result']['consensus_score']:.3f}")
            print(f"对话轮数: {result['consensus_result']['total_rounds']}")
            print(f"是否收敛: {result['consensus_result']['convergence_achieved']}")

            # 保存结果
            import json

            output_file = (
                f"{args.output_dir}/patient_{patient_data['patient_id']}_analysis.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                # 处理不可序列化的对象
                serializable_result = result.copy()
                serializable_result.pop("visualizations", None)  # 移除可视化对象
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)

            print(f"详细结果已保存到: {output_file}")

    elif args.mode == "patient":
        print("\n=== 单患者分析模式 ===")
        if not args.patient_file:
            print("错误: 请提供患者数据文件 (--patient-file)")
            return

        # 加载患者数据
        import json

        with open(args.patient_file, "r", encoding="utf-8") as f:
            patient_data = json.load(f)

        result = system.run_single_patient_analysis(patient_data)

        print(f"患者 {patient_data['patient_id']} 分析完成")
        print(f"推荐治疗: {result['consensus_result']['recommended_treatment']}")

        # 保存结果
        output_file = (
            f"{args.output_dir}/patient_{patient_data['patient_id']}_analysis.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            serializable_result = result.copy()
            serializable_result.pop("visualizations", None)
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到: {output_file}")

    elif args.mode == "training":
        print(f"\n=== RL训练模式 ({args.episodes} episodes) ===")
        result = system.run_training_experiment(args.episodes)

        print("训练完成!")
        print(f"最终平均奖励: {result['final_metrics']['recent_average_reward']:.3f}")
        print(f"学习改进: {result['final_metrics']['improvement']:+.3f}")

        # 保存训练结果
        import json

        output_file = f"{args.output_dir}/training_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            serializable_result = result.copy()
            serializable_result.pop("visualizations", None)
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        print(f"训练结果已保存到: {output_file}")

    elif args.mode == "comparison":
        print(
            f"\n=== 基线对比模式 ({args.num_patients} 患者, {args.num_trials} 试验) ==="
        )
        result = system.run_baseline_comparison(args.num_patients, args.num_trials)

        print("对比实验完成!")
        print("\n" + result["report"])

        # 保存对比结果
        import json

        output_file = f"{args.output_dir}/comparison_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"对比结果已保存到: {output_file}")
        print(f"对比图表已保存到: {args.output_dir}/figures/baseline_comparison.png")

    elif args.mode == "simulation":
        print(f"\n=== 时序模拟模式 ({args.simulation_days} 天) ===")
        result = system.run_integrated_simulation("SIM_001", args.simulation_days)

        print("时序模拟完成!")
        print(f"总决策次数: {result['simulation_result']['total_decisions']}")
        print(f"平均共识得分: {result['simulation_result']['avg_consensus_score']:.3f}")

        # 保存模拟结果
        import json

        output_file = f"{args.output_dir}/simulation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            serializable_result = result.copy()
            serializable_result.pop("visualizations", None)
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        print(f"模拟结果已保存到: {output_file}")

    elif args.mode == "dialogue":
        print("\n=== 患者对话模式 ===")
        
        # 检查FAISS数据库是否存在
        faiss_db_path = "clinical_memory_db"
        if not os.path.exists(faiss_db_path):
            print(f"错误: FAISS数据库目录不存在: {faiss_db_path}")
            print("请确保已经初始化FAISS数据库")
            return
        
        # 启动对话模式
        try:
            result = system.run_patient_dialogue(args.patient_id)
            
            # 保存对话历史
            if result['dialogue_history']:
                import json
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"{args.output_dir}/dialogue_history_{timestamp}.json"
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"\n对话历史已保存到: {output_file}")
                print(f"总查询次数: {result['total_queries']}")
            else:
                print("\n未进行任何查询")
                
        except Exception as e:
            print(f"对话模式运行出错: {e}")
            logger.error(f"对话模式错误: {e}")

    print(f"\n所有输出文件保存在: {args.output_dir}/")
    
    # 生成系统性能报告
    print("生成系统性能报告...")
    try:
        report_path = system_optimizer.generate_report(args.output_dir)
        print(f"系统性能报告已保存到: {report_path}")
    except Exception as e:
        logger.error(f"生成性能报告失败: {e}")
    
    # 关闭系统优化器
    print("关闭系统优化器...")
    system_optimizer.shutdown()
    
    print("系统运行完成!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，系统退出")
    except Exception as e:
        logger.error(f"系统运行出错: {e}", exc_info=True)
        print(f"系统运行出错: {e}")
        sys.exit(1)
