"""
患者对话工作流管理器
文件路径: src/workflow/patient_dialogue_workflow.py
作者: AI Assistant
功能: 管理完整的患者对话工作流，集成记忆系统和治疗方案生成
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid

from ..knowledge.dialogue_memory_manager import DialogueMemoryManager
from ..knowledge.enhanced_faiss_integration import EnhancedFAISSManager
from ..treatment.enhanced_treatment_planner import EnhancedTreatmentPlanner, TreatmentPlan


@dataclass
class DialogueSession:
    """对话会话数据类"""
    session_id: str
    patient_id: str
    start_time: str
    end_time: Optional[str]
    total_turns: int
    session_type: str  # "consultation", "follow_up", "emergency"
    session_status: str  # "active", "completed", "interrupted"
    generated_plans: List[str]  # 生成的治疗方案ID列表
    session_summary: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class DialogueTurn:
    """单轮对话数据类"""
    turn_id: str
    session_id: str
    patient_id: str
    timestamp: str
    user_input: str
    agent_response: str
    response_type: str  # "information", "treatment_plan", "clarification", "emergency"
    confidence_score: float
    processing_time: float
    context_used: Dict[str, Any]
    generated_plan_id: Optional[str]


class PatientDialogueWorkflow:
    """患者对话工作流管理器"""
    
    def __init__(self, 
                 dialogue_memory_manager: DialogueMemoryManager,
                 faiss_manager: EnhancedFAISSManager,
                 treatment_planner: EnhancedTreatmentPlanner):
        self.dialogue_memory = dialogue_memory_manager
        self.faiss_manager = faiss_manager
        self.treatment_planner = treatment_planner
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 活跃会话管理
        self.active_sessions: Dict[str, DialogueSession] = {}
        
        # 响应模板
        self.response_templates = self._load_response_templates()
        
        self.logger.info("患者对话工作流管理器初始化完成")
    
    def start_dialogue_session(self, 
                             patient_id: str,
                             session_type: str = "consultation",
                             metadata: Dict[str, Any] = None) -> str:
        """开始新的对话会话"""
        try:
            session_id = f"session_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            session = DialogueSession(
                session_id=session_id,
                patient_id=patient_id,
                start_time=datetime.now().isoformat(),
                end_time=None,
                total_turns=0,
                session_type=session_type,
                session_status="active",
                generated_plans=[],
                session_summary=None,
                metadata=metadata or {}
            )
            
            self.active_sessions[session_id] = session
            
            self.logger.info(f"开始新对话会话: {session_id} (患者: {patient_id})")
            
            # 生成欢迎消息
            welcome_message = self._generate_welcome_message(patient_id, session_type)
            
            return session_id, welcome_message
            
        except Exception as e:
            self.logger.error(f"开始对话会话失败: {e}")
            raise
    
    def process_dialogue_turn(self, 
                            session_id: str,
                            user_input: str,
                            include_treatment_planning: bool = True) -> Tuple[str, Dict[str, Any]]:
        """处理单轮对话"""
        try:
            start_time = datetime.now()
            
            # 检查会话是否存在
            if session_id not in self.active_sessions:
                raise ValueError(f"会话不存在: {session_id}")
            
            session = self.active_sessions[session_id]
            patient_id = session.patient_id
            
            # 生成对话轮次ID
            turn_id = f"turn_{session_id}_{session.total_turns + 1}"
            
            # 分析用户输入
            input_analysis = self._analyze_user_input(user_input, patient_id)
            
            # 获取对话上下文
            dialogue_context = self.dialogue_memory.get_dialogue_context(
                patient_id, user_input
            )
            
            # 生成响应
            response_data = self._generate_intelligent_response(
                user_input, 
                input_analysis, 
                dialogue_context,
                include_treatment_planning
            )
            
            agent_response = response_data["response"]
            response_type = response_data["type"]
            confidence_score = response_data["confidence"]
            generated_plan_id = response_data.get("treatment_plan_id")
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 创建对话轮次记录
            dialogue_turn = DialogueTurn(
                turn_id=turn_id,
                session_id=session_id,
                patient_id=patient_id,
                timestamp=datetime.now().isoformat(),
                user_input=user_input,
                agent_response=agent_response,
                response_type=response_type,
                confidence_score=confidence_score,
                processing_time=processing_time,
                context_used=dialogue_context,
                generated_plan_id=generated_plan_id
            )
            
            # 保存到记忆系统
            dialogue_id = self.dialogue_memory.save_dialogue_turn(
                patient_id=patient_id,
                user_query=user_input,
                agent_response=agent_response,
                session_id=session_id,
                additional_metadata={
                    "turn_id": turn_id,
                    "response_type": response_type,
                    "confidence_score": confidence_score,
                    "processing_time": processing_time,
                    "input_analysis": input_analysis,
                    "generated_plan_id": generated_plan_id
                }
            )
            
            # 更新会话信息
            session.total_turns += 1
            if generated_plan_id:
                session.generated_plans.append(generated_plan_id)
            
            # 保存对话轮次
            self._save_dialogue_turn(dialogue_turn)
            
            # 准备返回数据
            return_data = {
                "turn_id": turn_id,
                "dialogue_id": dialogue_id,
                "response_type": response_type,
                "confidence_score": confidence_score,
                "processing_time": processing_time,
                "treatment_plan_id": generated_plan_id,
                "session_info": {
                    "total_turns": session.total_turns,
                    "session_type": session.session_type,
                    "generated_plans_count": len(session.generated_plans)
                }
            }
            
            self.logger.info(f"处理对话轮次完成: {turn_id}")
            
            return agent_response, return_data
            
        except Exception as e:
            self.logger.error(f"处理对话轮次失败: {e}")
            raise
    
    def end_dialogue_session(self, session_id: str, reason: str = "normal") -> Dict[str, Any]:
        """结束对话会话"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"会话不存在: {session_id}")
            
            session = self.active_sessions[session_id]
            session.end_time = datetime.now().isoformat()
            session.session_status = "completed" if reason == "normal" else "interrupted"
            
            # 生成会话摘要
            session_summary = self._generate_session_summary(session)
            session.session_summary = session_summary
            
            # 保存会话记录
            self._save_session_record(session)
            
            # 从活跃会话中移除
            del self.active_sessions[session_id]
            
            self.logger.info(f"结束对话会话: {session_id} (原因: {reason})")
            
            return {
                "session_id": session_id,
                "total_turns": session.total_turns,
                "duration": self._calculate_session_duration(session),
                "generated_plans": session.generated_plans,
                "session_summary": session_summary,
                "end_reason": reason
            }
            
        except Exception as e:
            self.logger.error(f"结束对话会话失败: {e}")
            raise
    
    def _analyze_user_input(self, user_input: str, patient_id: str) -> Dict[str, Any]:
        """分析用户输入"""
        try:
            analysis = {
                "input_length": len(user_input),
                "intent": "unknown",
                "urgency_level": "normal",
                "keywords": [],
                "medical_terms": [],
                "emotional_indicators": [],
                "requires_treatment_plan": False
            }
            
            user_input_lower = user_input.lower()
            
            # 意图识别
            if any(word in user_input_lower for word in ["治疗", "方案", "建议", "怎么办"]):
                analysis["intent"] = "treatment_inquiry"
                analysis["requires_treatment_plan"] = True
            elif any(word in user_input_lower for word in ["症状", "疼痛", "不舒服"]):
                analysis["intent"] = "symptom_report"
            elif any(word in user_input_lower for word in ["检查", "结果", "报告"]):
                analysis["intent"] = "test_inquiry"
            elif any(word in user_input_lower for word in ["药物", "副作用", "吃药"]):
                analysis["intent"] = "medication_inquiry"
            
            # 紧急程度识别
            if any(word in user_input_lower for word in ["急", "紧急", "严重", "痛", "出血"]):
                analysis["urgency_level"] = "high"
            elif any(word in user_input_lower for word in ["担心", "焦虑", "害怕"]):
                analysis["urgency_level"] = "medium"
            
            # 关键词提取
            medical_keywords = ["癌症", "肿瘤", "化疗", "手术", "放疗", "诊断", "治疗", "药物"]
            analysis["keywords"] = [kw for kw in medical_keywords if kw in user_input]
            
            # 情感指标
            emotional_words = ["担心", "害怕", "焦虑", "痛苦", "希望", "感谢"]
            analysis["emotional_indicators"] = [ew for ew in emotional_words if ew in user_input]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"分析用户输入失败: {e}")
            return {"error": str(e)}
    
    def _generate_intelligent_response(self, 
                                     user_input: str,
                                     input_analysis: Dict[str, Any],
                                     dialogue_context: Dict[str, Any],
                                     include_treatment_planning: bool) -> Dict[str, Any]:
        """生成智能响应"""
        try:
            response_data = {
                "response": "",
                "type": "information",
                "confidence": 0.7,
                "treatment_plan_id": None
            }
            
            intent = input_analysis.get("intent", "unknown")
            urgency = input_analysis.get("urgency_level", "normal")
            patient_id = dialogue_context.get("patient_id")
            
            # 根据意图生成响应
            if intent == "treatment_inquiry" and include_treatment_planning:
                # 生成治疗方案
                treatment_plan = self.treatment_planner.generate_comprehensive_treatment_plan(
                    patient_id=patient_id,
                    current_query=user_input,
                    include_dialogue_context=True
                )
                
                response_data["response"] = self._format_treatment_plan_response(treatment_plan)
                response_data["type"] = "treatment_plan"
                response_data["confidence"] = treatment_plan.confidence_score
                response_data["treatment_plan_id"] = treatment_plan.plan_id
                
            elif intent == "symptom_report":
                response_data["response"] = self._generate_symptom_response(
                    user_input, dialogue_context, urgency
                )
                response_data["type"] = "symptom_guidance"
                response_data["confidence"] = 0.8
                
            elif intent == "medication_inquiry":
                response_data["response"] = self._generate_medication_response(
                    user_input, dialogue_context
                )
                response_data["type"] = "medication_info"
                response_data["confidence"] = 0.75
                
            elif intent == "test_inquiry":
                response_data["response"] = self._generate_test_response(
                    user_input, dialogue_context
                )
                response_data["type"] = "test_interpretation"
                response_data["confidence"] = 0.8
                
            else:
                # 通用响应
                response_data["response"] = self._generate_general_response(
                    user_input, dialogue_context
                )
                response_data["type"] = "general_info"
                response_data["confidence"] = 0.6
            
            # 如果是高紧急度，添加紧急处理建议
            if urgency == "high":
                response_data["response"] += "\n\n⚠️ 根据您描述的症状，建议您立即联系医生或前往急诊科。"
                response_data["type"] = "emergency_guidance"
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"生成智能响应失败: {e}")
            return {
                "response": "抱歉，我在处理您的问题时遇到了技术问题。请稍后再试或联系医护人员。",
                "type": "error",
                "confidence": 0.0,
                "treatment_plan_id": None
            }
    
    def _format_treatment_plan_response(self, treatment_plan: TreatmentPlan) -> str:
        """格式化治疗方案响应"""
        try:
            response = f"基于您的病情和历史对话记录，我为您生成了以下治疗方案：\n\n"
            
            # 主要治疗选项
            response += "🎯 **推荐治疗方案**：\n"
            for i, option in enumerate(treatment_plan.primary_options, 1):
                response += f"{i}. **{option.name}** (置信度: {option.confidence_score:.1%})\n"
                response += f"   - 描述: {option.description}\n"
                response += f"   - 风险等级: {option.risk_level}\n"
                if option.expected_outcomes:
                    response += f"   - 预期效果: {', '.join(option.expected_outcomes)}\n"
                response += "\n"
            
            # 备选方案
            if treatment_plan.alternative_options:
                response += "🔄 **备选方案**：\n"
                for option in treatment_plan.alternative_options:
                    response += f"- {option.name} (置信度: {option.confidence_score:.1%})\n"
                response += "\n"
            
            # 注意事项
            if treatment_plan.contraindications:
                response += "⚠️ **注意事项**：\n"
                for contraindication in treatment_plan.contraindications:
                    response += f"- {contraindication}\n"
                response += "\n"
            
            # 监测要求
            if treatment_plan.monitoring_requirements:
                response += "📊 **监测要求**：\n"
                for requirement in treatment_plan.monitoring_requirements:
                    response += f"- {requirement}\n"
                response += "\n"
            
            # 随访计划
            if treatment_plan.follow_up_schedule:
                response += "📅 **随访计划**：\n"
                for schedule in treatment_plan.follow_up_schedule:
                    response += f"- {schedule}\n"
                response += "\n"
            
            response += f"💡 **方案置信度**: {treatment_plan.confidence_score:.1%}\n"
            response += f"🤝 **专家共识度**: {treatment_plan.consensus_score:.1%}\n\n"
            
            if treatment_plan.dialogue_context_used:
                response += "📝 此方案已考虑您的历史对话记录和关注点。\n"
            
            response += "\n请注意：此方案仅供参考，最终治疗决策请咨询您的主治医生。"
            
            return response
            
        except Exception as e:
            self.logger.error(f"格式化治疗方案响应失败: {e}")
            return "治疗方案生成完成，但格式化时出现问题。请联系技术支持。"
    
    def _generate_symptom_response(self, 
                                 user_input: str,
                                 dialogue_context: Dict[str, Any],
                                 urgency: str) -> str:
        """生成症状相关响应"""
        try:
            response = "我理解您对症状的担心。"
            
            # 基于历史对话调整响应
            similar_dialogues = dialogue_context.get("similar_dialogues", [])
            if similar_dialogues:
                response += "根据您之前的对话记录，"
            
            if urgency == "high":
                response += "您描述的症状需要立即关注。建议您：\n"
                response += "1. 立即联系您的主治医生\n"
                response += "2. 如果无法联系到医生，请前往急诊科\n"
                response += "3. 记录症状的详细情况和时间\n"
            else:
                response += "建议您：\n"
                response += "1. 详细记录症状的时间、程度和变化\n"
                response += "2. 在下次复诊时向医生详细描述\n"
                response += "3. 如果症状加重，及时联系医护团队\n"
            
            return response
            
        except Exception as e:
            self.logger.error(f"生成症状响应失败: {e}")
            return "我理解您的症状担忧，建议您联系医护团队获得专业指导。"
    
    def _generate_medication_response(self, 
                                    user_input: str,
                                    dialogue_context: Dict[str, Any]) -> str:
        """生成药物相关响应"""
        try:
            response = "关于药物使用，我建议您：\n"
            response += "1. 严格按照医生处方服用药物\n"
            response += "2. 如有副作用，及时记录并告知医生\n"
            response += "3. 不要自行调整药物剂量或停药\n"
            response += "4. 服药期间注意饮食和生活习惯\n\n"
            
            # 检查是否有相关的历史对话
            similar_dialogues = dialogue_context.get("similar_dialogues", [])
            if similar_dialogues:
                response += "根据您之前的咨询记录，请特别注意之前提到的注意事项。\n"
            
            response += "如有具体的药物问题，请咨询您的医生或药师。"
            
            return response
            
        except Exception as e:
            self.logger.error(f"生成药物响应失败: {e}")
            return "关于药物使用，请咨询您的医生或药师获得专业指导。"
    
    def _generate_test_response(self, 
                              user_input: str,
                              dialogue_context: Dict[str, Any]) -> str:
        """生成检查相关响应"""
        try:
            response = "关于检查结果，我建议：\n"
            response += "1. 检查结果需要专业医生解读\n"
            response += "2. 请在复诊时携带完整的检查报告\n"
            response += "3. 如有异常指标，医生会制定相应的处理方案\n"
            response += "4. 定期复查有助于监测病情变化\n\n"
            
            response += "请注意，我无法替代医生对检查结果的专业判断。"
            
            return response
            
        except Exception as e:
            self.logger.error(f"生成检查响应失败: {e}")
            return "关于检查结果，请咨询您的医生获得专业解读。"
    
    def _generate_general_response(self, 
                                 user_input: str,
                                 dialogue_context: Dict[str, Any]) -> str:
        """生成通用响应"""
        try:
            response = "感谢您的咨询。"
            
            # 基于对话历史个性化响应
            patterns = dialogue_context.get("dialogue_patterns", {})
            if patterns:
                most_common_type = patterns.get("most_common_query_type", "")
                if most_common_type:
                    response += f"我注意到您经常关注{most_common_type}相关的问题。"
            
            response += "\n\n我可以帮助您：\n"
            response += "1. 解答医疗相关问题\n"
            response += "2. 生成个性化治疗建议\n"
            response += "3. 提供健康管理指导\n"
            response += "4. 协助理解医疗信息\n\n"
            
            response += "请告诉我您具体想了解什么，我会尽力为您提供帮助。"
            
            return response
            
        except Exception as e:
            self.logger.error(f"生成通用响应失败: {e}")
            return "我是您的医疗AI助手，请告诉我您需要什么帮助。"
    
    def _generate_welcome_message(self, patient_id: str, session_type: str) -> str:
        """生成欢迎消息"""
        try:
            # 获取患者历史信息
            dialogue_patterns = self.dialogue_memory.analyze_dialogue_patterns(patient_id)
            
            if session_type == "follow_up":
                message = f"欢迎回来！我是您的医疗AI助手。"
                if dialogue_patterns.get("total_dialogues", 0) > 0:
                    message += f"我记录了您之前的 {dialogue_patterns['total_dialogues']} 次对话。"
            else:
                message = "您好！我是您的医疗AI助手，很高兴为您服务。"
            
            message += "\n\n我可以帮助您：\n"
            message += "• 解答医疗问题\n"
            message += "• 生成个性化治疗建议\n"
            message += "• 分析症状和检查结果\n"
            message += "• 提供用药指导\n\n"
            message += "请告诉我您今天想咨询什么问题？"
            
            return message
            
        except Exception as e:
            self.logger.error(f"生成欢迎消息失败: {e}")
            return "您好！我是您的医疗AI助手，请告诉我您需要什么帮助？"
    
    def _generate_session_summary(self, session: DialogueSession) -> str:
        """生成会话摘要"""
        try:
            summary = f"会话摘要 (患者ID: {session.patient_id}):\n"
            summary += f"- 会话类型: {session.session_type}\n"
            summary += f"- 对话轮次: {session.total_turns}\n"
            summary += f"- 生成治疗方案: {len(session.generated_plans)} 个\n"
            
            if session.generated_plans:
                summary += f"- 方案ID: {', '.join(session.generated_plans)}\n"
            
            duration = self._calculate_session_duration(session)
            summary += f"- 会话时长: {duration}\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生成会话摘要失败: {e}")
            return "会话摘要生成失败"
    
    def _calculate_session_duration(self, session: DialogueSession) -> str:
        """计算会话持续时间"""
        try:
            if not session.end_time:
                return "进行中"
            
            start = datetime.fromisoformat(session.start_time)
            end = datetime.fromisoformat(session.end_time)
            duration = end - start
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}小时{minutes}分钟"
            else:
                return f"{minutes}分钟"
                
        except Exception as e:
            self.logger.error(f"计算会话时长失败: {e}")
            return "未知"
    
    def _save_dialogue_turn(self, dialogue_turn: DialogueTurn):
        """保存对话轮次记录"""
        try:
            # 创建对话轮次目录
            turns_dir = "dialogue_turns"
            import os
            os.makedirs(turns_dir, exist_ok=True)
            
            # 保存到文件
            turn_file = os.path.join(turns_dir, f"{dialogue_turn.turn_id}.json")
            with open(turn_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(dialogue_turn), f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"保存对话轮次失败: {e}")
    
    def _save_session_record(self, session: DialogueSession):
        """保存会话记录"""
        try:
            # 创建会话记录目录
            sessions_dir = "dialogue_sessions"
            import os
            os.makedirs(sessions_dir, exist_ok=True)
            
            # 保存到文件
            session_file = os.path.join(sessions_dir, f"{session.session_id}.json")
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"保存会话记录失败: {e}")
    
    def _load_response_templates(self) -> Dict[str, str]:
        """加载响应模板"""
        try:
            # 这里可以从配置文件加载模板
            return {
                "welcome": "欢迎使用医疗AI助手",
                "goodbye": "感谢您的使用，祝您健康！",
                "error": "抱歉，处理您的请求时出现了问题。"
            }
        except Exception as e:
            self.logger.error(f"加载响应模板失败: {e}")
            return {}
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        try:
            active_count = len(self.active_sessions)
            
            # 统计活跃会话信息
            session_types = {}
            total_turns = 0
            
            for session in self.active_sessions.values():
                session_type = session.session_type
                session_types[session_type] = session_types.get(session_type, 0) + 1
                total_turns += session.total_turns
            
            return {
                "active_sessions": active_count,
                "session_types": session_types,
                "total_active_turns": total_turns,
                "average_turns_per_session": total_turns / active_count if active_count > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取会话统计失败: {e}")
            return {"error": str(e)}