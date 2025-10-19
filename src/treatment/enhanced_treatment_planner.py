"""
增强的治疗方案生成系统
文件路径: src/treatment/enhanced_treatment_planner.py
作者: AI Assistant
功能: 基于历史对话记忆和共识矩阵生成优化的治疗方案
"""

import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle
import os

from ..knowledge.dialogue_memory_manager import DialogueMemoryManager
from ..knowledge.enhanced_faiss_integration import EnhancedFAISSManager
from ..utils.llm_interface import LLMInterface, LLMConfig


@dataclass
class TreatmentOption:
    """
    治疗选项数据类
    
    参数:
    option_id: 治疗选项的唯一标识符
    name: 治疗选项的名称
    description: 治疗选项的详细描述
    confidence_score: 治疗选项的置信度分数（0-1之间）
    evidence_sources: 支持治疗选项的证据来源列表
    contraindications: 治疗选项的禁忌条件列表
    expected_outcomes: 治疗选项的预期结果列表
    risk_level: 治疗选项的风险等级（"low", "medium", "high"）
    cost_estimate: 治疗选项的成本估计（可选）
    duration_estimate: 治疗选项的持续时间估计（可选）
    """
    option_id: str
    name: str
    description: str
    confidence_score: float
    evidence_sources: List[str]
    contraindications: List[str]
    expected_outcomes: List[str]
    risk_level: str  # "low", "medium", "high"
    cost_estimate: Optional[float] = None
    duration_estimate: Optional[str] = None


@dataclass
class TreatmentPlan:
    """治疗方案数据类
    参数:
    plan_id: 治疗方案的唯一标识符
    patient_id: 患者ID
    primary_options: 主要治疗选项列表
    alternative_options: 替代治疗选项列表
    contraindications: 治疗选项的禁忌条件列表
    monitoring_requirements: 治疗过程中需要监控的指标列表
    follow_up_schedule: 后续治疗计划的时间 schedule
    confidence_score: 治疗方案的置信度分数（0-1之间）
    consensus_score: 治疗选项的共识分数（0-1之间）
    generated_timestamp: 治疗方案生成的时间戳
    dialogue_context_used: 是否使用了对话上下文
    historical_patterns_considered: 考虑的历史治疗模式列表
    """
    plan_id: str
    patient_id: str
    primary_options: List[TreatmentOption]
    alternative_options: List[TreatmentOption]
    contraindications: List[str]
    monitoring_requirements: List[str]
    follow_up_schedule: List[str]
    confidence_score: float
    consensus_score: float
    generated_timestamp: str
    dialogue_context_used: bool
    historical_patterns_considered: List[str]


class ConsensusMatrix:
    """共识矩阵系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.expert_weights = {
            "oncologist": 0.25,
            "surgeon": 0.20,
            "radiologist": 0.18,
            "pathologist": 0.12,
            "pharmacist": 0.08,
            "nutritionist": 0.09,
            "rehabilitation_therapist": 0.08
        }
    
    def calculate_consensus(self, 
                          treatment_options: List[TreatmentOption],
                          expert_opinions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算治疗选项的共识分数"""
        try:
            consensus_scores = {}
            
            for option in treatment_options:
                weighted_score = 0.0
                total_weight = 0.0
                
                for expert, weight in self.expert_weights.items():
                    if expert in expert_opinions and option.option_id in expert_opinions[expert]:
                        expert_score = expert_opinions[expert][option.option_id]
                        weighted_score += expert_score * weight
                        total_weight += weight
                
                if total_weight > 0:
                    consensus_scores[option.option_id] = weighted_score / total_weight
                else:
                    consensus_scores[option.option_id] = option.confidence_score
            
            return consensus_scores
            
        except Exception as e:
            self.logger.error(f"计算共识分数失败: {e}")
            return {option.option_id: option.confidence_score for option in treatment_options}
    
    def simulate_expert_opinions(self, 
                                treatment_options: List[TreatmentOption],
                                patient_context: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """模拟专家意见（在实际应用中应该是真实的专家评分）"""
        try:
            expert_opinions = {}
            
            for expert in self.expert_weights.keys():
                expert_opinions[expert] = {}
                
                for option in treatment_options:
                    # 基于治疗选项特征模拟专家评分
                    base_score = option.confidence_score
                    
                    # 根据专家类型调整评分
                    if expert == "oncologist":
                        # 肿瘤科医生更关注治疗效果
                        if "化疗" in option.name or "靶向" in option.name:
                            base_score += 0.1
                    elif expert == "surgeon":
                        # 外科医生更关注手术相关治疗
                        if "手术" in option.name or "切除" in option.name:
                            base_score += 0.15
                    elif expert == "radiologist":
                        # 放射科医生更关注影像相关治疗
                        if "放疗" in option.name or "影像" in option.name:
                            base_score += 0.1
                    
                    # 根据风险等级调整
                    if option.risk_level == "high":
                        base_score -= 0.05
                    elif option.risk_level == "low":
                        base_score += 0.05
                    
                    # 添加一些随机性模拟专家意见差异
                    noise = np.random.normal(0, 0.05)
                    final_score = np.clip(base_score + noise, 0.0, 1.0)
                    
                    expert_opinions[expert][option.option_id] = final_score
            
            return expert_opinions
            
        except Exception as e:
            self.logger.error(f"模拟专家意见失败: {e}")
            return {}


class ReinforcementLearningOptimizer:
    """强化学习优化器"""
    
    def __init__(self, model_path: str = "rl_treatment_model.pkl"):
        self.model_path = model_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        
        # 加载或初始化模型
        self._load_or_initialize_model()
    
    def _load_or_initialize_model(self):
        """加载或初始化强化学习模型"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
                self.logger.info("加载现有强化学习模型")
            else:
                self.model_data = {
                    "q_table": {},
                    "state_action_counts": {},
                    "total_episodes": 0,
                    "success_rate": 0.0
                }
                self.logger.info("初始化新的强化学习模型")
        except Exception as e:
            self.logger.error(f"加载强化学习模型失败: {e}")
            self.model_data = {
                "q_table": {},
                "state_action_counts": {},
                "total_episodes": 0,
                "success_rate": 0.0
            }
    
    def optimize_treatment_selection(self, 
                                   treatment_options: List[TreatmentOption],
                                   patient_state: Dict[str, Any],
                                   historical_outcomes: List[Dict[str, Any]]) -> List[TreatmentOption]:
        """使用强化学习优化治疗选项选择"""
        try:
            # 将患者状态转换为状态键
            state_key = self._encode_patient_state(patient_state)
            
            # 为每个治疗选项计算Q值
            optimized_options = []
            for option in treatment_options:
                action_key = f"{state_key}_{option.option_id}"
                
                # 获取Q值
                q_value = self.model_data["q_table"].get(action_key, option.confidence_score)
                
                # 基于历史结果调整
                historical_adjustment = self._calculate_historical_adjustment(option, historical_outcomes)
                
                # 更新置信度分数
                optimized_score = np.clip(q_value + historical_adjustment, 0.0, 1.0)
                
                # 创建优化后的选项
                optimized_option = TreatmentOption(
                    option_id=option.option_id,
                    name=option.name,
                    description=option.description,
                    confidence_score=optimized_score,
                    evidence_sources=option.evidence_sources,
                    contraindications=option.contraindications,
                    expected_outcomes=option.expected_outcomes,
                    risk_level=option.risk_level,
                    cost_estimate=option.cost_estimate,
                    duration_estimate=option.duration_estimate
                )
                
                optimized_options.append(optimized_option)
            
            # 按优化后的置信度排序
            optimized_options.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return optimized_options
            
        except Exception as e:
            self.logger.error(f"强化学习优化失败: {e}")
            return treatment_options
    
    def _encode_patient_state(self, patient_state: Dict[str, Any]) -> str:
        """将患者状态编码为字符串键"""
        try:
            # 提取关键特征
            age_group = "young" if patient_state.get("age", 0) < 50 else "old"
            gender = patient_state.get("gender", "unknown")
            cancer_stage = patient_state.get("cancer_stage", "unknown")
            comorbidities = len(patient_state.get("comorbidities", []))
            
            return f"{age_group}_{gender}_{cancer_stage}_{comorbidities}"
            
        except Exception as e:
            self.logger.error(f"编码患者状态失败: {e}")
            return "unknown_state"
    
    def _calculate_historical_adjustment(self, 
                                       option: TreatmentOption,
                                       historical_outcomes: List[Dict[str, Any]]) -> float:
        """基于历史结果计算调整值"""
        try:
            if not historical_outcomes:
                return 0.0
            
            # 查找相似治疗的历史结果
            similar_outcomes = [
                outcome for outcome in historical_outcomes
                if outcome.get("treatment_name", "").lower() in option.name.lower()
            ]
            
            if not similar_outcomes:
                return 0.0
            
            # 计算平均成功率
            success_rates = [outcome.get("success_rate", 0.5) for outcome in similar_outcomes]
            avg_success_rate = np.mean(success_rates)
            
            # 转换为调整值
            adjustment = (avg_success_rate - 0.5) * 0.2  # 最大调整±0.1
            
            return adjustment
            
        except Exception as e:
            self.logger.error(f"计算历史调整值失败: {e}")
            return 0.0
    
    def update_model(self, 
                    patient_state: Dict[str, Any],
                    selected_treatment: str,
                    outcome_score: float):
        """更新强化学习模型"""
        try:
            state_key = self._encode_patient_state(patient_state)
            action_key = f"{state_key}_{selected_treatment}"
            
            # 更新Q表
            current_q = self.model_data["q_table"].get(action_key, 0.5)
            new_q = current_q + self.learning_rate * (outcome_score - current_q)
            self.model_data["q_table"][action_key] = new_q
            
            # 更新计数
            self.model_data["state_action_counts"][action_key] = \
                self.model_data["state_action_counts"].get(action_key, 0) + 1
            
            # 更新总体统计
            self.model_data["total_episodes"] += 1
            
            # 保存模型
            self._save_model()
            
            self.logger.info(f"更新强化学习模型: {action_key} -> {new_q:.3f}")
            
        except Exception as e:
            self.logger.error(f"更新强化学习模型失败: {e}")
    
    def _save_model(self):
        """保存强化学习模型"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model_data, f)
        except Exception as e:
            self.logger.error(f"保存强化学习模型失败: {e}")


class EnhancedTreatmentPlanner:
    """增强的治疗方案规划器"""
    
    def __init__(self, 
                 dialogue_memory_manager: DialogueMemoryManager,
                 faiss_manager: EnhancedFAISSManager,
                 llm_config: LLMConfig = None):
        self.dialogue_memory = dialogue_memory_manager
        self.faiss_manager = faiss_manager
        self.consensus_matrix = ConsensusMatrix()
        self.rl_optimizer = ReinforcementLearningOptimizer()
        self.llm_interface = LLMInterface(llm_config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info("增强治疗方案规划器初始化完成")
    
    def generate_comprehensive_treatment_plan(self, 
                                            patient_id: str,
                                            current_query: str = None,
                                            include_dialogue_context: bool = True) -> TreatmentPlan:
        """生成综合治疗方案"""
        try:
            self.logger.info(f"为患者 {patient_id} 生成综合治疗方案")
            
            # 1. 获取患者基础信息
            patient_info = self._get_patient_info(patient_id)
            
            # 2. 获取对话上下文（如果启用）
            dialogue_context = None
            if include_dialogue_context and current_query:
                dialogue_context = self.dialogue_memory.get_dialogue_context(
                    patient_id, current_query
                )
            
            # 3. 生成基础治疗选项
            base_treatment_options = self._generate_base_treatment_options(
                patient_info, dialogue_context
            )
            
            # 4. 应用共识矩阵优化
            expert_opinions = self.consensus_matrix.simulate_expert_opinions(
                base_treatment_options, patient_info
            )
            consensus_scores = self.consensus_matrix.calculate_consensus(
                base_treatment_options, expert_opinions
            )
            
            # 5. 应用强化学习优化
            historical_outcomes = self._get_historical_outcomes(patient_id)
            optimized_options = self.rl_optimizer.optimize_treatment_selection(
                base_treatment_options, patient_info, historical_outcomes
            )
            
            # 6. 更新选项的共识分数
            for option in optimized_options:
                if option.option_id in consensus_scores:
                    option.confidence_score = (
                        option.confidence_score * 0.7 + 
                        consensus_scores[option.option_id] * 0.3
                    )
            
            # 7. 分类主要和备选方案
            optimized_options.sort(key=lambda x: x.confidence_score, reverse=True)
            primary_options = optimized_options[:3]
            alternative_options = optimized_options[3:6]
            
            # 8. 生成监测和随访要求
            monitoring_requirements = self._generate_monitoring_requirements(
                primary_options, patient_info
            )
            follow_up_schedule = self._generate_follow_up_schedule(
                primary_options, patient_info
            )
            
            # 9. 计算整体置信度和共识分数
            overall_confidence = np.mean([opt.confidence_score for opt in primary_options])
            overall_consensus = np.mean(list(consensus_scores.values()))
            
            # 10. 分析历史模式
            historical_patterns = self._analyze_historical_patterns(dialogue_context)
            
            # 11. 创建治疗方案
            treatment_plan = TreatmentPlan(
                plan_id=f"plan_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                patient_id=patient_id,
                primary_options=primary_options,
                alternative_options=alternative_options,
                contraindications=self._get_contraindications(patient_info),
                monitoring_requirements=monitoring_requirements,
                follow_up_schedule=follow_up_schedule,
                confidence_score=overall_confidence,
                consensus_score=overall_consensus,
                generated_timestamp=datetime.now().isoformat(),
                dialogue_context_used=include_dialogue_context and dialogue_context is not None,
                historical_patterns_considered=historical_patterns
            )
            
            # 12. 保存治疗方案
            self._save_treatment_plan(treatment_plan)
            
            self.logger.info(f"治疗方案生成完成: {treatment_plan.plan_id}")
            return treatment_plan
            
        except Exception as e:
            self.logger.error(f"生成治疗方案失败: {e}")
            raise
    
    def _get_patient_info(self, patient_id: str) -> Dict[str, Any]:
        """获取患者基础信息"""
        try:
            # 从FAISS数据库搜索患者信息
            search_results = self.faiss_manager.search_patient_data(
                f"patient_id:{patient_id}", k=1
            )
            
            if search_results:
                # 提取患者信息
                patient_info = search_results[0]["metadata"]
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
                return patient_info
            else:
                # 返回默认患者信息
                return {
                    "patient_id": patient_id,
                    "age": 50,
                    "gender": "unknown",
                    "diagnoses": [],
                    "comorbidities": [],
                    "current_medications": [],
                    "allergies": []
                }
                
        except Exception as e:
            self.logger.error(f"获取患者信息失败: {e}")
            return {"patient_id": patient_id}
    
    def _generate_base_treatment_options(self, 
                                       patient_info: Dict[str, Any],
                                       dialogue_context: Dict[str, Any] = None) -> List[TreatmentOption]:
        """基于LLM生成基础治疗选项"""
        try:
            # 首先尝试使用LLM生成治疗选项
            llm_options = self._generate_llm_treatment_options(patient_info, dialogue_context)
            
            if llm_options:
                self.logger.info(f"LLM成功生成 {len(llm_options)} 个治疗选项")
                return llm_options
            else:
                # 如果LLM失败，回退到传统方法
                self.logger.warning("LLM生成失败，使用传统方法生成治疗选项")
                return self._generate_fallback_treatment_options(patient_info, dialogue_context)
                
        except Exception as e:
            self.logger.error(f"生成治疗选项失败: {e}")
            return self._generate_fallback_treatment_options(patient_info, dialogue_context)
    
    def _generate_llm_treatment_options(self, 
                                      patient_info: Dict[str, Any],
                                      dialogue_context: Dict[str, Any] = None) -> List[TreatmentOption]:
        """使用LLM生成治疗选项"""
        try:
            # 构建LLM提示词
            prompt = self._build_treatment_options_prompt(patient_info, dialogue_context)
            
            # 调用LLM
            if self.llm_interface.client:
                response = self.llm_interface.client.chat.completions.create(
                    model=self.llm_interface.config.model_name,
                    messages=[
                        {"role": "system", "content": "你是一位资深的临床医生，擅长制定个性化治疗方案。请基于患者信息生成详细的治疗选项，以JSON格式返回。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # 降低温度以获得更一致的结果
                    max_tokens=2000
                )
                
                # 解析LLM响应
                llm_response = response.choices[0].message.content.strip()
                return self._parse_llm_treatment_response(llm_response)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"LLM生成治疗选项失败: {e}")
            return []
    
    def _generate_fallback_treatment_options(self, 
                                           patient_info: Dict[str, Any],
                                           dialogue_context: Dict[str, Any] = None) -> List[TreatmentOption]:
        """传统方法生成治疗选项（作为LLM失败时的备选方案）"""
        try:
            treatment_options = []
            
            # 基于患者诊断生成治疗选项
            diagnoses = patient_info.get("diagnoses", [])
            
            # 示例治疗选项（实际应用中应该基于医学知识库）
            if any("cancer" in str(diag).lower() or "tumor" in str(diag).lower() or 
                   "癌" in str(diag) or "肿瘤" in str(diag) for diag in diagnoses):
                # 癌症相关治疗
                treatment_options.extend([
                    TreatmentOption(
                        option_id="surgery_001",
                        name="手术切除",
                        description="通过外科手术切除肿瘤组织",
                        confidence_score=0.8,
                        evidence_sources=["NCCN指南", "临床试验数据"],
                        contraindications=["严重心脏病", "凝血功能异常"],
                        expected_outcomes=["肿瘤完全切除", "症状缓解"],
                        risk_level="medium"
                    ),
                    TreatmentOption(
                        option_id="chemo_001",
                        name="化学治疗",
                        description="使用化疗药物杀死癌细胞",
                        confidence_score=0.75,
                        evidence_sources=["临床试验", "Meta分析"],
                        contraindications=["严重肝肾功能不全", "骨髓抑制"],
                        expected_outcomes=["肿瘤缩小", "延长生存期"],
                        risk_level="high"
                    ),
                    TreatmentOption(
                        option_id="radiation_001",
                        name="放射治疗",
                        description="使用高能射线杀死癌细胞",
                        confidence_score=0.7,
                        evidence_sources=["放疗指南", "临床研究"],
                        contraindications=["既往放疗史", "妊娠"],
                        expected_outcomes=["局部控制", "症状缓解"],
                        risk_level="medium"
                    )
                ])
            
            # 基于对话上下文调整治疗选项
            if dialogue_context:
                recent_concerns = self._extract_patient_concerns(dialogue_context)
                for option in treatment_options:
                    if any(concern in option.description.lower() for concern in recent_concerns):
                        option.confidence_score += 0.1
            
            return treatment_options
            
        except Exception as e:
            self.logger.error(f"传统方法生成治疗选项失败: {e}")
            return []
    
    def _extract_patient_concerns(self, dialogue_context: Dict[str, Any]) -> List[str]:
        """从对话上下文中提取患者关注点"""
        try:
            concerns = []
            recent_dialogues = dialogue_context.get("recent_dialogues", [])
            
            for dialogue in recent_dialogues:
                query = dialogue.get("user_query", "").lower()
                if "疼痛" in query or "pain" in query:
                    concerns.append("疼痛管理")
                if "副作用" in query or "side effect" in query:
                    concerns.append("副作用")
                if "费用" in query or "cost" in query:
                    concerns.append("费用")
                if "时间" in query or "duration" in query:
                    concerns.append("治疗时间")
            
            return list(set(concerns))
            
        except Exception as e:
            self.logger.error(f"提取患者关注点失败: {e}")
            return []
    
    def _get_historical_outcomes(self, patient_id: str) -> List[Dict[str, Any]]:
        """获取历史治疗结果"""
        try:
            # 这里应该从实际的治疗结果数据库获取
            # 目前返回模拟数据
            return [
                {
                    "treatment_name": "手术切除",
                    "success_rate": 0.85,
                    "patient_satisfaction": 0.8,
                    "side_effects": ["轻微疼痛", "短期疲劳"]
                },
                {
                    "treatment_name": "化学治疗",
                    "success_rate": 0.7,
                    "patient_satisfaction": 0.6,
                    "side_effects": ["恶心", "脱发", "疲劳"]
                }
            ]
            
        except Exception as e:
            self.logger.error(f"获取历史治疗结果失败: {e}")
            return []
    
    def _generate_monitoring_requirements(self, 
                                        treatment_options: List[TreatmentOption],
                                        patient_info: Dict[str, Any]) -> List[str]:
        """生成监测要求"""
        try:
            monitoring = []
            
            for option in treatment_options:
                if "手术" in option.name:
                    monitoring.extend(["术后伤口检查", "感染指标监测", "疼痛评估"])
                elif "化疗" in option.name:
                    monitoring.extend(["血常规检查", "肝肾功能监测", "心电图检查"])
                elif "放疗" in option.name:
                    monitoring.extend(["皮肤反应检查", "放疗区域评估", "副作用监测"])
            
            # 去重并返回
            return list(set(monitoring))
            
        except Exception as e:
            self.logger.error(f"生成监测要求失败: {e}")
            return []
    
    def _generate_follow_up_schedule(self, 
                                   treatment_options: List[TreatmentOption],
                                   patient_info: Dict[str, Any]) -> List[str]:
        """生成随访计划"""
        try:
            schedule = [
                "治疗后1周复查",
                "治疗后1个月复查",
                "治疗后3个月复查",
                "治疗后6个月复查",
                "治疗后1年复查"
            ]
            
            # 根据治疗类型调整随访频率
            for option in treatment_options:
                if option.risk_level == "high":
                    schedule.insert(1, "治疗后3天复查")
                    break
            
            return schedule
            
        except Exception as e:
            self.logger.error(f"生成随访计划失败: {e}")
            return []
    
    def _get_contraindications(self, patient_info: Dict[str, Any]) -> List[str]:
        """获取禁忌症"""
        try:
            contraindications = []
            
            # 基于患者信息生成禁忌症
            allergies = patient_info.get("allergies", [])
            comorbidities = patient_info.get("comorbidities", [])
            
            if allergies:
                contraindications.extend([f"对{allergy}过敏" for allergy in allergies])
            
            if any("心脏" in str(comorb) for comorb in comorbidities):
                contraindications.append("严重心脏疾病")
            
            if any("肝" in str(comorb) for comorb in comorbidities):
                contraindications.append("严重肝功能不全")
            
            return contraindications
            
        except Exception as e:
            self.logger.error(f"获取禁忌症失败: {e}")
            return []
    
    def _analyze_historical_patterns(self, dialogue_context: Dict[str, Any]) -> List[str]:
        """分析历史模式"""
        try:
            if not dialogue_context:
                return []
            
            patterns = []
            dialogue_patterns = dialogue_context.get("dialogue_patterns", {})
            
            if dialogue_patterns:
                most_common_type = dialogue_patterns.get("most_common_query_type", "")
                if most_common_type:
                    patterns.append(f"患者最关注: {most_common_type}")
                
                total_dialogues = dialogue_patterns.get("total_dialogues", 0)
                if total_dialogues > 10:
                    patterns.append("高频互动患者")
                
                query_distribution = dialogue_patterns.get("query_type_distribution", {})
                if query_distribution.get("治疗相关", 0) > query_distribution.get("诊断相关", 0):
                    patterns.append("更关注治疗方案")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"分析历史模式失败: {e}")
            return []
    
    def _save_treatment_plan(self, treatment_plan: TreatmentPlan):
        """保存治疗方案"""
        try:
            # 创建治疗方案目录
            plans_dir = "treatment_plans"
            os.makedirs(plans_dir, exist_ok=True)
            
            # 保存到文件
            plan_file = os.path.join(plans_dir, f"{treatment_plan.plan_id}.json")
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(treatment_plan), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"治疗方案已保存: {plan_file}")
            
        except Exception as e:
            self.logger.error(f"保存治疗方案失败: {e}")
    
    def update_treatment_outcome(self, 
                               plan_id: str,
                               outcome_score: float,
                               feedback: Dict[str, Any] = None):
        """更新治疗结果，用于强化学习"""
        try:
            # 加载治疗方案
            plan_file = f"treatment_plans/{plan_id}.json"
            if not os.path.exists(plan_file):
                self.logger.warning(f"治疗方案文件不存在: {plan_file}")
                return
            
            with open(plan_file, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)
            
            # 获取患者信息
            patient_info = self._get_patient_info(plan_data["patient_id"])
            
            # 更新强化学习模型
            if plan_data["primary_options"]:
                selected_treatment = plan_data["primary_options"][0]["option_id"]
                self.rl_optimizer.update_model(patient_info, selected_treatment, outcome_score)
            
            # 保存反馈
            if feedback:
                feedback_file = f"treatment_plans/{plan_id}_feedback.json"
                with open(feedback_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "plan_id": plan_id,
                        "outcome_score": outcome_score,
                        "feedback": feedback,
                        "updated_timestamp": datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"治疗结果已更新: {plan_id}, 评分: {outcome_score}")
            
        except Exception as e:
            self.logger.error(f"更新治疗结果失败: {e}")
    
    def _build_treatment_options_prompt(self, 
                                      patient_info: Dict[str, Any],
                                      dialogue_context: Dict[str, Any] = None) -> str:
        """构建LLM治疗选项生成的提示词"""
        try:
            # 基础患者信息
            patient_summary = f"""
患者基本信息：
- 患者ID: {patient_info.get('patient_id', '未知')}
- 年龄: {patient_info.get('age', '未知')}
- 性别: {patient_info.get('gender', '未知')}
- 主要诊断: {', '.join(patient_info.get('diagnoses', ['未明确诊断']))}
- 既往病史: {', '.join(patient_info.get('medical_history', ['无特殊病史']))}
- 当前症状: {', '.join(patient_info.get('symptoms', ['无明显症状']))}
- 实验室检查: {patient_info.get('lab_results', '暂无检查结果')}
- 影像学检查: {patient_info.get('imaging_results', '暂无影像学检查')}
- 过敏史: {', '.join(patient_info.get('allergies', ['无已知过敏']))}
- 当前用药: {', '.join(patient_info.get('current_medications', ['无当前用药']))}
"""
            
            # 对话上下文信息
            context_info = ""
            if dialogue_context:
                recent_queries = dialogue_context.get('recent_queries', [])
                patient_concerns = dialogue_context.get('patient_concerns', [])
                
                if recent_queries:
                    context_info += f"\n最近的咨询问题: {', '.join(recent_queries[-3:])}"
                if patient_concerns:
                    context_info += f"\n患者关注点: {', '.join(patient_concerns)}"
            
            # 构建完整提示词
            prompt = f"""
请基于以下患者信息，生成3-5个个性化的治疗选项。每个治疗选项应该包含详细的医学信息。

{patient_summary}
{context_info}

请以JSON格式返回治疗选项，格式如下：
{{
  "treatment_options": [
    {{
      "option_id": "唯一标识符（如：treatment_001）",
      "name": "治疗方案名称",
      "description": "详细的治疗描述，包括具体的治疗方法、药物、剂量等",
      "confidence_score": 0.0-1.0之间的置信度分数,
      "evidence_sources": ["支持该治疗的证据来源列表"],
      "contraindications": ["禁忌症列表"],
      "expected_outcomes": ["预期治疗结果列表"],
      "risk_level": "low/medium/high",
      "cost_estimate": 预估费用（数字，可选）,
      "duration_estimate": "预估治疗时间（字符串，可选）"
    }}
  ]
}}

要求：
1. 治疗选项应该基于循证医学，符合当前医学指南
2. 考虑患者的具体情况（年龄、性别、既往病史、过敏史等）
3. 提供不同风险等级和治疗强度的选项
4. 置信度分数应该反映治疗的适用性和有效性
5. 确保返回的是有效的JSON格式
"""
            
            return prompt.strip()
            
        except Exception as e:
            self.logger.error(f"构建LLM提示词失败: {e}")
            return "请为患者生成治疗选项"
    
    def _parse_llm_treatment_response(self, llm_response: str) -> List[TreatmentOption]:
        """解析LLM响应并构建TreatmentOption对象"""
        try:
            # 清理响应文本，移除可能的markdown标记
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # 解析JSON
            try:
                response_data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试提取JSON部分
                import re
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group())
                else:
                    raise ValueError("无法找到有效的JSON数据")
            
            # 构建TreatmentOption对象列表
            treatment_options = []
            options_data = response_data.get('treatment_options', [])
            
            for i, option_data in enumerate(options_data):
                try:
                    # 验证必需字段
                    option_id = option_data.get('option_id', f'llm_generated_{i+1:03d}')
                    name = option_data.get('name', f'治疗选项{i+1}')
                    description = option_data.get('description', '详细描述待补充')
                    
                    # 处理置信度分数
                    confidence_score = float(option_data.get('confidence_score', 0.5))
                    confidence_score = max(0.0, min(1.0, confidence_score))  # 确保在0-1范围内
                    
                    # 处理列表字段
                    evidence_sources = option_data.get('evidence_sources', [])
                    if isinstance(evidence_sources, str):
                        evidence_sources = [evidence_sources]
                    
                    contraindications = option_data.get('contraindications', [])
                    if isinstance(contraindications, str):
                        contraindications = [contraindications]
                    
                    expected_outcomes = option_data.get('expected_outcomes', [])
                    if isinstance(expected_outcomes, str):
                        expected_outcomes = [expected_outcomes]
                    
                    # 处理风险等级
                    risk_level = option_data.get('risk_level', 'medium').lower()
                    if risk_level not in ['low', 'medium', 'high']:
                        risk_level = 'medium'
                    
                    # 处理可选字段
                    cost_estimate = option_data.get('cost_estimate')
                    if cost_estimate is not None:
                        try:
                            cost_estimate = float(cost_estimate)
                        except (ValueError, TypeError):
                            cost_estimate = None
                    
                    duration_estimate = option_data.get('duration_estimate')
                    if duration_estimate and not isinstance(duration_estimate, str):
                        duration_estimate = str(duration_estimate)
                    
                    # 创建TreatmentOption对象
                    treatment_option = TreatmentOption(
                        option_id=option_id,
                        name=name,
                        description=description,
                        confidence_score=confidence_score,
                        evidence_sources=evidence_sources,
                        contraindications=contraindications,
                        expected_outcomes=expected_outcomes,
                        risk_level=risk_level,
                        cost_estimate=cost_estimate,
                        duration_estimate=duration_estimate
                    )
                    
                    treatment_options.append(treatment_option)
                    
                except Exception as option_error:
                    self.logger.warning(f"解析治疗选项 {i+1} 失败: {option_error}")
                    continue
            
            if treatment_options:
                self.logger.info(f"成功解析 {len(treatment_options)} 个LLM生成的治疗选项")
            else:
                self.logger.warning("未能解析出任何有效的治疗选项")
            
            return treatment_options
            
        except Exception as e:
            self.logger.error(f"解析LLM治疗选项响应失败: {e}")
            self.logger.debug(f"原始响应: {llm_response}")
            return []