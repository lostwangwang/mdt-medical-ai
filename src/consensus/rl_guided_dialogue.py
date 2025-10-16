"""
RL指导的对话管理器
文件路径: src/consensus/rl_guided_dialogue.py
作者: AI Assistant
功能: 实现RL策略反馈到智能体决策的机制，支持RL建议指导的多智能体对话
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    DialogueMessage,
    DialogueRound,
    ConsensusResult,
    RoleOpinion,
)
from .dialogue_manager import MultiAgentDialogueManager
from .role_agents import RoleAgent
from ..knowledge.rag_system import MedicalKnowledgeRAG

logger = logging.getLogger(__name__)


class RLGuidanceMode(Enum):
    """RL指导模式"""
    SOFT_GUIDANCE = "soft_guidance"      # 软指导：RL建议作为参考
    STRONG_GUIDANCE = "strong_guidance"  # 强指导：RL建议影响初始偏好
    ADAPTIVE_GUIDANCE = "adaptive_guidance"  # 自适应指导：基于RL置信度调整


@dataclass
class RLGuidance:
    """RL指导信息"""
    recommended_treatment: TreatmentOption
    confidence: float
    reasoning: str
    value_estimates: Dict[TreatmentOption, float]
    uncertainty: float
    guidance_mode: RLGuidanceMode


@dataclass
class RLInfluencedOpinion:
    """受RL影响的意见"""
    original_opinion: RoleOpinion
    rl_influenced_opinion: RoleOpinion
    influence_strength: float
    reasoning_modification: str


class RLGuidedDialogueManager(MultiAgentDialogueManager):
    """RL指导的对话管理器"""
    
    def __init__(
        self, 
        rag_system: MedicalKnowledgeRAG,
        guidance_mode: RLGuidanceMode = RLGuidanceMode.ADAPTIVE_GUIDANCE
    ):
        """
        初始化RL指导的对话管理器
        
        Args:
            rag_system: 医学知识RAG系统
            guidance_mode: RL指导模式
        """
        super().__init__(rag_system)
        self.guidance_mode = guidance_mode
        self.rl_guidance_history = []
        
        # RL指导配置
        self.guidance_config = {
            "influence_threshold": 0.7,  # RL置信度阈值
            "max_influence": 0.8,        # 最大影响强度
            "min_influence": 0.1,        # 最小影响强度
            "adaptation_rate": 0.1,      # 自适应调整率
        }
        
        logger.info(f"RLGuidedDialogueManager initialized with {guidance_mode.value} mode")
    
    def conduct_rl_guided_discussion(
        self, 
        patient_state: PatientState,
        rl_guidance: RLGuidance,
        context: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """
        进行RL指导的MDT讨论
        
        Args:
            patient_state: 患者状态
            rl_guidance: RL指导信息
            context: 额外上下文
            
        Returns:
            共识结果
        """
        logger.info(f"Starting RL-guided MDT discussion for patient {patient_state.patient_id}")
        logger.info(f"RL recommendation: {rl_guidance.recommended_treatment.value} (confidence: {rl_guidance.confidence:.3f})")
        
        # 记录RL指导历史
        self.rl_guidance_history.append(rl_guidance)
        
        # 根据指导模式调整智能体行为
        self._apply_rl_guidance_to_agents(rl_guidance, patient_state)
        
        # 初始化RL指导的讨论
        self._initialize_rl_guided_discussion(patient_state, rl_guidance)
        
        # 进行多轮对话（与基类相同的流程，但智能体已受RL影响）
        while self.current_round < self.max_rounds and not self._check_convergence():
            self.current_round += 1
            logger.info(f"Starting RL-guided dialogue round {self.current_round}")
            
            current_round = self._conduct_rl_guided_round(patient_state, rl_guidance)
            self.dialogue_rounds.append(current_round)
            
            # 更新智能体立场（考虑RL指导）
            self._update_agent_stances_with_rl(current_round, rl_guidance)
            
            # 检查是否需要调整RL指导强度
            if self.guidance_mode == RLGuidanceMode.ADAPTIVE_GUIDANCE:
                self._adapt_guidance_strength(rl_guidance)
        
        # 生成最终共识结果
        final_result = self._generate_rl_influenced_consensus(patient_state, rl_guidance)
        
        logger.info(f"RL-guided MDT discussion completed after {self.current_round} rounds")
        return final_result
    
    def _apply_rl_guidance_to_agents(
        self, 
        rl_guidance: RLGuidance, 
        patient_state: PatientState
    ):
        """将RL指导应用到智能体"""
        for role, agent in self.agents.items():
            influence_strength = self._calculate_influence_strength(
                role, rl_guidance, patient_state
            )
            
            # 为每个智能体设置RL指导信息
            agent.set_rl_guidance(rl_guidance, influence_strength)
            
            logger.debug(f"Applied RL guidance to {role.value} with strength {influence_strength:.3f}")
    
    def _calculate_influence_strength(
        self, 
        role: RoleType, 
        rl_guidance: RLGuidance, 
        patient_state: PatientState
    ) -> float:
        """计算RL对特定角色的影响强度"""
        base_influence = rl_guidance.confidence
        
        # 根据指导模式调整
        if self.guidance_mode == RLGuidanceMode.SOFT_GUIDANCE:
            influence_multiplier = 0.3
        elif self.guidance_mode == RLGuidanceMode.STRONG_GUIDANCE:
            influence_multiplier = 0.8
        else:  # ADAPTIVE_GUIDANCE
            # 基于RL置信度和不确定性自适应调整
            uncertainty_factor = 1.0 - rl_guidance.uncertainty
            influence_multiplier = 0.2 + 0.6 * uncertainty_factor
        
        # 角色特定的调整
        role_adjustments = {
            RoleType.ONCOLOGIST: 1.0,      # 肿瘤科医生对治疗建议最敏感
            RoleType.RADIOLOGIST: 0.8,     # 放射科医生中等敏感
            RoleType.NURSE: 0.6,           # 护士关注实用性
            RoleType.PSYCHOLOGIST: 0.4,    # 心理医生关注心理因素
            RoleType.PATIENT_ADVOCATE: 0.7 # 患者代表关注整体利益
        }
        
        role_factor = role_adjustments.get(role, 0.5)
        
        # 计算最终影响强度
        final_influence = base_influence * influence_multiplier * role_factor
        
        # 应用配置限制
        final_influence = np.clip(
            final_influence,
            self.guidance_config["min_influence"],
            self.guidance_config["max_influence"]
        )
        
        return final_influence
    
    def _initialize_rl_guided_discussion(
        self, 
        patient_state: PatientState, 
        rl_guidance: RLGuidance
    ):
        """初始化RL指导的讨论"""
        # 获取相关医学知识
        initial_knowledge = self.rag_system.retrieve_relevant_knowledge(
            patient_state, "initial_assessment"
        )
        
        # 添加RL指导信息到知识库
        rl_knowledge = {
            "rl_recommendation": rl_guidance.recommended_treatment.value,
            "rl_confidence": rl_guidance.confidence,
            "rl_reasoning": rl_guidance.reasoning,
            "value_estimates": {t.value: v for t, v in rl_guidance.value_estimates.items()}
        }
        
        enhanced_knowledge = {**initial_knowledge, "rl_guidance": rl_knowledge}
        
        # 生成初始轮次
        initial_round = DialogueRound(
            round_number=0,
            messages=[],
            focus_treatment=rl_guidance.recommended_treatment,  # 以RL建议为焦点
            consensus_status="discussing",
        )
        
        # 生成各角色的RL影响意见
        for role, agent in self.agents.items():
            opinion = agent.generate_rl_influenced_opinion(patient_state, enhanced_knowledge)
            
            # 创建初始消息
            message = self._create_rl_influenced_message(agent, opinion, patient_state, rl_guidance)
            initial_round.messages.append(message)
        
        self.dialogue_rounds.append(initial_round)
    
    def _conduct_rl_guided_round(
        self, 
        patient_state: PatientState, 
        rl_guidance: RLGuidance
    ) -> DialogueRound:
        """进行RL指导的对话轮次"""
        # 选择焦点治疗方案（考虑RL建议）
        focus_treatment = self._select_rl_influenced_focus(rl_guidance)
        
        # 确定发言顺序（优先考虑与RL建议相关的角色）
        speaking_order = self._determine_rl_influenced_order(focus_treatment, rl_guidance)
        
        # 创建新轮次
        current_round = DialogueRound(
            round_number=self.current_round,
            messages=[],
            focus_treatment=focus_treatment,
            consensus_status="discussing",
        )
        
        # 获取轮次相关知识
        round_knowledge = self.rag_system.retrieve_relevant_knowledge(
            patient_state, f"treatment_{focus_treatment.value}"
        )
        
        # 添加RL价值估计信息
        rl_value_info = {
            "rl_value_estimate": rl_guidance.value_estimates.get(focus_treatment, 0.0),
            "rl_uncertainty": rl_guidance.uncertainty,
            "alternative_values": {
                t.value: v for t, v in rl_guidance.value_estimates.items() 
                if t != focus_treatment
            }
        }
        
        enhanced_knowledge = {**round_knowledge, "rl_value_info": rl_value_info}
        
        # 各角色依次发言
        for role in speaking_order:
            agent = self.agents[role]
            
            # 生成基于RL指导的响应
            response = agent.generate_rl_guided_response(
                patient_state, 
                enhanced_knowledge, 
                self.dialogue_rounds,
                focus_treatment,
                rl_guidance
            )
            
            message = DialogueMessage(
                role=role,
                content=response["content"],
                timestamp=datetime.now(),
                message_type="rl_guided",
                referenced_roles=[],
                evidence_cited=response.get("references", []),
                treatment_focus=response.get("treatment_preference", focus_treatment)
            )
            
            current_round.messages.append(message)
        
        return current_round
    
    def _select_rl_influenced_focus(self, rl_guidance: RLGuidance) -> TreatmentOption:
        """选择受RL影响的焦点治疗方案"""
        # 如果RL置信度高，优先讨论RL推荐的治疗
        if rl_guidance.confidence > self.guidance_config["influence_threshold"]:
            return rl_guidance.recommended_treatment
        
        # 否则选择价值估计最高的几个方案中的一个
        sorted_treatments = sorted(
            rl_guidance.value_estimates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 在前3个方案中随机选择
        top_treatments = [t for t, v in sorted_treatments[:3]]
        return np.random.choice(top_treatments)
    
    def _determine_rl_influenced_order(
        self, 
        focus_treatment: TreatmentOption, 
        rl_guidance: RLGuidance
    ) -> List[RoleType]:
        """确定受RL影响的发言顺序"""
        # 基础发言顺序
        base_order = list(RoleType)
        
        # 如果焦点治疗与RL推荐一致，让支持该治疗的角色优先发言
        if focus_treatment == rl_guidance.recommended_treatment:
            # 根据角色对RL建议的敏感度排序
            role_sensitivity = {
                RoleType.ONCOLOGIST: 1.0,
                RoleType.RADIOLOGIST: 0.8,
                RoleType.PATIENT_ADVOCATE: 0.7,
                RoleType.NURSE: 0.6,
                RoleType.PSYCHOLOGIST: 0.4,
            }
            
            base_order.sort(key=lambda r: role_sensitivity.get(r, 0.5), reverse=True)
        
        return base_order
    
    def _update_agent_stances_with_rl(
        self, 
        round_data: DialogueRound, 
        rl_guidance: RLGuidance
    ):
        """更新智能体立场（考虑RL指导）"""
        # 使用现有的立场更新方法
        for role, agent in self.agents.items():
            agent.update_stance_based_on_dialogue(round_data.messages)
            
            # 额外考虑RL指导的影响
            if rl_guidance.confidence > 0.7:
                recommended_treatment = rl_guidance.recommended_treatment
                current_stance = agent.current_stance.get(recommended_treatment, 0.0)
                # 轻微调整立场朝向RL推荐
                adjustment = 0.1 * rl_guidance.confidence
                new_stance = min(1.0, max(-1.0, current_stance + adjustment))
                agent.current_stance[recommended_treatment] = new_stance
    
    def _adapt_guidance_strength(self, rl_guidance: RLGuidance):
        """自适应调整指导强度"""
        if self.guidance_mode != RLGuidanceMode.ADAPTIVE_GUIDANCE:
            return
        
        # 基于当前讨论状态调整指导强度
        current_consensus = self._assess_current_consensus()
        
        # 如果智能体与RL建议分歧较大，增强指导
        if current_consensus["agreement_with_rl"] < 0.5:
            for agent in self.agents.values():
                if hasattr(agent, 'rl_influence_strength'):
                    agent.rl_influence_strength = min(
                        self.guidance_config["max_influence"],
                        agent.rl_influence_strength + self.guidance_config["adaptation_rate"]
                    )
        
        # 如果智能体与RL建议高度一致，减弱指导
        elif current_consensus["agreement_with_rl"] > 0.8:
            for agent in self.agents.values():
                if hasattr(agent, 'rl_influence_strength'):
                    agent.rl_influence_strength = max(
                        self.guidance_config["min_influence"],
                        agent.rl_influence_strength - self.guidance_config["adaptation_rate"]
                    )
    
    def _assess_current_consensus(self) -> Dict[str, float]:
        """评估当前共识状态"""
        if not self.dialogue_rounds:
            return {"agreement_with_rl": 0.0, "overall_consensus": 0.0}
        
        latest_round = self.dialogue_rounds[-1]
        
        # 计算与RL建议的一致性
        rl_agreement_count = 0
        total_opinions = len(latest_round.messages)
        
        for message in latest_round.messages:
            if hasattr(message, 'treatment_focus'):
                # 这里需要获取当前的RL指导
                if len(self.rl_guidance_history) > 0:
                    current_rl_guidance = self.rl_guidance_history[-1]
                    if message.treatment_focus == current_rl_guidance.recommended_treatment:
                        rl_agreement_count += 1
        
        agreement_with_rl = rl_agreement_count / total_opinions if total_opinions > 0 else 0.0
        
        # 计算整体共识度
        treatment_counts = {}
        for message in latest_round.messages:
            if hasattr(message, 'treatment_focus'):
                treatment = message.treatment_focus
                treatment_counts[treatment] = treatment_counts.get(treatment, 0) + 1
        
        if treatment_counts:
            max_count = max(treatment_counts.values())
            overall_consensus = max_count / total_opinions
        else:
            overall_consensus = 0.0
        
        return {
            "agreement_with_rl": agreement_with_rl,
            "overall_consensus": overall_consensus
        }
    
    def _generate_rl_influenced_consensus(
        self, 
        patient_state: PatientState, 
        rl_guidance: RLGuidance
    ) -> ConsensusResult:
        """生成受RL影响的最终共识"""
        # 获取各角色的最终意见
        final_opinions = {}
        for role, agent in self.agents.items():
            # 设置RL指导属性
            agent.rl_guidance = rl_guidance
            agent.rl_influence_strength = 0.3  # 设置适中的影响强度
            
            # 使用现有的方法生成受RL影响的意见
            final_opinions[role] = agent.generate_rl_influenced_opinion(
                patient_state, {}
            )
        
        # 计算治疗方案得分（结合智能体意见和RL指导）
        treatment_scores = self._calculate_rl_influenced_scores(final_opinions, rl_guidance)
        
        # 选择最佳治疗方案
        recommended_treatment = max(treatment_scores.items(), key=lambda x: x[1])[0]
        
        # 计算共识分数（考虑RL影响）
        consensus_score = self._calculate_rl_influenced_consensus_score(
            final_opinions, recommended_treatment, rl_guidance
        )
        
        # 计算置信度
        confidence_level = self._calculate_rl_influenced_confidence(
            final_opinions, recommended_treatment, rl_guidance
        )
        
        # 生成讨论摘要
        discussion_summary = self._generate_rl_influenced_summary(rl_guidance)
        
        # 识别关键因素（包括RL因素）
        key_factors = self._identify_rl_influenced_factors(rl_guidance)
        
        # 识别风险
        risks_identified = self._identify_rl_influenced_risks(rl_guidance)
        
        # 生成替代方案（基于RL价值估计）
        alternative_options = self._generate_rl_influenced_alternatives(
            recommended_treatment, rl_guidance
        )
        
        return ConsensusResult(
            consensus_matrix=None,  # 可以后续添加矩阵计算
            role_opinions=final_opinions,
            aggregated_scores=treatment_scores,
            conflicts=[],  # 可以后续添加冲突检测
            agreements=[],  # 可以后续添加一致性检测
            dialogue_summary={
                "recommended_treatment": recommended_treatment,
                "consensus_score": consensus_score,
                "confidence_level": confidence_level,
                "discussion_summary": discussion_summary,
                "key_factors": key_factors,
                "risks_identified": risks_identified,
                "alternative_options": alternative_options,
                "follow_up_required": True
            },
            timestamp=datetime.now(),
            convergence_achieved=True,
            total_rounds=1
        )
    
    def _calculate_rl_influenced_scores(
        self, 
        final_opinions: Dict[RoleType, RoleOpinion], 
        rl_guidance: RLGuidance
    ) -> Dict[TreatmentOption, float]:
        """计算受RL影响的治疗方案得分"""
        treatment_scores = {}
        
        # 初始化所有治疗方案的得分
        for treatment in TreatmentOption:
            treatment_scores[treatment] = 0.0
        
        # 基于智能体意见计算得分
        for role, opinion in final_opinions.items():
            # 从治疗偏好中获取推荐治疗（评分最高的）
            treatment = max(opinion.treatment_preferences.items(), key=lambda x: x[1])[0]
            confidence = opinion.confidence
            role_weight = self._get_role_weight(role)
            
            treatment_scores[treatment] += confidence * role_weight
        
        # 添加RL指导的影响
        rl_weight = self._calculate_rl_weight(rl_guidance)
        for treatment, value in rl_guidance.value_estimates.items():
            treatment_scores[treatment] += value * rl_weight
        
        # 归一化得分
        max_score = max(treatment_scores.values()) if treatment_scores.values() else 1.0
        if max_score > 0:
            for treatment in treatment_scores:
                treatment_scores[treatment] /= max_score
        
        return treatment_scores
    
    def _calculate_rl_weight(self, rl_guidance: RLGuidance) -> float:
        """计算RL指导的权重"""
        base_weight = 0.3  # 基础权重
        
        # 基于置信度调整
        confidence_factor = rl_guidance.confidence
        
        # 基于不确定性调整
        uncertainty_factor = 1.0 - rl_guidance.uncertainty
        
        # 基于指导模式调整
        mode_factors = {
            RLGuidanceMode.SOFT_GUIDANCE: 0.5,
            RLGuidanceMode.STRONG_GUIDANCE: 1.5,
            RLGuidanceMode.ADAPTIVE_GUIDANCE: 1.0
        }
        
        mode_factor = mode_factors.get(self.guidance_mode, 1.0)
        
        final_weight = base_weight * confidence_factor * uncertainty_factor * mode_factor
        
        return np.clip(final_weight, 0.1, 0.8)
    
    def _get_role_weight(self, role: RoleType) -> float:
        """获取角色权重"""
        role_weights = {
            RoleType.ONCOLOGIST: 0.25,
            RoleType.RADIOLOGIST: 0.20,
            RoleType.NURSE: 0.15,
            RoleType.PSYCHOLOGIST: 0.20,
            RoleType.PATIENT_ADVOCATE: 0.20
        }
        return role_weights.get(role, 0.2)
    
    def _calculate_rl_influenced_consensus_score(
        self, 
        final_opinions: Dict[RoleType, RoleOpinion], 
        recommended_treatment: TreatmentOption,
        rl_guidance: RLGuidance
    ) -> float:
        """计算受RL影响的共识分数"""
        # 基础共识分数（智能体一致性）
        supporting_agents = sum(
            1 for opinion in final_opinions.values()
            if max(opinion.treatment_preferences.items(), key=lambda x: x[1])[0] == recommended_treatment
        )
        base_consensus = supporting_agents / len(final_opinions)
        
        # RL一致性奖励
        rl_bonus = 0.0
        if recommended_treatment == rl_guidance.recommended_treatment:
            rl_bonus = rl_guidance.confidence * 0.2
        
        # 综合共识分数
        final_consensus = base_consensus + rl_bonus
        
        return np.clip(final_consensus, 0.0, 1.0)
    
    def _calculate_rl_influenced_confidence(
        self, 
        final_opinions: Dict[RoleType, RoleOpinion], 
        recommended_treatment: TreatmentOption,
        rl_guidance: RLGuidance
    ) -> float:
        """计算受RL影响的置信度"""
        # 智能体平均置信度
        agent_confidences = [
            opinion.confidence for opinion in final_opinions.values()
            if max(opinion.treatment_preferences.items(), key=lambda x: x[1])[0] == recommended_treatment
        ]
        
        avg_agent_confidence = np.mean(agent_confidences) if agent_confidences else 0.5
        
        # RL置信度
        rl_confidence = rl_guidance.confidence if recommended_treatment == rl_guidance.recommended_treatment else 0.3
        
        # 综合置信度
        final_confidence = 0.7 * avg_agent_confidence + 0.3 * rl_confidence
        
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _generate_rl_influenced_summary(self, rl_guidance: RLGuidance) -> str:
        """生成受RL影响的讨论摘要"""
        summary_parts = [
            f"经过{self.current_round}轮讨论，团队在RL系统指导下达成共识。",
            f"RL系统推荐{rl_guidance.recommended_treatment.value}，置信度{rl_guidance.confidence:.2f}。",
            f"讨论采用{self.guidance_mode.value}模式，充分考虑了数据驱动的决策支持。"
        ]
        
        return " ".join(summary_parts)
    
    def _identify_rl_influenced_factors(self, rl_guidance: RLGuidance) -> List[str]:
        """识别受RL影响的关键因素"""
        factors = [
            "多学科专家意见",
            "RL系统数据分析",
            f"治疗价值评估: {rl_guidance.recommended_treatment.value}",
            "历史案例学习结果"
        ]
        
        # 添加高价值的替代方案
        sorted_treatments = sorted(
            rl_guidance.value_estimates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if len(sorted_treatments) > 1:
            second_best = sorted_treatments[1]
            factors.append(f"次优方案: {second_best[0].value} (价值: {second_best[1]:.2f})")
        
        return factors
    
    def _identify_rl_influenced_risks(self, rl_guidance: RLGuidance) -> List[str]:
        """识别受RL影响的风险"""
        risks = ["治疗效果不确定性", "患者个体差异"]
        
        # 基于RL不确定性添加风险
        if rl_guidance.uncertainty > 0.3:
            risks.append("RL模型预测不确定性较高")
        
        # 基于价值估计差异添加风险
        value_estimates = list(rl_guidance.value_estimates.values())
        if len(value_estimates) > 1:
            value_std = np.std(value_estimates)
            if value_std > 0.2:
                risks.append("治疗方案价值评估存在较大差异")
        
        return risks
    
    def _generate_rl_influenced_alternatives(
        self, 
        recommended_treatment: TreatmentOption, 
        rl_guidance: RLGuidance
    ) -> List[TreatmentOption]:
        """生成受RL影响的替代方案"""
        # 基于RL价值估计排序
        sorted_treatments = sorted(
            rl_guidance.value_estimates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 选择除推荐方案外的前2个高价值方案
        alternatives = []
        for treatment, value in sorted_treatments:
            if treatment != recommended_treatment and len(alternatives) < 2:
                alternatives.append(treatment)
        
        return alternatives
    
    def _create_rl_influenced_message(
        self, 
        agent: RoleAgent, 
        opinion: RoleOpinion, 
        patient_state: PatientState,
        rl_guidance: RLGuidance
    ) -> DialogueMessage:
        """创建受RL影响的消息"""
        # 从治疗偏好中找到最推荐的治疗方案
        recommended_treatment = max(opinion.treatment_preferences.items(), key=lambda x: x[1])[0]
        
        # 生成考虑RL指导的内容
        content_parts = [
            f"基于{agent.role.value}专业角度和RL系统分析，",
            f"我{self._get_recommendation_phrase(opinion.confidence)}{recommended_treatment.value}。"
        ]
        
        # 如果与RL建议一致，添加支持性说明
        if recommended_treatment == rl_guidance.recommended_treatment:
            content_parts.append(f"这与RL系统的推荐一致（置信度{rl_guidance.confidence:.2f}）。")
        
        # 如果与RL建议不一致，添加解释
        elif rl_guidance.confidence > 0.7:
            content_parts.append(f"虽然RL系统推荐{rl_guidance.recommended_treatment.value}，但从{agent.role.value}角度考虑，我认为{recommended_treatment.value}更合适。")
        
        content = "".join(content_parts)
        
        return DialogueMessage(
            role=agent.role,
            content=content,
            timestamp=datetime.now(),
            message_type="rl_influenced",
            referenced_roles=[],
            evidence_cited=[],
            treatment_focus=recommended_treatment
        )
    
    def get_rl_guidance_summary(self) -> Dict[str, Any]:
        """获取RL指导摘要"""
        if not self.rl_guidance_history:
            return {"message": "No RL guidance history"}
        
        recent_guidance = self.rl_guidance_history[-10:]  # 最近10次指导
        
        return {
            "total_guidance_sessions": len(self.rl_guidance_history),
            "current_mode": self.guidance_mode.value,
            "average_rl_confidence": np.mean([g.confidence for g in recent_guidance]),
            "average_uncertainty": np.mean([g.uncertainty for g in recent_guidance]),
            "most_recommended_treatment": self._analyze_rl_recommendations(recent_guidance),
            "guidance_effectiveness": self._assess_guidance_effectiveness()
        }
    
    def _analyze_rl_recommendations(self, guidance_list: List[RLGuidance]) -> Dict[str, Any]:
        """分析RL推荐分布"""
        recommendations = [g.recommended_treatment.value for g in guidance_list]
        unique_treatments = list(set(recommendations))
        
        distribution = {}
        for treatment in unique_treatments:
            distribution[treatment] = recommendations.count(treatment) / len(recommendations)
        
        most_common = max(distribution.items(), key=lambda x: x[1])
        
        return {
            "distribution": distribution,
            "most_common": most_common[0],
            "frequency": most_common[1]
        }
    
    def _assess_guidance_effectiveness(self) -> float:
        """评估指导效果"""
        if len(self.rl_guidance_history) < 2:
            return 0.5
        
        # 简化的效果评估：基于最近的共识质量
        # 实际应该基于更复杂的指标
        return 0.75  # 占位符
    
    def set_guidance_mode(self, mode: RLGuidanceMode):
        """设置指导模式"""
        self.guidance_mode = mode
        logger.info(f"RL guidance mode changed to {mode.value}")
    
    def update_guidance_config(self, new_config: Dict[str, Any]):
        """更新指导配置"""
        self.guidance_config.update(new_config)
        logger.info("RL guidance configuration updated")