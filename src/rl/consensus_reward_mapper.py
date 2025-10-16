"""
智能体共识到RL奖励映射器
文件路径: src/rl/consensus_reward_mapper.py
作者: AI Assistant
功能: 将智能体共识结果映射为RL奖励信号，实现专家知识指导的强化学习
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    ConsensusResult,
    RoleOpinion,
    RLReward,
)

logger = logging.getLogger(__name__)


class RewardComponent(Enum):
    """奖励组件类型"""
    CONSENSUS_QUALITY = "consensus_quality"          # 共识质量
    EXPERT_AGREEMENT = "expert_agreement"            # 专家一致性
    ROLE_DIVERSITY = "role_diversity"                # 角色多样性
    CONFIDENCE_LEVEL = "confidence_level"            # 置信度水平
    CLINICAL_REASONING = "clinical_reasoning"        # 临床推理质量
    SAFETY_ASSESSMENT = "safety_assessment"          # 安全性评估
    TREATMENT_APPROPRIATENESS = "treatment_appropriateness"  # 治疗适宜性


@dataclass
class RewardBreakdown:
    """奖励分解结构"""
    consensus_quality: float
    expert_agreement: float
    role_diversity: float
    confidence_level: float
    clinical_reasoning: float
    safety_assessment: float
    treatment_appropriateness: float
    patient_suitability: float
    total_reward: float
    component_weights: Dict[str, float]
    explanation: str


class ConsensusRewardMapper:
    """智能体共识到RL奖励映射器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化奖励映射器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()
        self.reward_history = []
        
        # 角色权重配置
        self.role_weights = {
            RoleType.ONCOLOGIST: 0.25,      # 肿瘤科医生
            RoleType.RADIOLOGIST: 0.20,     # 放射科医生
            RoleType.NURSE: 0.15,           # 护士
            RoleType.PSYCHOLOGIST: 0.20,    # 心理医生
            RoleType.PATIENT_ADVOCATE: 0.20 # 患者代表
        }
        
        logger.info("ConsensusRewardMapper initialized")
    
    def map_consensus_to_reward(
        self,
        consensus_result: ConsensusResult,
        patient_state: PatientState,
        selected_action: TreatmentOption,
        context: Optional[Dict[str, Any]] = None
    ) -> RewardBreakdown:
        """
        将智能体共识结果映射为RL奖励
        
        Args:
            consensus_result: 智能体共识结果
            patient_state: 患者状态
            selected_action: 选择的治疗方案
            context: 额外上下文信息
            
        Returns:
            奖励分解结果
        """
        logger.debug(f"Mapping consensus to reward for action {selected_action.value}")
        
        # 计算各个奖励组件
        consensus_quality = self._calculate_consensus_quality(consensus_result)
        expert_agreement = self._calculate_expert_agreement(consensus_result, selected_action)
        role_diversity = self._calculate_role_diversity(consensus_result)
        confidence_level = self._calculate_confidence_level(consensus_result)
        clinical_reasoning = self._calculate_clinical_reasoning(consensus_result)
        safety_assessment = self._calculate_safety_assessment(consensus_result, patient_state)
        treatment_appropriateness = self._calculate_treatment_appropriateness(
            consensus_result, patient_state, selected_action
        )
        patient_suitability = self._calculate_patient_suitability(patient_state, selected_action)
        
        # 获取组件权重
        weights = self.config["component_weights"]
        
        # 计算总奖励
        total_reward = (
            consensus_quality * weights[RewardComponent.CONSENSUS_QUALITY.value] +
            expert_agreement * weights[RewardComponent.EXPERT_AGREEMENT.value] +
            role_diversity * weights[RewardComponent.ROLE_DIVERSITY.value] +
            confidence_level * weights[RewardComponent.CONFIDENCE_LEVEL.value] +
            clinical_reasoning * weights[RewardComponent.CLINICAL_REASONING.value] +
            safety_assessment * weights[RewardComponent.SAFETY_ASSESSMENT.value] +
            treatment_appropriateness * weights[RewardComponent.TREATMENT_APPROPRIATENESS.value] +
            patient_suitability * weights.get("patient_suitability", 0.2)
        )
        
        # 应用奖励调节
        total_reward = self._apply_reward_modulation(total_reward, context)
        
        # 生成解释
        explanation = self._generate_reward_explanation(
            consensus_quality, expert_agreement, role_diversity,
            confidence_level, clinical_reasoning, safety_assessment,
            treatment_appropriateness, patient_suitability, selected_action
        )
        
        # 创建奖励分解结果
        reward_breakdown = RewardBreakdown(
            consensus_quality=consensus_quality,
            expert_agreement=expert_agreement,
            role_diversity=role_diversity,
            confidence_level=confidence_level,
            clinical_reasoning=clinical_reasoning,
            safety_assessment=safety_assessment,
            treatment_appropriateness=treatment_appropriateness,
            patient_suitability=patient_suitability,
            total_reward=total_reward,
            component_weights=weights,
            explanation=explanation
        )
        
        # 记录奖励历史
        self.reward_history.append(reward_breakdown)
        
        return reward_breakdown
    
    def _calculate_consensus_quality(self, consensus_result: ConsensusResult) -> float:
        """计算共识质量分数"""
        # 基于聚合分数计算基础分数
        if consensus_result.aggregated_scores:
            base_score = max(consensus_result.aggregated_scores.values())
        else:
            base_score = 0.5
        
        # 考虑讨论轮次（更多轮次可能表示更深入的讨论）
        discussion_depth_bonus = min(0.1, consensus_result.total_rounds * 0.02)
        
        # 考虑冲突数量（更少冲突表示更好的共识）
        conflict_penalty = min(0.2, len(consensus_result.conflicts) * 0.05)
        
        quality_score = base_score + discussion_depth_bonus - conflict_penalty
        
        return np.clip(quality_score, 0.0, 1.0)
    
    def _calculate_expert_agreement(
        self, 
        consensus_result: ConsensusResult, 
        selected_action: TreatmentOption
    ) -> float:
        """计算专家一致性分数"""
        if not consensus_result.role_opinions:
            return 0.0
        
        # 计算支持选定治疗方案的专家比例
        supporting_experts = 0
        total_experts = len(consensus_result.role_opinions)
        
        for role, opinion in consensus_result.role_opinions.items():
            # 检查该角色对选定治疗的偏好分数
            if selected_action in opinion.treatment_preferences:
                preference_score = opinion.treatment_preferences[selected_action]
                if preference_score > 0.5:  # 偏好分数大于0.5认为是支持
                    supporting_experts += 1
        
        agreement_ratio = supporting_experts / total_experts
        
        # 加权计算（考虑不同角色的重要性）
        weighted_agreement = 0.0
        total_weight = 0.0
        
        for role, opinion in consensus_result.role_opinions.items():
            weight = self.role_weights.get(role, 0.2)
            # 检查该角色对选定治疗的偏好分数
            if selected_action in opinion.treatment_preferences:
                preference_score = opinion.treatment_preferences[selected_action]
                if preference_score > 0.5:  # 偏好分数大于0.5认为是支持
                    weighted_agreement += weight * opinion.confidence
            total_weight += weight
        
        if total_weight > 0:
            weighted_agreement /= total_weight
        
        # 结合简单比例和加权一致性
        final_agreement = 0.6 * agreement_ratio + 0.4 * weighted_agreement
        
        return np.clip(final_agreement, 0.0, 1.0)
    
    def _calculate_role_diversity(self, consensus_result: ConsensusResult) -> float:
        """计算角色多样性分数"""
        if not consensus_result.role_opinions:
            return 0.0
        
        # 检查是否有来自不同专业的意见
        represented_roles = set(consensus_result.role_opinions.keys())
        total_roles = len(RoleType)
        diversity_ratio = len(represented_roles) / total_roles
        
        # 检查意见的多样性（不同的治疗建议）
        unique_treatments = set()
        for opinion in consensus_result.role_opinions.values():
            # 找到该角色偏好最高的治疗方案
            if opinion.treatment_preferences:
                best_treatment = max(opinion.treatment_preferences, key=opinion.treatment_preferences.get)
                unique_treatments.add(best_treatment)
        
        # 适度的多样性是好的，但过度分歧不好
        treatment_diversity = len(unique_treatments) / len(consensus_result.role_opinions)
        
        # 最优多样性在2-3个不同意见之间
        if treatment_diversity <= 0.5:  # 过度一致
            diversity_bonus = treatment_diversity * 0.8
        elif treatment_diversity <= 0.7:  # 适度多样性
            diversity_bonus = 1.0
        else:  # 过度分歧
            diversity_bonus = 1.0 - (treatment_diversity - 0.7) * 2
        
        final_diversity = 0.7 * diversity_ratio + 0.3 * diversity_bonus
        
        return np.clip(final_diversity, 0.0, 1.0)
    
    def _calculate_confidence_level(self, consensus_result: ConsensusResult) -> float:
        """计算置信度水平分数"""
        if not consensus_result.role_opinions:
            # Get confidence_level from dialogue_summary if available
            if hasattr(consensus_result, 'dialogue_summary') and consensus_result.dialogue_summary:
                return consensus_result.dialogue_summary.get('confidence_level', 0.5)
            return 0.5
        
        # 计算所有专家的平均置信度
        total_confidence = sum(
            opinion.confidence for opinion in consensus_result.role_opinions.values()
        )
        average_confidence = total_confidence / len(consensus_result.role_opinions)
        
        # 结合整体置信度和专家平均置信度
        # 使用convergence_achieved作为整体置信度的代理
        overall_confidence = 1.0 if consensus_result.convergence_achieved else 0.5
        combined_confidence = 0.6 * overall_confidence + 0.4 * average_confidence
        
        return np.clip(combined_confidence, 0.0, 1.0)
    
    def _calculate_clinical_reasoning(self, consensus_result: ConsensusResult) -> float:
        """计算临床推理质量分数"""
        reasoning_score = 0.0
        
        # 基于推理的详细程度
        if consensus_result.role_opinions:
            reasoning_lengths = []
            for opinion in consensus_result.role_opinions.values():
                if hasattr(opinion, 'reasoning') and opinion.reasoning:
                    reasoning_lengths.append(len(opinion.reasoning.split()))
            
            if reasoning_lengths:
                avg_reasoning_length = np.mean(reasoning_lengths)
                # 推理长度在50-200词之间为最佳
                if avg_reasoning_length < 50:
                    reasoning_score = avg_reasoning_length / 50 * 0.7
                elif avg_reasoning_length <= 200:
                    reasoning_score = 0.7 + (avg_reasoning_length - 50) / 150 * 0.3
                else:
                    reasoning_score = 1.0 - (avg_reasoning_length - 200) / 200 * 0.2
        
        # 基于关键因素的数量和质量（使用total_rounds作为代理）
        key_factors_score = min(1.0, consensus_result.total_rounds / 5)
        
        # 基于风险识别的完整性（使用conflicts作为代理）
        risk_identification_score = min(1.0, len(consensus_result.conflicts) / 3) if consensus_result.conflicts else 0.5
        
        # 综合推理质量分数
        final_reasoning = (
            0.5 * reasoning_score +
            0.3 * key_factors_score +
            0.2 * risk_identification_score
        )
        
        return np.clip(final_reasoning, 0.0, 1.0)
    
    def _calculate_safety_assessment(
        self, 
        consensus_result: ConsensusResult, 
        patient_state: PatientState
    ) -> float:
        """计算安全性评估分数"""
        safety_score = 0.8  # 基础安全分数
        
        # 检查是否识别了重要风险（使用conflicts作为代理）
        identified_risks = consensus_result.conflicts if consensus_result.conflicts else []
        
        # 基于患者状态的风险评估
        patient_risk_factors = self._assess_patient_risk_factors(patient_state)
        
        # 检查是否识别了患者特定的风险
        risk_coverage = 0.0
        if patient_risk_factors:
            covered_risks = 0
            for risk in identified_risks:
                if any(patient_risk in risk.lower() for patient_risk in patient_risk_factors):
                    covered_risks += 1
            risk_coverage = covered_risks / len(patient_risk_factors)
        
        # 检查是否有后续随访计划（使用convergence_achieved作为代理）
        follow_up_bonus = 0.1 if consensus_result.convergence_achieved else 0.0
        
        # 检查是否考虑了替代方案（使用aggregated_scores的数量作为代理）
        alternative_bonus = min(0.1, len(consensus_result.aggregated_scores) * 0.05) if consensus_result.aggregated_scores else 0.0
        
        final_safety = safety_score + risk_coverage * 0.2 + follow_up_bonus + alternative_bonus
        
        return np.clip(final_safety, 0.0, 1.0)
    
    def _calculate_treatment_appropriateness(
        self,
        consensus_result: ConsensusResult,
        patient_state: PatientState,
        selected_action: TreatmentOption
    ) -> float:
        """计算治疗适宜性分数"""
        # 基础适宜性评估
        base_appropriateness = self._assess_treatment_patient_match(selected_action, patient_state)
        
        # 检查是否为推荐的治疗方案（使用aggregated_scores中的最高分方案）
        if consensus_result.aggregated_scores:
            best_treatment = max(consensus_result.aggregated_scores, key=consensus_result.aggregated_scores.get)
            recommendation_match = 1.0 if selected_action == best_treatment else 0.5
        else:
            recommendation_match = 0.5
        
        # 检查是否在替代方案中（使用aggregated_scores中的其他方案）
        alternative_consideration = 0.0
        if consensus_result.aggregated_scores and selected_action in consensus_result.aggregated_scores:
            alternative_consideration = 0.3
        
        # 综合适宜性分数
        final_appropriateness = (
            0.6 * base_appropriateness +
            0.3 * recommendation_match +
            0.1 * alternative_consideration
        )
        
        return np.clip(final_appropriateness, 0.0, 1.0)
    
    def _assess_patient_risk_factors(self, patient_state: PatientState) -> List[str]:
        """评估患者风险因素"""
        risk_factors = []
        
        if patient_state.age > 75:
            risk_factors.append("advanced_age")
        if patient_state.age < 18:
            risk_factors.append("pediatric")
        
        # 这里可以添加更多基于患者状态的风险因素评估
        # 例如：慢性疾病、药物过敏、器官功能等
        
        return risk_factors
    
    def _assess_treatment_patient_match(
        self, 
        treatment: TreatmentOption, 
        patient_state: PatientState
    ) -> float:
        """评估治疗方案与患者的匹配度"""
        # 简化的匹配度评估
        # 实际应该基于更复杂的临床指南和患者特征
        
        age = patient_state.age
        
        if treatment == TreatmentOption.SURGERY:
            if age > 80:
                return 0.3  # 高龄手术风险大
            elif age < 65:
                return 0.9  # 年轻患者手术耐受性好
            else:
                return 0.7
        
        elif treatment == TreatmentOption.CHEMOTHERAPY:
            if age > 75:
                return 0.6  # 高龄化疗需谨慎
            else:
                return 0.8
        
        elif treatment == TreatmentOption.RADIOTHERAPY:
            return 0.8  # 放疗相对安全
        
        elif treatment == TreatmentOption.IMMUNOTHERAPY:
            if age > 70:
                return 0.7
            else:
                return 0.9
        
        elif treatment == TreatmentOption.PALLIATIVE_CARE:
            if age > 80:
                return 0.9  # 高龄患者适合姑息治疗
            else:
                return 0.6
        
        elif treatment == TreatmentOption.WATCHFUL_WAITING:
            return 0.7  # 观察等待适中
        
        return 0.5  # 默认值
    
    def _apply_reward_modulation(
        self, 
        base_reward: float, 
        context: Optional[Dict[str, Any]]
    ) -> float:
        """应用奖励调节"""
        modulated_reward = base_reward
        
        if context:
            # 基于训练阶段的调节
            if "training_stage" in context:
                stage = context["training_stage"]
                if stage == "early":
                    # 早期训练阶段，增强奖励信号
                    modulated_reward *= 1.2
                elif stage == "late":
                    # 后期训练阶段，更严格的奖励
                    modulated_reward *= 0.9
            
            # 基于历史表现的调节
            if "recent_performance" in context:
                performance = context["recent_performance"]
                if performance < 0.5:
                    # 表现不佳时，给予鼓励
                    modulated_reward *= 1.1
        
        return np.clip(modulated_reward, -1.0, 1.0)
    
    def _generate_reward_explanation(
        self,
        consensus_quality: float,
        expert_agreement: float,
        role_diversity: float,
        confidence_level: float,
        clinical_reasoning: float,
        safety_assessment: float,
        treatment_appropriateness: float,
        patient_suitability: float,
        selected_action: TreatmentOption
    ) -> str:
        """生成奖励解释"""
        explanation_parts = []
        
        explanation_parts.append(f"治疗方案: {selected_action.value}")
        explanation_parts.append(f"共识质量: {consensus_quality:.3f}")
        explanation_parts.append(f"专家一致性: {expert_agreement:.3f}")
        explanation_parts.append(f"角色多样性: {role_diversity:.3f}")
        explanation_parts.append(f"置信度水平: {confidence_level:.3f}")
        explanation_parts.append(f"临床推理: {clinical_reasoning:.3f}")
        explanation_parts.append(f"安全性评估: {safety_assessment:.3f}")
        explanation_parts.append(f"治疗适宜性: {treatment_appropriateness:.3f}")
        explanation_parts.append(f"患者适用性: {patient_suitability:.3f}")
        
        return " | ".join(explanation_parts)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "component_weights": {
                RewardComponent.CONSENSUS_QUALITY.value: 0.20,
                RewardComponent.EXPERT_AGREEMENT.value: 0.20,
                RewardComponent.ROLE_DIVERSITY.value: 0.10,
                RewardComponent.CONFIDENCE_LEVEL.value: 0.15,
                RewardComponent.CLINICAL_REASONING.value: 0.15,
                RewardComponent.SAFETY_ASSESSMENT.value: 0.10,
                RewardComponent.TREATMENT_APPROPRIATENESS.value: 0.10,
            },
            "reward_range": (-1.0, 1.0),
            "normalization": True,
            "history_length": 1000
        }
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """获取奖励统计信息"""
        if not self.reward_history:
            return {"message": "No reward history available"}
        
        recent_rewards = self.reward_history[-100:]  # 最近100个奖励
        
        return {
            "total_rewards": len(self.reward_history),
            "average_total_reward": np.mean([r.total_reward for r in recent_rewards]),
            "reward_std": np.std([r.total_reward for r in recent_rewards]),
            "component_averages": {
                "consensus_quality": np.mean([r.consensus_quality for r in recent_rewards]),
                "expert_agreement": np.mean([r.expert_agreement for r in recent_rewards]),
                "role_diversity": np.mean([r.role_diversity for r in recent_rewards]),
                "confidence_level": np.mean([r.confidence_level for r in recent_rewards]),
                "clinical_reasoning": np.mean([r.clinical_reasoning for r in recent_rewards]),
                "safety_assessment": np.mean([r.safety_assessment for r in recent_rewards]),
                "treatment_appropriateness": np.mean([r.treatment_appropriateness for r in recent_rewards]),
                "patient_suitability": np.mean([r.patient_suitability for r in recent_rewards]),
            }
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        logger.info("Reward mapper configuration updated")
    
    def reset_history(self):
        """重置奖励历史"""
        self.reward_history = []
        logger.info("Reward history reset")
    
    def _calculate_patient_suitability(self, patient_state: PatientState, selected_action: TreatmentOption) -> float:
        """计算患者适用性分数"""
        suitability_score = 0.5  # 基础分数
        
        # 基于年龄的适用性
        age = patient_state.age
        if selected_action == TreatmentOption.SURGERY:
            if age < 70:
                suitability_score += 0.2
            elif age > 80:
                suitability_score -= 0.3
        elif selected_action == TreatmentOption.CHEMOTHERAPY:
            if age < 75:
                suitability_score += 0.1
            elif age > 85:
                suitability_score -= 0.2
        
        # 基于疾病分期的适用性
        stage = patient_state.stage
        if selected_action == TreatmentOption.SURGERY and stage in ["I", "II"]:
            suitability_score += 0.3
        elif selected_action == TreatmentOption.CHEMOTHERAPY and stage in ["III", "IV"]:
            suitability_score += 0.2
        elif selected_action == TreatmentOption.WATCHFUL_WAITING and stage == "I":
            suitability_score += 0.1
        
        # 基于合并症的适用性
        comorbidities = patient_state.comorbidities
        if selected_action == TreatmentOption.SURGERY and "cardiac_dysfunction" in comorbidities:
            suitability_score -= 0.2
        if selected_action == TreatmentOption.CHEMOTHERAPY and "kidney_disease" in comorbidities:
            suitability_score -= 0.1
        
        # 确保分数在合理范围内
        return max(0.0, min(1.0, suitability_score))