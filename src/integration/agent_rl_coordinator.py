"""
智能体与强化学习协调器
文件路径: src/integration/agent_rl_coordinator.py
作者: AI Assistant
功能: 协调五个智能体与RL模块之间的双向交互
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    ConsensusResult,
    RLState,
    RLAction,
    RLReward,
)
from ..consensus.dialogue_manager import MultiAgentDialogueManager
from ..consensus.role_agents import RoleAgent
from ..rl.rl_environment import MDTReinforcementLearning
from ..knowledge.rag_system import MedicalKnowledgeRAG

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """交互模式"""
    TRAINING = "training"  # 训练模式：智能体指导RL学习
    INFERENCE = "inference"  # 推理模式：RL辅助智能体决策
    COLLABORATIVE = "collaborative"  # 协作模式：双向交互优化


@dataclass
class AgentRLInteractionResult:
    """智能体-RL交互结果"""
    patient_id: str
    interaction_mode: InteractionMode
    agent_consensus: ConsensusResult
    rl_recommendation: Dict[str, Any]
    final_decision: TreatmentOption
    confidence_score: float
    interaction_metrics: Dict[str, float]
    timestamp: datetime


class AgentRLCoordinator:
    """智能体与强化学习协调器"""
    
    def __init__(
        self,
        rag_system: MedicalKnowledgeRAG,
        interaction_mode: InteractionMode = InteractionMode.COLLABORATIVE
    ):
        """
        初始化协调器
        
        Args:
            rag_system: 医学知识RAG系统
            interaction_mode: 交互模式
        """
        self.rag_system = rag_system
        self.interaction_mode = interaction_mode
        
        # 初始化组件
        self.dialogue_manager = MultiAgentDialogueManager(rag_system)
        self.rl_environment = MDTReinforcementLearning(consensus_system=self)
        
        # 交互历史
        self.interaction_history = []
        
        # 配置参数
        self.config = {
            "agent_weight": 0.6,  # 智能体决策权重
            "rl_weight": 0.4,     # RL决策权重
            "confidence_threshold": 0.8,  # 置信度阈值
            "consensus_threshold": 0.7,   # 共识阈值
            "max_iterations": 3,          # 最大迭代次数
        }
        
        logger.info(f"AgentRLCoordinator initialized in {interaction_mode.value} mode")
    
    def coordinate_decision(
        self, 
        patient_state: PatientState,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentRLInteractionResult:
        """
        协调智能体和RL进行医疗决策
        
        Args:
            patient_state: 患者状态
            context: 额外上下文信息
            
        Returns:
            交互结果
        """
        logger.info(f"Starting decision coordination for patient {patient_state.patient_id}")
        
        if self.interaction_mode == InteractionMode.TRAINING:
            return self._training_mode_interaction(patient_state, context)
        elif self.interaction_mode == InteractionMode.INFERENCE:
            return self._inference_mode_interaction(patient_state, context)
        else:  # COLLABORATIVE
            return self._collaborative_mode_interaction(patient_state, context)
    
    def _training_mode_interaction(
        self, 
        patient_state: PatientState, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentRLInteractionResult:
        """
        训练模式：智能体作为专家指导RL学习
        """
        logger.info("Executing training mode interaction")
        
        # 1. 智能体生成专家决策
        agent_consensus = self.dialogue_manager.conduct_mdt_discussion(patient_state)
        
        # 2. 将专家决策作为监督信号训练RL
        rl_state = self._create_rl_state(patient_state, agent_consensus)
        expert_action = self._map_treatment_to_action(agent_consensus.recommended_treatment)
        
        # 3. RL环境学习专家决策
        rl_reward = self._calculate_expert_guided_reward(
            expert_action, agent_consensus, patient_state
        )
        
        # 4. 记录训练数据
        training_data = {
            "state": rl_state,
            "expert_action": expert_action,
            "reward": rl_reward,
            "consensus_score": agent_consensus.consensus_score
        }
        
        # 5. 生成交互结果
        interaction_result = AgentRLInteractionResult(
            patient_id=patient_state.patient_id,
            interaction_mode=self.interaction_mode,
            agent_consensus=agent_consensus,
            rl_recommendation={"action": expert_action, "training_data": training_data},
            final_decision=agent_consensus.recommended_treatment,
            confidence_score=agent_consensus.consensus_score,
            interaction_metrics=self._calculate_training_metrics(agent_consensus, rl_reward),
            timestamp=datetime.now()
        )
        
        self.interaction_history.append(interaction_result)
        return interaction_result
    
    def _inference_mode_interaction(
        self, 
        patient_state: PatientState, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentRLInteractionResult:
        """
        推理模式：RL辅助智能体决策
        """
        logger.info("Executing inference mode interaction")
        
        # 1. RL生成初始建议
        rl_state = self._create_rl_state(patient_state)
        rl_action = self._get_rl_recommendation(rl_state)
        rl_treatment = self._map_action_to_treatment(rl_action)
        
        # 2. 将RL建议提供给智能体
        enhanced_context = {
            "rl_recommendation": rl_treatment,
            "rl_confidence": self._calculate_rl_confidence(rl_state, rl_action),
            **(context or {})
        }
        
        # 3. 智能体基于RL建议进行讨论
        agent_consensus = self._conduct_rl_guided_discussion(
            patient_state, enhanced_context
        )
        
        # 4. 融合决策
        final_decision, confidence = self._fuse_decisions(
            agent_consensus.recommended_treatment,
            rl_treatment,
            agent_consensus.consensus_score,
            enhanced_context["rl_confidence"]
        )
        
        # 5. 生成交互结果
        interaction_result = AgentRLInteractionResult(
            patient_id=patient_state.patient_id,
            interaction_mode=self.interaction_mode,
            agent_consensus=agent_consensus,
            rl_recommendation={
                "treatment": rl_treatment,
                "action": rl_action,
                "confidence": enhanced_context["rl_confidence"]
            },
            final_decision=final_decision,
            confidence_score=confidence,
            interaction_metrics=self._calculate_inference_metrics(
                agent_consensus, rl_treatment, final_decision
            ),
            timestamp=datetime.now()
        )
        
        self.interaction_history.append(interaction_result)
        return interaction_result
    
    def _collaborative_mode_interaction(
        self, 
        patient_state: PatientState, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentRLInteractionResult:
        """
        协作模式：智能体和RL双向迭代优化
        """
        logger.info("Executing collaborative mode interaction")
        
        iteration = 0
        current_consensus = None
        current_rl_recommendation = None
        
        while iteration < self.config["max_iterations"]:
            iteration += 1
            logger.info(f"Collaborative iteration {iteration}")
            
            # 1. 智能体讨论（考虑之前的RL建议）
            if current_rl_recommendation:
                enhanced_context = {
                    "rl_recommendation": current_rl_recommendation,
                    "iteration": iteration,
                    **(context or {})
                }
                current_consensus = self._conduct_rl_guided_discussion(
                    patient_state, enhanced_context
                )
            else:
                current_consensus = self.dialogue_manager.conduct_mdt_discussion(patient_state)
            
            # 2. RL基于智能体共识生成建议
            rl_state = self._create_rl_state(patient_state, current_consensus)
            rl_action = self._get_rl_recommendation(rl_state)
            current_rl_recommendation = self._map_action_to_treatment(rl_action)
            
            # 3. 检查收敛
            if self._check_convergence(current_consensus, current_rl_recommendation):
                logger.info(f"Convergence achieved at iteration {iteration}")
                break
        
        # 4. 生成最终决策
        # 获取智能体共识的最佳治疗方案
        if current_consensus.aggregated_scores:
            best_agent_treatment = max(current_consensus.aggregated_scores, key=current_consensus.aggregated_scores.get)
            best_agent_score = current_consensus.aggregated_scores[best_agent_treatment]
        else:
            best_agent_treatment = TreatmentOption.WATCHFUL_WAITING
            best_agent_score = 0.5
            
        final_decision, confidence = self._fuse_decisions(
            best_agent_treatment,
            current_rl_recommendation,
            best_agent_score,
            self._calculate_rl_confidence(rl_state, rl_action)
        )
        
        # 5. 生成交互结果
        interaction_result = AgentRLInteractionResult(
            patient_id=patient_state.patient_id,
            interaction_mode=self.interaction_mode,
            agent_consensus=current_consensus,
            rl_recommendation={
                "treatment": current_rl_recommendation,
                "action": rl_action,
                "iterations": iteration
            },
            final_decision=final_decision,
            confidence_score=confidence,
            interaction_metrics=self._calculate_collaborative_metrics(
                current_consensus, current_rl_recommendation, iteration
            ),
            timestamp=datetime.now()
        )
        
        self.interaction_history.append(interaction_result)
        return interaction_result
    
    def generate_consensus(self, patient_state: PatientState) -> ConsensusResult:
        """
        为RL环境生成共识结果（实现consensus_system接口）
        """
        return self.dialogue_manager.conduct_mdt_discussion(patient_state)
    
    def _create_rl_state(
        self, 
        patient_state: PatientState, 
        consensus_result: Optional[ConsensusResult] = None
    ) -> np.ndarray:
        """创建RL状态向量"""
        if consensus_result is None:
            # 生成模拟共识用于状态创建
            consensus_result = self._generate_mock_consensus()
        
        return self.rl_environment.create_state_vector(patient_state, consensus_result)
    
    def _get_rl_recommendation(self, state: np.ndarray) -> int:
        """获取RL推荐动作"""
        # 这里应该调用训练好的RL模型
        # 暂时使用随机策略作为占位符
        return np.random.choice(len(TreatmentOption))
    
    def _map_treatment_to_action(self, treatment: TreatmentOption) -> int:
        """将治疗方案映射为RL动作"""
        treatment_list = list(TreatmentOption)
        return treatment_list.index(treatment)
    
    def _map_action_to_treatment(self, action: int) -> TreatmentOption:
        """将RL动作映射为治疗方案"""
        treatment_list = list(TreatmentOption)
        return treatment_list[action]
    
    def _calculate_expert_guided_reward(
        self, 
        action: int, 
        consensus: ConsensusResult, 
        patient_state: PatientState
    ) -> float:
        """计算专家指导的奖励"""
        # 基于专家共识质量计算奖励
        base_reward = consensus.consensus_score
        
        # 考虑治疗方案的适宜性
        treatment = self._map_action_to_treatment(action)
        suitability = self._calculate_treatment_suitability(treatment, patient_state)
        
        return base_reward * 0.7 + suitability * 0.3
    
    def _calculate_treatment_suitability(
        self, 
        treatment: TreatmentOption, 
        patient_state: PatientState
    ) -> float:
        """计算治疗方案适宜性"""
        # 简化的适宜性计算
        # 实际应该基于患者特征和治疗方案特点
        return np.random.uniform(0.5, 1.0)  # 占位符
    
    def _calculate_rl_confidence(self, state: np.ndarray, action: int) -> float:
        """计算RL推荐的置信度"""
        # 这里应该基于模型的不确定性估计
        # 暂时返回固定值
        return 0.75
    
    def _conduct_rl_guided_discussion(
        self, 
        patient_state: PatientState, 
        context: Dict[str, Any]
    ) -> ConsensusResult:
        """进行RL指导的智能体讨论"""
        # 修改智能体的初始偏好以考虑RL建议
        # 这里需要扩展dialogue_manager以支持外部建议
        return self.dialogue_manager.conduct_mdt_discussion(patient_state)
    
    def _fuse_decisions(
        self, 
        agent_decision: TreatmentOption,
        rl_decision: TreatmentOption,
        agent_confidence: float,
        rl_confidence: float
    ) -> Tuple[TreatmentOption, float]:
        """融合智能体和RL的决策"""
        if agent_decision == rl_decision:
            # 决策一致，提高置信度
            final_confidence = min(1.0, (agent_confidence + rl_confidence) / 2 + 0.1)
            return agent_decision, final_confidence
        
        # 决策不一致，基于权重和置信度选择
        agent_score = agent_confidence * self.config["agent_weight"]
        rl_score = rl_confidence * self.config["rl_weight"]
        
        if agent_score > rl_score:
            return agent_decision, agent_confidence * 0.9  # 降低置信度
        else:
            return rl_decision, rl_confidence * 0.9
    
    def _check_convergence(
        self, 
        consensus: ConsensusResult, 
        rl_recommendation: TreatmentOption
    ) -> bool:
        """检查智能体和RL是否收敛"""
        # 找到聚合分数最高的治疗方案
        if consensus.aggregated_scores:
            best_treatment = max(consensus.aggregated_scores, key=consensus.aggregated_scores.get)
            best_score = consensus.aggregated_scores[best_treatment]
            return (
                best_treatment == rl_recommendation and
                best_score > self.config["consensus_threshold"]
            )
        return False
    
    def _generate_mock_consensus(self) -> ConsensusResult:
        """生成模拟共识结果"""
        from ..core.data_models import RoleOpinion
        
        return ConsensusResult(
            recommended_treatment=TreatmentOption.CHEMOTHERAPY,
            consensus_score=0.8,
            confidence_level=0.75,
            role_opinions={
                RoleType.ONCOLOGIST: RoleOpinion(
                    role=RoleType.ONCOLOGIST,
                    recommended_treatment=TreatmentOption.CHEMOTHERAPY,
                    confidence=0.9,
                    reasoning="Mock reasoning"
                )
            },
            discussion_summary="Mock discussion",
            key_factors=["factor1", "factor2"],
            risks_identified=["risk1"],
            alternative_options=[TreatmentOption.SURGERY],
            follow_up_required=True,
            timestamp=datetime.now()
        )
    
    def _calculate_training_metrics(
        self, 
        consensus: ConsensusResult, 
        reward: float
    ) -> Dict[str, float]:
        """计算训练模式指标"""
        return {
            "consensus_quality": consensus.consensus_score,
            "expert_reward": reward,
            "agreement_level": len(consensus.role_opinions) / len(RoleType)
        }
    
    def _calculate_inference_metrics(
        self, 
        consensus: ConsensusResult,
        rl_treatment: TreatmentOption,
        final_decision: TreatmentOption
    ) -> Dict[str, float]:
        """计算推理模式指标"""
        agent_rl_agreement = 1.0 if consensus.recommended_treatment == rl_treatment else 0.0
        decision_stability = 1.0 if final_decision == consensus.recommended_treatment else 0.5
        
        return {
            "agent_rl_agreement": agent_rl_agreement,
            "decision_stability": decision_stability,
            "consensus_quality": consensus.consensus_score
        }
    
    def _calculate_collaborative_metrics(
        self,
        consensus: ConsensusResult,
        rl_treatment: TreatmentOption,
        iterations: int
    ) -> Dict[str, float]:
        """计算协作模式指标"""
        convergence_efficiency = 1.0 - (iterations - 1) / self.config["max_iterations"]
        
        # 获取智能体共识的最佳治疗方案
        if consensus.aggregated_scores:
            best_agent_treatment = max(consensus.aggregated_scores, key=consensus.aggregated_scores.get)
            best_agent_score = consensus.aggregated_scores[best_agent_treatment]
            final_agreement = 1.0 if best_agent_treatment == rl_treatment else 0.0
        else:
            best_agent_score = 0.5
            final_agreement = 0.0
        
        return {
            "convergence_efficiency": convergence_efficiency,
            "final_agreement": final_agreement,
            "consensus_quality": best_agent_score,
            "iterations_used": iterations
        }
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """获取交互总结"""
        if not self.interaction_history:
            return {"message": "No interactions recorded"}
        
        recent_interactions = self.interaction_history[-10:]  # 最近10次交互
        
        return {
            "total_interactions": len(self.interaction_history),
            "current_mode": self.interaction_mode.value,
            "average_confidence": np.mean([i.confidence_score for i in recent_interactions]),
            "decision_distribution": self._analyze_decision_distribution(recent_interactions),
            "performance_trends": self._analyze_performance_trends(recent_interactions)
        }
    
    def _analyze_decision_distribution(
        self, 
        interactions: List[AgentRLInteractionResult]
    ) -> Dict[str, float]:
        """分析决策分布"""
        decisions = [i.final_decision.value for i in interactions]
        unique_decisions = list(set(decisions))
        
        distribution = {}
        for decision in unique_decisions:
            distribution[decision] = decisions.count(decision) / len(decisions)
        
        return distribution
    
    def _analyze_performance_trends(
        self, 
        interactions: List[AgentRLInteractionResult]
    ) -> Dict[str, float]:
        """分析性能趋势"""
        if len(interactions) < 2:
            return {"trend": "insufficient_data"}
        
        confidences = [i.confidence_score for i in interactions]
        
        return {
            "confidence_trend": (confidences[-1] - confidences[0]) / len(confidences),
            "stability": np.std(confidences),
            "improvement_rate": np.mean(np.diff(confidences))
        }
    
    def set_interaction_mode(self, mode: InteractionMode):
        """设置交互模式"""
        self.interaction_mode = mode
        logger.info(f"Interaction mode changed to {mode.value}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置参数"""
        self.config.update(new_config)
        logger.info("Configuration updated")