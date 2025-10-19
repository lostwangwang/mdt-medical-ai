"""
多智能体对话管理器
文件路径: src/consensus/dialogue_manager.py
作者: 姚刚
功能: 管理医疗团队多智能体间的对话协商过程
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ..core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    DialogueMessage,
    DialogueRound,
    ConsensusResult,
)
from .role_agents import RoleAgent
from ..knowledge.rag_system import MedicalKnowledgeRAG

logger = logging.getLogger(__name__)


class MultiAgentDialogueManager:
    """多智能体对话管理器"""

    def __init__(self, rag_system: MedicalKnowledgeRAG):
        """
        初始化多智能体对话管理器

        Args:
            rag_system (MedicalKnowledgeRAG): 医学知识RAG系统实例
            agents (Dict[RoleType, RoleAgent]): 各角色智能体映射，键为角色枚举，值为对应智能体实例
            rag_system (MedicalKnowledgeRAG): 医学知识检索与生成系统，用于提供循证支持
            dialogue_rounds (List[DialogueRound]): 已完成的对话轮次列表，每轮包含多条消息与焦点治疗方案
            current_round (int): 当前对话轮次编号，从0开始计数
            max_rounds (int): 允许的最大对话轮次上限，超过则强制终止
            convergence_threshold (float): 立场收敛阈值，超过此比例的角色立场稳定即视为达成共识
        """
        self.agents = {role: RoleAgent(role) for role in RoleType}
        self.rag_system = rag_system
        self.dialogue_rounds = []
        self.current_round = 0
        self.max_rounds = 5
        self.convergence_threshold = 0.8


    def conduct_mdt_discussion(self, patient_state: PatientState) -> ConsensusResult:
        """进行MDT讨论"""
        logger.info(f"Starting MDT discussion for patient {patient_state.patient_id}")

        # 初始化对话
        self._initialize_discussion(patient_state)

        # 进行多轮对话
        while self.current_round < self.max_rounds and not self._check_convergence():

            self.current_round += 1
            logger.info(f"Starting dialogue round {self.current_round}")

            current_round = self._conduct_dialogue_round(patient_state)
            self.dialogue_rounds.append(current_round)

            # 更新智能体立场
            self._update_agent_stances(current_round)

            # 检查是否需要聚焦讨论
            if self.current_round > 2:
                self._focus_on_contentious_treatments(patient_state)

        # 生成最终共识结果
        final_result = self._generate_final_consensus(patient_state)

        logger.info(f"MDT discussion completed after {self.current_round} rounds")
        return final_result

    def _initialize_discussion(self, patient_state: PatientState) -> None:
        """初始化讨论"""
        # 获取相关医学知识 这里怎么检索? 从RAG系统中检索与患者状态相关的初始评估知识,要这里检索
        initial_knowledge = self.rag_system.retrieve_relevant_knowledge(
            patient_state, "initial_assessment"
        )

        # 生成各角色的初始意见
        # 这里怎么生成? 调用每个智能体的generate_initial_opinion方法，传入患者状态和初始知识
        initial_round = DialogueRound(
            round_number=0,
            messages=[],
            focus_treatment=None,
            consensus_status="discussing",
        )

        # 遍历每个角色智能体，生成其初始意见并构建首轮对话消息
        for role, agent in self.agents.items():
            opinion = agent.generate_initial_opinion(patient_state, initial_knowledge)

            # 生成初始发言
            initial_message = self._create_initial_message(
                agent, opinion, patient_state
            )
            initial_round.messages.append(initial_message)

        self.dialogue_rounds.append(initial_round)
        logger.info(
            f"Initialized discussion with {len(initial_round.messages)} initial opinions"
        )

    def _create_initial_message(
        self, agent: RoleAgent, opinion, patient_state: PatientState
    ) -> DialogueMessage:
        """创建初始消息"""
        # 智能选择治疗方案（不仅仅是最高分）
        focus_treatment = self._select_focus_treatment_for_role(
            agent, opinion, patient_state
        )

        # 尝试使用LLM生成个性化初始消息
        content = self._generate_llm_initial_message(agent, opinion, patient_state, focus_treatment)
        
        # 如果LLM生成失败，降级到模板化方法
        if not content:
            content = self._generate_template_initial_message(agent, opinion, focus_treatment)

        return DialogueMessage(
            role=agent.role,
            content=content,
            timestamp=datetime.now(),
            message_type="initial_opinion",
            referenced_roles=[],
            evidence_cited=[],
            treatment_focus=focus_treatment,
        )

    def _select_focus_treatment_for_role(
        self, agent: RoleAgent, opinion, patient_state: PatientState
    ) -> TreatmentOption:
        """为特定角色智能选择焦点治疗方案"""
        prefs = opinion.treatment_preferences
        
        # 1. 过滤掉明显不合适的治疗方案（评分过低）
        viable_treatments = {
            treatment: score for treatment, score in prefs.items() 
            if score > -0.5  # 排除强烈反对的方案
        }
        
        if not viable_treatments:
            # 如果所有方案都被强烈反对，选择最不反对的
            return max(prefs.items(), key=lambda x: x[1])[0]
        
        # 2. 考虑角色专业特长
        role_preferred_treatments = self._get_role_preferred_treatments(agent.role)
        
        # 3. 智能选择策略
        return self._apply_intelligent_selection_strategy(
            viable_treatments, role_preferred_treatments, patient_state, agent.role
        )

    def _get_role_preferred_treatments(self, role: RoleType) -> List[TreatmentOption]:
        """获取角色偏好的治疗方案类型"""
        role_preferences = {
            RoleType.ONCOLOGIST: [
                TreatmentOption.CHEMOTHERAPY, 
                TreatmentOption.RADIOTHERAPY, 
                TreatmentOption.SURGERY,
                TreatmentOption.IMMUNOTHERAPY
            ],
            RoleType.NURSE: [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.CHEMOTHERAPY,  # 护理角度关注
                TreatmentOption.RADIOTHERAPY
            ],
            RoleType.PSYCHOLOGIST: [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.WATCHFUL_WAITING,
                TreatmentOption.IMMUNOTHERAPY  # 相对温和
            ],
            RoleType.RADIOLOGIST: [
                TreatmentOption.RADIOTHERAPY,
                TreatmentOption.SURGERY,
                TreatmentOption.CHEMOTHERAPY
            ],
            RoleType.PATIENT_ADVOCATE: [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.WATCHFUL_WAITING,
                TreatmentOption.IMMUNOTHERAPY
            ],
            RoleType.NUTRITIONIST: [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.IMMUNOTHERAPY,
                TreatmentOption.CHEMOTHERAPY
            ],
            RoleType.REHABILITATION_THERAPIST: [
                TreatmentOption.SURGERY,
                TreatmentOption.RADIOTHERAPY,
                TreatmentOption.PALLIATIVE_CARE
            ]
        }
        return role_preferences.get(role, list(TreatmentOption))

    def _apply_intelligent_selection_strategy(
        self, 
        viable_treatments: Dict[TreatmentOption, float],
        role_preferred_treatments: List[TreatmentOption],
        patient_state: PatientState,
        role: RoleType
    ) -> TreatmentOption:
        """应用智能选择策略"""
        
        # 策略1: 如果有明显的高分治疗方案（>0.7），直接选择
        high_score_treatments = {
            t: s for t, s in viable_treatments.items() if s > 0.7
        }
        if high_score_treatments:
            return max(high_score_treatments.items(), key=lambda x: x[1])[0]
        
        # 策略2: 在角色偏好的治疗方案中选择评分最高的
        role_viable_treatments = {
            t: s for t, s in viable_treatments.items() 
            if t in role_preferred_treatments
        }
        if role_viable_treatments:
            return max(role_viable_treatments.items(), key=lambda x: x[1])[0]
        
        # 策略3: 考虑患者状态的特殊情况
        if patient_state.quality_of_life_score < 0.3:
            # 生活质量很差，优先考虑姑息治疗
            if TreatmentOption.PALLIATIVE_CARE in viable_treatments:
                return TreatmentOption.PALLIATIVE_CARE
        
        if patient_state.age > 80:
            # 高龄患者，优先考虑温和治疗
            gentle_treatments = [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.WATCHFUL_WAITING,
                TreatmentOption.IMMUNOTHERAPY
            ]
            for treatment in gentle_treatments:
                if treatment in viable_treatments:
                    return treatment
        
        # 策略4: 如果评分接近，选择争议性较小的方案
        max_score = max(viable_treatments.values())
        close_treatments = {
            t: s for t, s in viable_treatments.items() 
            if abs(s - max_score) < 0.2
        }
        
        if len(close_treatments) > 1:
            # 选择通常争议较小的治疗方案
            preference_order = [
                TreatmentOption.IMMUNOTHERAPY,
                TreatmentOption.RADIOTHERAPY,
                TreatmentOption.CHEMOTHERAPY,
                TreatmentOption.SURGERY,
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.WATCHFUL_WAITING
            ]
            
            for preferred in preference_order:
                if preferred in close_treatments:
                    return preferred
        
        # 策略5: 默认选择评分最高的
        return max(viable_treatments.items(), key=lambda x: x[1])[0]

    def _generate_llm_initial_message(
        self, agent: RoleAgent, opinion, patient_state: PatientState, focus_treatment: TreatmentOption
    ) -> str:
        """使用LLM生成个性化初始消息"""
        if not hasattr(agent, 'llm_interface') or not agent.llm_interface:
            return ""
            
        try:
            # 使用专门的治疗推理方法生成基础推理
            if hasattr(agent.llm_interface, 'generate_treatment_reasoning'):
                reasoning = agent.llm_interface.generate_treatment_reasoning(
                    patient_state=patient_state,
                    role=agent.role,
                    treatment_option=focus_treatment,
                    knowledge_context={}
                )
                
                if reasoning and len(reasoning.strip()) > 0:
                    # 基于推理生成MDT发言格式的消息
                    content = self._format_reasoning_as_mdt_message(
                        agent, reasoning, focus_treatment, opinion
                    )
                    logger.debug(f"Generated LLM initial message for {agent.role.value}: {content[:100]}...")
                    return content
                    
        except Exception as e:
            logger.warning(f"LLM initial message generation failed for {agent.role}: {e}")
            
        return ""

    def _format_reasoning_as_mdt_message(
        self, agent: RoleAgent, reasoning: str, focus_treatment: TreatmentOption, opinion
    ) -> str:
        """将LLM推理格式化为MDT发言消息"""
        
        # 获取推荐强度
        treatment_score = opinion.treatment_preferences.get(focus_treatment, 0.0)
        recommendation_phrase = self._get_recommendation_phrase(treatment_score)
        
        # 构建MDT发言格式
        content = f"作为{agent.role.value}，我{recommendation_phrase}{focus_treatment.value}。\n\n"
        content += f"我的专业分析：{reasoning}"
        
        # 添加关注事项
        if opinion.concerns:
            content += f"\n\n需要特别关注的问题：{', '.join(opinion.concerns[:3])}"
            
        return content



    def _generate_template_initial_message(
        self, agent: RoleAgent, opinion, focus_treatment: TreatmentOption
    ) -> str:
        """生成模板化初始消息（降级方案）"""
        content = f"As the {agent.role.value}, I {self._get_recommendation_phrase(opinion.treatment_preferences[focus_treatment])} {focus_treatment.value}. "
        content += f"My reasoning: {opinion.reasoning}"

        if opinion.concerns:
            content += (
                f" However, I have concerns about: {', '.join(opinion.concerns[:2])}"
            )
            
        return content

    def _get_recommendation_phrase(self, score: float) -> str:
        """获取推荐措辞"""
        if score > 0.7:
            return "strongly recommend"
        elif score > 0.3:
            return "recommend"
        elif score > -0.3:
            return "have mixed feelings about"
        elif score > -0.7:
            return "have concerns about"
        else:
            return "strongly advise against"

    def _conduct_dialogue_round(self, patient_state: PatientState) -> DialogueRound:
        """进行一轮对话"""
        current_round = DialogueRound(
            round_number=self.current_round,
            messages=[],
            focus_treatment=self._select_focus_treatment(),
            consensus_status="discussing",
        )

        logger.info(
            f"Round {self.current_round} focusing on: {current_round.focus_treatment.value}"
        )

        # 获取该轮的相关知识
        round_knowledge = self.rag_system.retrieve_relevant_knowledge(
            patient_state, "treatment_discussion", current_round.focus_treatment
        )

        # 确定发言顺序（按争议程度）
        speaking_order = self._determine_speaking_order(current_round.focus_treatment)

        # 各角色依次发言
        for role in speaking_order:
            agent = self.agents[role]

            # 获取当前对话历史
            all_previous_messages = []
            for round in self.dialogue_rounds:
                all_previous_messages.extend(round.messages)
            all_previous_messages.extend(current_round.messages)

            # 生成回应
            response = agent.generate_dialogue_response(
                patient_state,
                round_knowledge,
                all_previous_messages,
                current_round.focus_treatment,
            )

            current_round.messages.append(response)

            logger.debug(f"{role.value} responded: {response.content[:100]}...")

        return current_round

    def _select_focus_treatment(self) -> TreatmentOption:
        """选择焦点治疗方案"""
        if not self.dialogue_rounds:
            return TreatmentOption.SURGERY  # 默认开始话题

        # 分析前一轮的争议点
        last_round = self.dialogue_rounds[-1]
        treatment_mentions = {}

        for message in last_round.messages:
            treatment = message.treatment_focus
            if treatment not in treatment_mentions:
                treatment_mentions[treatment] = 0
            treatment_mentions[treatment] += 1

        # 选择提及最多或争议最大的治疗方案
        if treatment_mentions:
            return max(treatment_mentions.items(), key=lambda x: x[1])[0]

        # 轮换讨论不同治疗方案
        treatments = list(TreatmentOption)
        return treatments[self.current_round % len(treatments)]

    def _determine_speaking_order(
        self, focus_treatment: TreatmentOption
    ) -> List[RoleType]:
        """确定发言顺序"""
        # 基于角色对治疗方案的相关性确定顺序
        relevance_scores = {}

        for role, agent in self.agents.items():
            current_stance = agent.current_stance.get(focus_treatment, 0)
            relevance_scores[role] = abs(current_stance)  # 立场越强烈越早发言

        # 按相关性排序
        sorted_roles = sorted(
            relevance_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [role for role, _ in sorted_roles]

    def _update_agent_stances(self, round_data: DialogueRound) -> None:
        """更新智能体立场"""
        for role, agent in self.agents.items():
            agent.update_stance_based_on_dialogue(round_data.messages)

    def _check_convergence(self) -> bool:
        """检查是否收敛"""
        if len(self.dialogue_rounds) < 2:
            return False

        # 计算立场稳定性
        stable_agents = 0

        for role, agent in self.agents.items():
            current_stances = agent.current_stance
            strong_stances = [abs(score) > 0.6 for score in current_stances.values()]

            if (
                len(strong_stances) > 0
                and sum(strong_stances) / len(strong_stances) > 0.7
            ):
                stable_agents += 1

        # 如果大多数角色都有明确且稳定的立场，认为已收敛
        convergence_ratio = stable_agents / len(self.agents)

        logger.debug(
            f"Convergence check: {convergence_ratio:.2f} (threshold: {self.convergence_threshold})"
        )

        return convergence_ratio >= self.convergence_threshold

    def _focus_on_contentious_treatments(self, patient_state: PatientState) -> None:
        """聚焦争议治疗方案"""
        # 识别争议最大的治疗方案
        treatment_variances = {}

        for treatment in TreatmentOption:
            scores = []
            for agent in self.agents.values():
                score = agent.current_stance.get(treatment, 0)
                scores.append(score)

            if len(scores) > 1:
                treatment_variances[treatment] = np.var(scores)

        # 如果方差过大，在下一轮重点讨论
        if treatment_variances:
            most_contentious = max(treatment_variances.items(), key=lambda x: x[1])
            if most_contentious[1] > 0.5:  # 高争议阈值
                logger.info(
                    f"High contention detected for: {most_contentious[0].value} (variance: {most_contentious[1]:.3f})"
                )

    def _generate_final_consensus(self, patient_state: PatientState) -> ConsensusResult:
        """生成最终共识结果"""
        from .consensus_matrix import ConsensusMatrix

        # 收集最终立场
        final_opinions = {}
        for role, agent in self.agents.items():
            final_opinion = agent.generate_initial_opinion(patient_state, {})
            # 使用对话后的立场覆盖初始立场
            final_opinion.treatment_preferences = agent.current_stance.copy()
            final_opinion.reasoning = (
                f"Final position after {self.current_round} rounds of discussion"
            )
            final_opinions[role] = final_opinion

        # 使用共识矩阵系统处理最终结果
        consensus_system = ConsensusMatrix()
        consensus_system.agents = self.agents  # 使用更新后的智能体

        # 生成详细的对话摘要
        dialogue_summary = self._generate_dialogue_summary()

        # 构建最终共识结果
        import pandas as pd

        # 构建共识矩阵
        treatments = list(TreatmentOption)
        roles = list(RoleType)

        matrix_data = np.zeros((len(treatments), len(roles)))

        for i, treatment in enumerate(treatments):
            for j, role in enumerate(roles):
                if role in final_opinions:
                    score = final_opinions[role].treatment_preferences.get(
                        treatment, 0.0
                    )
                    matrix_data[i, j] = score

        consensus_matrix = pd.DataFrame(
            matrix_data,
            index=[t.value for t in treatments],
            columns=[r.value for r in roles],
        )

        # 计算聚合评分
        aggregated_scores = {}
        for treatment in TreatmentOption:
            scores = []
            weights = []

            for role, opinion in final_opinions.items():
                score = opinion.treatment_preferences.get(treatment, 0.0)
                weight = opinion.confidence
                scores.append(score)
                weights.append(weight)

            if scores:
                weighted_score = np.average(scores, weights=weights)
                aggregated_scores[treatment] = weighted_score

        # 识别冲突与一致性
        conflicts = self._identify_final_conflicts(final_opinions)
        agreements = self._identify_final_agreements(final_opinions)

        return ConsensusResult(
            consensus_matrix=consensus_matrix,
            role_opinions=final_opinions,
            aggregated_scores=aggregated_scores,
            conflicts=conflicts,
            agreements=agreements,
            dialogue_summary=dialogue_summary,
            timestamp=datetime.now(),
            convergence_achieved=self._check_convergence(),
            total_rounds=self.current_round,
        )

    def _generate_dialogue_summary(self) -> Dict[str, Any]:
        """生成对话摘要"""
        summary = {
            "total_messages": sum(
                len(round.messages) for round in self.dialogue_rounds
            ),
            "key_topics": [],
            "major_agreements": [],
            "persistent_disagreements": [],
            "evidence_cited": [],
        }

        # 统计话题频率
        topic_counts = {}
        all_evidence = set()

        for round in self.dialogue_rounds:
            for message in round.messages:
                treatment = message.treatment_focus
                if treatment:
                    topic_counts[treatment] = topic_counts.get(treatment, 0) + 1

                all_evidence.update(message.evidence_cited)

        # 提取关键信息
        summary["key_topics"] = sorted(
            topic_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        summary["evidence_cited"] = list(all_evidence)[:5]

        return summary

    def _identify_final_conflicts(
        self, final_opinions: Dict[RoleType, Any]
    ) -> List[Dict[str, Any]]:
        """识别最终冲突"""
        conflicts = []

        for treatment in TreatmentOption:
            scores = [
                opinion.treatment_preferences.get(treatment, 0.0)
                for opinion in final_opinions.values()
            ]

            if len(scores) > 1:
                variance = np.var(scores)
                if variance > 0.5:  # 高方差表示冲突
                    conflicts.append(
                        {
                            "treatment": treatment,
                            "variance": variance,
                            "min_score": min(scores),
                            "max_score": max(scores),
                            "conflicting_roles": self._find_conflicting_roles(
                                treatment, final_opinions
                            ),
                        }
                    )

        return conflicts

    def _identify_final_agreements(
        self, final_opinions: Dict[RoleType, Any]
    ) -> List[Dict[str, Any]]:
        """识别最终一致意见"""
        agreements = []

        for treatment in TreatmentOption:
            scores = [
                opinion.treatment_preferences.get(treatment, 0.0)
                for opinion in final_opinions.values()
            ]

            if len(scores) > 1:
                variance = np.var(scores)
                mean_score = np.mean(scores)

                if variance < 0.2 and abs(mean_score) > 0.3:  # 低方差且有明确倾向
                    agreements.append(
                        {
                            "treatment": treatment,
                            "consensus_score": mean_score,
                            "agreement_strength": 1.0 - variance,
                            "unanimous": all(s > 0.5 for s in scores)
                            or all(s < -0.5 for s in scores),
                        }
                    )

        return agreements

    def _find_conflicting_roles(
        self, treatment: TreatmentOption, final_opinions: Dict[RoleType, Any]
    ) -> List[str]:
        """找出在特定治疗上有冲突的角色"""
        scores_by_role = {
            role.value: opinion.treatment_preferences.get(treatment, 0.0)
            for role, opinion in final_opinions.items()
        }

        mean_score = np.mean(list(scores_by_role.values()))
        conflicting_roles = [
            role
            for role, score in scores_by_role.items()
            if abs(score - mean_score) > 0.5
        ]

        return conflicting_roles

    def get_dialogue_transcript(self) -> str:
        """获取完整对话记录"""
        transcript = f"=== MDT Discussion Transcript ===\n"
        transcript += f"Total Rounds: {len(self.dialogue_rounds)}\n\n"

        for round in self.dialogue_rounds:
            transcript += f"--- Round {round.round_number} ---\n"
            if round.focus_treatment:
                transcript += f"Focus: {round.focus_treatment.value}\n"

            for message in round.messages:
                transcript += f"\n{message.role.value}: {message.content}\n"

            transcript += "\n"

        return transcript
