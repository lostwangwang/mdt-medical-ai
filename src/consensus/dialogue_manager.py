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
        # 获取相关医学知识 这里怎么检索? 从RAG系统中检索与患者状态相关的初始评估知识
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
        # 找到最推荐的治疗方案
        best_treatment = max(opinion.treatment_preferences.items(), key=lambda x: x[1])[
            0
        ]

        # 生成初始发言内容
        content = f"As the {agent.role.value}, I {self._get_recommendation_phrase(opinion.treatment_preferences[best_treatment])} {best_treatment.value}. "
        content += f"My reasoning: {opinion.reasoning}"

        if opinion.concerns:
            content += (
                f" However, I have concerns about: {', '.join(opinion.concerns[:2])}"
            )

        return DialogueMessage(
            role=agent.role,
            content=content,
            timestamp=datetime.now(),
            message_type="initial_opinion",
            referenced_roles=[],
            evidence_cited=[],
            treatment_focus=best_treatment,
        )

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
