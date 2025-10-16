"""
共识矩阵系统
文件路径: src/consensus/consensus_matrix.py
作者: 姚刚
功能: 构建和分析医疗团队的治疗方案共识矩阵
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ..core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    RoleOpinion,
    ConsensusResult,
)
from .role_agents import RoleAgent
from ..knowledge.rag_system import MedicalKnowledgeRAG

logger = logging.getLogger(__name__)


class ConsensusMatrix:
    """共识矩阵系统"""

    def __init__(self):
        self.agents = {role: RoleAgent(role) for role in RoleType}
        self.rag_system = MedicalKnowledgeRAG()

    def generate_consensus(
        self, patient_state: PatientState, use_dialogue: bool = False
    ) -> ConsensusResult:
        """生成共识矩阵"""
        logger.info(
            f"Generating consensus matrix for patient {patient_state.patient_id}"
        )

        if use_dialogue:
            # 使用对话系统生成更丰富的共识
            return self._generate_dialogue_based_consensus(patient_state)
        else:
            # 传统的直接共识生成
            return self._generate_direct_consensus(patient_state)

    def _generate_direct_consensus(
        self, patient_state: PatientState
    ) -> ConsensusResult:
        """直接生成共识（不经过对话）"""
        # 1. RAG知识检索
        relevant_knowledge = self.rag_system.retrieve_relevant_knowledge(
            patient_state, "treatment_recommendation"
        )

        # 2. 收集各角色意见
        role_opinions = {}
        for role, agent in self.agents.items():
            opinion = agent.generate_initial_opinion(patient_state, relevant_knowledge)
            role_opinions[role] = opinion

        # 3. 构建共识矩阵
        consensus_matrix = self._build_consensus_matrix(role_opinions)

        # 4. 计算综合评分
        aggregated_scores = self._aggregate_scores(role_opinions)

        # 5. 识别冲突与一致性
        conflicts = self._identify_conflicts(role_opinions)
        agreements = self._identify_agreements(role_opinions)

        return ConsensusResult(
            consensus_matrix=consensus_matrix,
            role_opinions=role_opinions,
            aggregated_scores=aggregated_scores,
            conflicts=conflicts,
            agreements=agreements,
            dialogue_summary=None,
            timestamp=datetime.now(),
            convergence_achieved=True,
            total_rounds=1,
        )

    def _generate_dialogue_based_consensus(
        self, patient_state: PatientState
    ) -> ConsensusResult:
        """基于对话生成共识"""
        from .dialogue_manager import MultiAgentDialogueManager

        dialogue_manager = MultiAgentDialogueManager(self.rag_system)
        return dialogue_manager.conduct_mdt_discussion(patient_state)

    def _build_consensus_matrix(
        self, role_opinions: Dict[RoleType, RoleOpinion]
    ) -> pd.DataFrame:
        """构建共识矩阵"""
        treatments = list(TreatmentOption)
        roles = list(RoleType)

        matrix_data = np.zeros((len(treatments), len(roles)))

        for i, treatment in enumerate(treatments):
            for j, role in enumerate(roles):
                if role in role_opinions:
                    score = role_opinions[role].treatment_preferences.get(
                        treatment, 0.0
                    )
                    matrix_data[i, j] = score

        consensus_matrix = pd.DataFrame(
            matrix_data,
            index=[t.value for t in treatments],
            columns=[r.value for r in roles],
        )
        print("共识矩阵", consensus_matrix)
        logger.info("Consensus matrix generated successfully", consensus_matrix)
        return consensus_matrix

    def _aggregate_scores(
        self, role_opinions: Dict[RoleType, RoleOpinion]
    ) -> Dict[TreatmentOption, float]:
        """聚合评分"""
        aggregated = {}

        for treatment in TreatmentOption:
            scores = []
            weights = []

            for role, opinion in role_opinions.items():
                score = opinion.treatment_preferences.get(treatment, 0.0)
                weight = opinion.confidence
                scores.append(score)
                weights.append(weight)

            if scores:
                # 检查权重和是否为零，如果是则使用简单平均
                if sum(weights) == 0:
                    weighted_score = np.mean(scores)
                    logger.warning(f"All weights are zero for {treatment.value}, using simple average")
                else:
                    # 加权平均
                    weighted_score = np.average(scores, weights=weights)
                
                aggregated[treatment] = weighted_score

                logger.debug(
                    f"Treatment {treatment.value}: weighted score = {weighted_score:.3f}"
                )

        return aggregated

    def _identify_conflicts(
        self, role_opinions: Dict[RoleType, RoleOpinion]
    ) -> List[Dict[str, Any]]:
        """识别角色间冲突"""
        conflicts = []

        for treatment in TreatmentOption:
            scores = [
                opinion.treatment_preferences.get(treatment, 0.0)
                for opinion in role_opinions.values()
            ]

            if len(scores) > 1:
                variance = np.var(scores)
                if variance > 0.5:  # 高方差表示冲突
                    conflict_info = {
                        "treatment": treatment,
                        "variance": variance,
                        "min_score": min(scores),
                        "max_score": max(scores),
                        "score_range": max(scores) - min(scores),
                        "conflicting_roles": self._find_conflicting_roles(
                            treatment, role_opinions
                        ),
                        "conflict_severity": self._assess_conflict_severity(
                            variance, scores
                        ),
                    }
                    conflicts.append(conflict_info)

                    logger.info(
                        f"Conflict identified for {treatment.value}: variance={variance:.3f}"
                    )

        # 按冲突严重程度排序
        conflicts.sort(key=lambda x: x["variance"], reverse=True)

        return conflicts

    def _identify_agreements(
        self, role_opinions: Dict[RoleType, RoleOpinion]
    ) -> List[Dict[str, Any]]:
        """识别角色间一致意见"""
        agreements = []

        for treatment in TreatmentOption:
            scores = [
                opinion.treatment_preferences.get(treatment, 0.0)
                for opinion in role_opinions.values()
            ]

            if len(scores) > 1:
                variance = np.var(scores)
                mean_score = np.mean(scores)

                if variance < 0.2 and abs(mean_score) > 0.3:  # 低方差且有明确倾向
                    agreement_info = {
                        "treatment": treatment,
                        "consensus_score": mean_score,
                        "agreement_strength": 1.0 - variance,
                        "unanimous": all(s > 0.5 for s in scores)
                        or all(s < -0.5 for s in scores),
                        "supporting_roles": self._find_supporting_roles(
                            treatment, role_opinions, mean_score
                        ),
                        "confidence_level": self._assess_agreement_confidence(
                            scores, variance
                        ),
                    }
                    agreements.append(agreement_info)

                    logger.info(
                        f"Agreement identified for {treatment.value}: consensus={mean_score:.3f}"
                    )

        # 按共识强度排序
        agreements.sort(key=lambda x: x["agreement_strength"], reverse=True)

        return agreements

    def _find_conflicting_roles(
        self, treatment: TreatmentOption, role_opinions: Dict[RoleType, RoleOpinion]
    ) -> List[str]:
        """找出在特定治疗上有冲突的角色"""
        scores_by_role = {
            role.value: opinion.treatment_preferences.get(treatment, 0.0)
            for role, opinion in role_opinions.items()
        }

        mean_score = np.mean(list(scores_by_role.values()))
        conflicting_roles = [
            role
            for role, score in scores_by_role.items()
            if abs(score - mean_score) > 0.5
        ]

        return conflicting_roles

    def _find_supporting_roles(
        self,
        treatment: TreatmentOption,
        role_opinions: Dict[RoleType, RoleOpinion],
        consensus_score: float,
    ) -> List[str]:
        """找出支持特定治疗的角色"""
        supporting_roles = []

        for role, opinion in role_opinions.items():
            score = opinion.treatment_preferences.get(treatment, 0.0)
            # 如果角色评分与共识方向一致
            if (consensus_score > 0 and score > 0.3) or (
                consensus_score < 0 and score < -0.3
            ):
                supporting_roles.append(role.value)

        return supporting_roles

    def _assess_conflict_severity(self, variance: float, scores: List[float]) -> str:
        """评估冲突严重程度"""
        if variance > 1.0:
            return "severe"
        elif variance > 0.7:
            return "moderate"
        elif variance > 0.5:
            return "mild"
        else:
            return "low"

    def _assess_agreement_confidence(self, scores: List[float], variance: float) -> str:
        """评估一致性置信度"""
        mean_abs_score = np.mean([abs(s) for s in scores])

        if variance < 0.1 and mean_abs_score > 0.7:
            return "high"
        elif variance < 0.2 and mean_abs_score > 0.5:
            return "medium"
        else:
            return "low"

    def analyze_consensus_patterns(
        self, consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """分析共识模式
        1. 计算整体共识水平
        2. 分析各角色的影响力
        3. 评估治疗方案的极化程度
        4. 评估决策的复杂性
        5. 评估推荐的强度

        overall_consensus_level是什么意思？
        1. 0.0 表示完全不一致
        2. 0.5 表示不一致
        3. 1.0 表示完全一致
        role_influence_analysis是什么意思？
        1. 分析各角色在共识中的影响力
        2. 评估每个角色对共识的贡献程度
        3. 识别哪些角色在共识中扮演了关键角色
        decision_complexity是什么意思？
        1. 评估决策的复杂性
        2. 考虑决策中涉及的角色数量和交互
        3. 评估决策是否过于复杂，是否需要简化
        recommendation_strength是什么意思？
        1. 评估推荐的强度
        2. 考虑推荐的治疗方案的效果和风险
        3. 评估推荐是否符合患者的需求和健康状态
        """
        analysis = {
            "overall_consensus_level": self._calculate_overall_consensus(
                consensus_result
            ),
            "role_influence_analysis": self._analyze_role_influence(consensus_result),
            "treatment_polarization": self._analyze_treatment_polarization(
                consensus_result
            ),
            "decision_complexity": self._assess_decision_complexity(consensus_result),
            "recommendation_strength": self._assess_recommendation_strength(
                consensus_result
            ),
        }

        return analysis

    def _calculate_overall_consensus(self, consensus_result: ConsensusResult) -> float:
        """计算整体共识水平"""
        # 基于冲突和一致意见的数量和强度
        num_conflicts = len(consensus_result.conflicts)
        num_agreements = len(consensus_result.agreements)

        if num_conflicts == 0 and num_agreements > 0:
            return 1.0
        elif num_conflicts == 0 and num_agreements == 0:
            return 0.5
        else:
            # 计算冲突强度的平均值
            conflict_strength = (
                np.mean([c["variance"] for c in consensus_result.conflicts])
                if num_conflicts > 0
                else 0
            )
            agreement_strength = (
                np.mean([a["agreement_strength"] for a in consensus_result.agreements])
                if num_agreements > 0
                else 0
            )

            # 综合评估
            consensus_level = max(0.0, agreement_strength - conflict_strength * 0.5)
            return min(1.0, consensus_level)

    def _analyze_role_influence(
        self, consensus_result: ConsensusResult
    ) -> Dict[str, float]:
        """分析各角色的影响力"""
        role_influence = {}

        for role in RoleType:
            if role in consensus_result.role_opinions:
                opinion = consensus_result.role_opinions[role]

                # 计算该角色观点与最终共识的一致性
                alignment_scores = []
                for (
                    treatment,
                    final_score,
                ) in consensus_result.aggregated_scores.items():
                    role_score = opinion.treatment_preferences.get(treatment, 0.0)
                    alignment = (
                        1.0 - abs(final_score - role_score) / 2.0
                    )  # 归一化到[0,1]
                    alignment_scores.append(alignment)

                # 影响力 = 一致性 * 置信度
                avg_alignment = np.mean(alignment_scores) if alignment_scores else 0
                influence = avg_alignment * opinion.confidence
                role_influence[role.value] = influence

        return role_influence

    def _analyze_treatment_polarization(
        self, consensus_result: ConsensusResult
    ) -> Dict[str, str]:
        """分析治疗方案的极化程度"""
        polarization = {}

        for treatment in TreatmentOption:
            scores = []
            for opinion in consensus_result.role_opinions.values():
                score = opinion.treatment_preferences.get(treatment, 0.0)
                scores.append(score)

            if scores:
                variance = np.var(scores)
                mean_abs_score = np.mean([abs(s) for s in scores])

                if variance > 0.7:
                    polarization[treatment.value] = "highly_polarized"
                elif variance > 0.4:
                    polarization[treatment.value] = "moderately_polarized"
                elif mean_abs_score > 0.6:
                    polarization[treatment.value] = "strong_consensus"
                else:
                    polarization[treatment.value] = "weak_preference"

        return polarization

    def _assess_decision_complexity(self, consensus_result: ConsensusResult) -> str:
        """评估决策复杂度"""
        num_strong_preferences = sum(
            1
            for score in consensus_result.aggregated_scores.values()
            if abs(score) > 0.6
        )

        num_conflicts = len(consensus_result.conflicts)
        num_agreements = len(consensus_result.agreements)

        if num_conflicts > 2 and num_strong_preferences < 2:
            return "high_complexity"
        elif num_conflicts > 0 and num_agreements > 0:
            return "moderate_complexity"
        elif num_agreements > 2:
            return "low_complexity"
        else:
            return "uncertain"

    def _assess_recommendation_strength(
        self, consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """评估推荐强度"""
        # 找到评分最高的治疗方案
        best_treatment = max(
            consensus_result.aggregated_scores.items(), key=lambda x: x[1]
        )

        # 找到评分第二高的治疗方案
        sorted_treatments = sorted(
            consensus_result.aggregated_scores.items(), key=lambda x: x[1], reverse=True
        )

        second_best = sorted_treatments[1] if len(sorted_treatments) > 1 else (None, 0)

        # 计算推荐强度
        score_difference = best_treatment[1] - second_best[1]

        recommendation_strength = {
            "recommended_treatment": best_treatment[0].value,
            "recommendation_score": best_treatment[1],
            "strength_level": self._categorize_strength(
                best_treatment[1], score_difference
            ),
            "confidence": self._calculate_recommendation_confidence(
                consensus_result, best_treatment[0]
            ),
            "alternatives": [t.value for t, s in sorted_treatments[1:3] if s > 0.2],
        }

        return recommendation_strength

    def _categorize_strength(self, score: float, difference: float) -> str:
        """分类推荐强度"""
        if score > 0.8 and difference > 0.3:
            return "very_strong"
        elif score > 0.6 and difference > 0.2:
            return "strong"
        elif score > 0.4:
            return "moderate"
        elif score > 0.2:
            return "weak"
        else:
            return "very_weak"

    def _calculate_recommendation_confidence(
        self, consensus_result: ConsensusResult, treatment: TreatmentOption
    ) -> float:
        """计算推荐的置信度"""
        # 基于该治疗方案是否存在冲突
        treatment_conflicts = [
            c for c in consensus_result.conflicts if c["treatment"] == treatment
        ]

        # 基于支持该治疗的角色数量和置信度
        supporting_confidence = []
        for opinion in consensus_result.role_opinions.values():
            score = opinion.treatment_preferences.get(treatment, 0.0)
            if score > 0.3:  # 支持该治疗
                supporting_confidence.append(opinion.confidence)

        if not supporting_confidence:
            return 0.3  # 基础置信度

        avg_confidence = np.mean(supporting_confidence)

        # 如果有冲突，降低置信度
        if treatment_conflicts:
            conflict_penalty = treatment_conflicts[0]["variance"] * 0.2
            avg_confidence = max(0.2, avg_confidence - conflict_penalty)

        return avg_confidence

    def export_consensus_report(
        self, consensus_result: ConsensusResult, filepath: str = None
    ) -> str:
        """导出共识报告"""
        report = []
        report.append("=== Medical Team Consensus Report ===\n")
        report.append(f"Generated: {consensus_result.timestamp}\n")
        report.append(
            f"Convergence Achieved: {consensus_result.convergence_achieved}\n"
        )
        report.append(f"Total Rounds: {consensus_result.total_rounds}\n\n")

        # 共识矩阵
        report.append("=== Consensus Matrix ===\n")
        report.append(consensus_result.consensus_matrix.to_string())
        report.append("\n\n")

        # 治疗推荐
        report.append("=== Treatment Recommendations ===\n")
        sorted_treatments = sorted(
            consensus_result.aggregated_scores.items(), key=lambda x: x[1], reverse=True
        )

        for i, (treatment, score) in enumerate(sorted_treatments, 1):
            report.append(f"{i}. {treatment.value}: {score:+.3f}\n")

        # 冲突分析
        if consensus_result.conflicts:
            report.append("\n=== Identified Conflicts ===\n")
            for conflict in consensus_result.conflicts:
                report.append(f"- {conflict['treatment'].value}: ")
                report.append(f"variance={conflict['variance']:.3f}, ")
                report.append(
                    f"conflicting roles: {', '.join(conflict['conflicting_roles'])}\n"
                )

        # 一致意见
        if consensus_result.agreements:
            report.append("\n=== Strong Agreements ===\n")
            for agreement in consensus_result.agreements:
                report.append(f"- {agreement['treatment'].value}: ")
                report.append(f"consensus={agreement['consensus_score']:+.3f}, ")
                report.append(f"strength={agreement['agreement_strength']:.3f}\n")

        report_text = "".join(report)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Consensus report exported to {filepath}")

        return report_text
