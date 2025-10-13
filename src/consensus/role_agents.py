"""
角色智能体模块
文件路径: src/consensus/role_agents.py
作者: 姚刚
功能: 实现医疗团队中不同角色的智能体
"""

import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import logging

from ..core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    RoleOpinion,
    DialogueMessage,
    MedicalEvent,
)

logger = logging.getLogger(__name__)


class RoleAgent:
    """角色智能体基类"""

    def __init__(self, role: RoleType, llm_interface=None):
        self.role = role
        self.llm_interface = llm_interface
        self.specialization = self._get_specialization()
        self.dialogue_history = []
        self.current_stance = {}

    def _get_specialization(self) -> Dict[str, Any]:
        """获取角色专业特征"""
        specializations = {
            RoleType.ONCOLOGIST: {
                "primary_concerns": [
                    "survival",
                    "treatment_efficacy",
                    "disease_progression",
                ],
                "weight_factors": {
                    "survival_rate": 0.4,
                    "side_effects": 0.2,
                    "quality_of_life": 0.4,
                },
                "expertise_areas": [
                    "systemic_therapy",
                    "treatment_sequencing",
                    "prognosis",
                ],
                "communication_style": "evidence-based, direct, technical",
            },
            RoleType.RADIOLOGIST: {
                "primary_concerns": [
                    "imaging_findings",
                    "treatment_response",
                    "anatomical_feasibility",
                ],
                "weight_factors": {
                    "tumor_size": 0.5,
                    "metastasis": 0.3,
                    "response": 0.2,
                },
                "expertise_areas": [
                    "imaging_interpretation",
                    "treatment_response",
                    "staging",
                ],
                "communication_style": "precise, diagnostic, detail-oriented",
            },
            RoleType.NURSE: {
                "primary_concerns": [
                    "feasibility",
                    "patient_compliance",
                    "daily_care_management",
                ],
                "weight_factors": {
                    "complexity": 0.3,
                    "side_effects": 0.4,
                    "adherence": 0.3,
                },
                "expertise_areas": [
                    "patient_education",
                    "symptom_management",
                    "care_coordination",
                ],
                "communication_style": "practical, patient-centered, caring",
            },
            RoleType.PSYCHOLOGIST: {
                "primary_concerns": [
                    "mental_health",
                    "coping_ability",
                    "family_dynamics",
                ],
                "weight_factors": {
                    "anxiety": 0.3,
                    "depression": 0.3,
                    "social_support": 0.4,
                },
                "expertise_areas": [
                    "psychological_assessment",
                    "coping_strategies",
                    "family_counseling",
                ],
                "communication_style": "empathetic, supportive, holistic",
            },
            RoleType.PATIENT_ADVOCATE: {
                "primary_concerns": [
                    "patient_preferences",
                    "quality_of_life",
                    "autonomy",
                ],
                "weight_factors": {
                    "autonomy": 0.4,
                    "comfort": 0.3,
                    "family_impact": 0.3,
                },
                "expertise_areas": [
                    "patient_rights",
                    "shared_decision_making",
                    "quality_of_life",
                ],
                "communication_style": "patient-focused, questioning, protective",
            },
        }
        return specializations.get(self.role, {})

    def generate_initial_opinion(
        self, patient_state: PatientState, knowledge: Dict[str, Any]
    ) -> RoleOpinion:
        """生成初始意见"""
        treatment_prefs = self._calculate_treatment_preferences(
            patient_state, knowledge
        )
        reasoning = self._generate_reasoning(patient_state, treatment_prefs, knowledge)

        opinion = RoleOpinion(
            role=self.role,
            treatment_preferences=treatment_prefs,
            reasoning=reasoning,
            confidence=self._calculate_confidence(patient_state),
            concerns=self._identify_concerns(patient_state),
        )

        self.current_stance = treatment_prefs
        return opinion

    def _calculate_treatment_preferences(
        self, patient_state: PatientState, knowledge: Dict[str, Any]
    ) -> Dict[TreatmentOption, float]:
        """计算治疗偏好评分"""
        prefs = {}

        for treatment in TreatmentOption:
            # 基于角色特征计算偏好评分
            score = self._evaluate_treatment_for_role(treatment, patient_state)
            prefs[treatment] = np.clip(score, -1.0, 1.0)

        return prefs

    def _evaluate_treatment_for_role(
        self, treatment: TreatmentOption, patient_state: PatientState
    ) -> float:
        """角色特异性治疗评估"""
        # 基础评分表
        base_scores = {
            RoleType.ONCOLOGIST: {
                TreatmentOption.SURGERY: 0.8,
                TreatmentOption.CHEMOTHERAPY: 0.7,
                TreatmentOption.RADIOTHERAPY: 0.6,
                TreatmentOption.IMMUNOTHERAPY: 0.5,
                TreatmentOption.PALLIATIVE_CARE: -0.2,
                TreatmentOption.WATCHFUL_WAITING: -0.5,
            },
            RoleType.NURSE: {
                TreatmentOption.SURGERY: 0.5,  # 关注术后护理复杂性
                TreatmentOption.CHEMOTHERAPY: 0.3,  # 关注副作用管理
                TreatmentOption.RADIOTHERAPY: 0.6,
                TreatmentOption.IMMUNOTHERAPY: 0.4,
                TreatmentOption.PALLIATIVE_CARE: 0.7,  # 强调舒适护理
                TreatmentOption.WATCHFUL_WAITING: 0.2,
            },
            RoleType.PSYCHOLOGIST: {
                TreatmentOption.SURGERY: 0.2,  # 关注心理创伤
                TreatmentOption.CHEMOTHERAPY: -0.1,  # 关注心理负担
                TreatmentOption.RADIOTHERAPY: 0.4,
                TreatmentOption.IMMUNOTHERAPY: 0.3,
                TreatmentOption.PALLIATIVE_CARE: 0.6,  # 重视心理支持
                TreatmentOption.WATCHFUL_WAITING: 0.5,  # 减少心理压力
            },
        }

        role_scores = base_scores.get(self.role, {})
        base_score = role_scores.get(treatment, 0.0)

        # 根据患者状态调整评分
        adjusted_score = self._adjust_score_by_patient_state(
            base_score, treatment, patient_state
        )

        return adjusted_score

    def _adjust_score_by_patient_state(
        self, base_score: float, treatment: TreatmentOption, patient_state: PatientState
    ) -> float:
        """根据患者状态调整评分"""
        # 年龄因素
        age_factor = 1.0
        if patient_state.age > 75:
            age_factor = (
                0.8
                if treatment in [TreatmentOption.SURGERY, TreatmentOption.CHEMOTHERAPY]
                else 1.0
            )

        # 并发症因素
        comorbidity_factor = 1.0
        if len(patient_state.comorbidities) > 2:
            comorbidity_factor = 0.7 if treatment == TreatmentOption.SURGERY else 0.9

        # 生活质量因素
        qol_factor = 1.0
        if patient_state.quality_of_life_score < 0.5:
            if treatment == TreatmentOption.PALLIATIVE_CARE:
                qol_factor = 1.3  # 提高姑息治疗评分
            elif treatment in [TreatmentOption.CHEMOTHERAPY, TreatmentOption.SURGERY]:
                qol_factor = 0.8  # 降低高强度治疗评分

        return base_score * age_factor * comorbidity_factor * qol_factor

    def _generate_reasoning(
        self,
        patient_state: PatientState,
        treatment_prefs: Dict[TreatmentOption, float],
        knowledge: Dict[str, Any],
    ) -> str:
        """生成决策推理"""
        # 找到最推荐的治疗
        best_treatment = max(treatment_prefs.items(), key=lambda x: x[1])

        reasoning_templates = {
            RoleType.ONCOLOGIST: f"Based on clinical evidence, {best_treatment[0].value} offers the best survival benefit for {patient_state.diagnosis} stage {patient_state.stage}.",
            RoleType.NURSE: f"From a care management perspective, {best_treatment[0].value} is feasible and manageable for this patient profile.",
            RoleType.PSYCHOLOGIST: f"Considering the patient's psychological well-being, {best_treatment[0].value} provides the best balance of treatment efficacy and mental health impact.",
            RoleType.RADIOLOGIST: f"Imaging findings support the feasibility of {best_treatment[0].value} in this case.",
            RoleType.PATIENT_ADVOCATE: f"In terms of patient autonomy and quality of life, {best_treatment[0].value} aligns with typical patient preferences.",
        }

        return reasoning_templates.get(
            self.role, f"{self.role.value} recommends {best_treatment[0].value}"
        )

    def _calculate_confidence(self, patient_state: PatientState) -> float:
        """计算置信度"""
        # 基于数据完整性、病情复杂度等因素计算
        data_completeness = min(1.0, len(patient_state.lab_results) / 5)
        complexity_penalty = max(0.5, 1.0 - len(patient_state.comorbidities) * 0.1)

        # 角色专业相关性
        role_relevance = {
            RoleType.ONCOLOGIST: 0.9,
            RoleType.RADIOLOGIST: 0.8,
            RoleType.NURSE: 0.7,
            RoleType.PSYCHOLOGIST: 0.6,
            RoleType.PATIENT_ADVOCATE: 0.7,
        }

        base_confidence = role_relevance.get(self.role, 0.7)

        return base_confidence * data_completeness * complexity_penalty

    def _identify_concerns(self, patient_state: PatientState) -> List[str]:
        """识别关注点"""
        concerns = []

        # 通用关注点
        if patient_state.age > 70:
            concerns.append("advanced_age")

        if len(patient_state.comorbidities) > 2:
            concerns.append("multiple_comorbidities")

        if patient_state.quality_of_life_score < 0.5:
            concerns.append("poor_quality_of_life")

        # 角色特异性关注点
        role_specific_concerns = {
            RoleType.ONCOLOGIST: ["disease_progression", "treatment_resistance"],
            RoleType.NURSE: ["patient_compliance", "care_complexity"],
            RoleType.PSYCHOLOGIST: ["psychological_burden", "family_stress"],
            RoleType.RADIOLOGIST: ["anatomical_constraints", "technical_feasibility"],
            RoleType.PATIENT_ADVOCATE: ["patient_autonomy", "informed_consent"],
        }

        concerns.extend(role_specific_concerns.get(self.role, [])[:2])

        return concerns[:3]  # 限制最多3个关注点

    def generate_dialogue_response(
        self,
        patient_state: PatientState,
        knowledge: Dict[str, Any],
        dialogue_context: List[DialogueMessage],
        target_treatment: TreatmentOption,
    ) -> DialogueMessage:
        """生成对话回应"""

        # 分析其他角色的观点
        opposing_views = self._analyze_opposing_views(
            dialogue_context, target_treatment
        )
        supporting_views = self._analyze_supporting_views(
            dialogue_context, target_treatment
        )

        # 生成回应内容
        response_content = self._construct_response(
            patient_state, knowledge, target_treatment, opposing_views, supporting_views
        )

        # 识别回应中引用的角色
        referenced_roles = self._identify_referenced_roles(
            dialogue_context, response_content
        )

        # 提取引用的证据
        evidence_cited = self._extract_evidence_citations(knowledge, response_content)

        return DialogueMessage(
            role=self.role,
            content=response_content,
            timestamp=datetime.now(),
            message_type="response",
            referenced_roles=referenced_roles,
            evidence_cited=evidence_cited,
            treatment_focus=target_treatment,
        )

    def _analyze_opposing_views(
        self, dialogue_context: List[DialogueMessage], treatment: TreatmentOption
    ) -> List[Dict[str, Any]]:
        """分析反对观点"""
        opposing_views = []
        my_stance = self.current_stance.get(treatment, 0)

        for message in dialogue_context:
            if message.role != self.role and message.treatment_focus == treatment:
                # 简化的观点分析
                if "not recommend" in message.content.lower() and my_stance > 0:
                    opposing_views.append(
                        {
                            "role": message.role,
                            "content": message.content,
                            "stance": "opposing",
                        }
                    )

        return opposing_views

    def _analyze_supporting_views(
        self, dialogue_context: List[DialogueMessage], treatment: TreatmentOption
    ) -> List[Dict[str, Any]]:
        """分析支持观点"""
        supporting_views = []
        my_stance = self.current_stance.get(treatment, 0)

        for message in dialogue_context:
            if message.role != self.role and message.treatment_focus == treatment:
                if "recommend" in message.content.lower() and my_stance > 0:
                    supporting_views.append(
                        {
                            "role": message.role,
                            "content": message.content,
                            "stance": "supporting",
                        }
                    )

        return supporting_views

    def _construct_response(
        self,
        patient_state: PatientState,
        knowledge: Dict[str, Any],
        treatment: TreatmentOption,
        opposing_views: List[Dict],
        supporting_views: List[Dict],
    ) -> str:
        """构建回应内容"""

        # 获取角色立场
        my_stance = self.current_stance.get(treatment, 0)

        # 构建基础立场
        if my_stance > 0.5:
            stance_phrase = "I strongly support"
        elif my_stance > 0:
            stance_phrase = "I cautiously recommend"
        elif my_stance < -0.5:
            stance_phrase = "I have serious concerns about"
        else:
            stance_phrase = "I have mixed feelings about"

        response_parts = [f"{stance_phrase} {treatment.value} for this patient."]

        # 基于专业领域的论据
        professional_reasoning = self._get_professional_reasoning(
            patient_state, treatment, knowledge
        )
        if professional_reasoning:
            response_parts.append(
                f"From my {self.role.value} perspective, {professional_reasoning}"
            )

        # 回应反对意见
        if opposing_views:
            counter_argument = self._generate_counter_argument(
                treatment, opposing_views[0], knowledge
            )
            response_parts.append(counter_argument)

        return " ".join(response_parts)

    def _get_professional_reasoning(
        self,
        patient_state: PatientState,
        treatment: TreatmentOption,
        knowledge: Dict[str, Any],
    ) -> str:
        """获取专业推理"""
        reasoning_map = {
            RoleType.ONCOLOGIST: {
                TreatmentOption.SURGERY: "surgical resection offers the best chance for cure",
                TreatmentOption.CHEMOTHERAPY: "systemic therapy is essential for micrometastatic disease",
            },
            RoleType.NURSE: {
                TreatmentOption.SURGERY: "this patient appears capable of handling the post-operative care requirements",
                TreatmentOption.CHEMOTHERAPY: "we need to carefully monitor for side effects and ensure compliance",
            },
        }

        role_reasoning = reasoning_map.get(self.role, {})
        return role_reasoning.get(treatment, "")

    def _generate_counter_argument(
        self,
        treatment: TreatmentOption,
        opposing_view: Dict[str, Any],
        knowledge: Dict[str, Any],
    ) -> str:
        """生成反驳论据"""
        counter_arguments = {
            RoleType.ONCOLOGIST: "the survival benefit outweighs the risks in this case.",
            RoleType.NURSE: "with proper support and monitoring, these concerns can be managed.",
            RoleType.PSYCHOLOGIST: "psychological support can help the patient cope with these challenges.",
        }

        return counter_arguments.get(
            self.role, "I believe the benefits justify proceeding."
        )

    def _identify_referenced_roles(
        self, dialogue_context: List[DialogueMessage], response_content: str
    ) -> List[RoleType]:
        """识别回应中引用的角色"""
        referenced = []
        for message in dialogue_context:
            if message.role.value in response_content.lower():
                referenced.append(message.role)
        return list(set(referenced))

    def _extract_evidence_citations(
        self, knowledge: Dict[str, Any], response_content: str
    ) -> List[str]:
        """提取证据引用"""
        citations = []
        if "guidelines" in response_content.lower():
            citations.extend(knowledge.get("guidelines", [])[:2])
        if "survival rate" in response_content.lower():
            citations.append("survival_statistics")
        return citations

    def update_stance_based_on_dialogue(
        self, dialogue_context: List[DialogueMessage]
    ) -> None:
        """基于对话更新立场"""
        for treatment in TreatmentOption:
            supporting_messages = [
                m
                for m in dialogue_context
                if m.treatment_focus == treatment
                and m.role != self.role
                and "recommend" in m.content.lower()
            ]

            if len(supporting_messages) >= 2:  # 多数支持时轻微调整立场
                current_stance = self.current_stance.get(treatment, 0)
                self.current_stance[treatment] = min(1.0, current_stance + 0.1)
