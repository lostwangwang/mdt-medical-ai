"""
医疗智能体系统 - 共识矩阵与强化学习模块
作者：姚刚
功能：多学科团队决策协同与强化学习优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoleType(Enum):
    """医疗团队角色类型"""

    ONCOLOGIST = "oncologist"  # 肿瘤科医生
    RADIOLOGIST = "radiologist"  # 影像科医生
    NURSE = "nurse"  # 护士
    PSYCHOLOGIST = "psychologist"  # 心理师
    PATIENT_ADVOCATE = "patient_advocate"  # 患者代表


class TreatmentOption(Enum):
    """治疗方案选项"""

    SURGERY = "surgery"
    CHEMOTHERAPY = "chemotherapy"
    RADIOTHERAPY = "radiotherapy"
    IMMUNOTHERAPY = "immunotherapy"
    PALLIATIVE_CARE = "palliative_care"
    WATCHFUL_WAITING = "watchful_waiting"


@dataclass
class PatientState:
    """患者状态摘要"""

    patient_id: str
    age: int
    diagnosis: str
    stage: str
    lab_results: Dict[str, float]
    vital_signs: Dict[str, float]
    symptoms: List[str]
    comorbidities: List[str]
    psychological_status: str
    quality_of_life_score: float
    timestamp: datetime


@dataclass
class RoleOpinion:
    """角色意见"""

    role: RoleType
    treatment_preferences: Dict[TreatmentOption, float]  # -1 to +1
    reasoning: str
    confidence: float  # 0 to 1
    concerns: List[str]


@dataclass
class DialogueMessage:
    """对话消息"""

    role: RoleType
    content: str
    timestamp: datetime
    message_type: str  # "initial_opinion", "response", "rebuttal", "consensus"
    referenced_roles: List[RoleType]
    evidence_cited: List[str]
    treatment_focus: TreatmentOption


@dataclass
class DialogueRound:
    """对话轮次"""

    round_number: int
    messages: List[DialogueMessage]
    focus_treatment: Optional[TreatmentOption]
    consensus_status: str  # "discussing", "converging", "concluded"


class MedicalKnowledgeRAG:
    """医学知识检索增强生成系统"""

    def __init__(self):
        # 在实际项目中，这里会连接到医学知识库
        self.knowledge_base = {
            "breast_cancer": {
                "early_stage": ["surgery", "adjuvant_therapy"],
                "advanced_stage": ["systemic_therapy", "palliative_care"],
                "contraindications": ["cardiac_dysfunction", "severe_comorbidities"],
            },
            "treatment_guidelines": {
                "NCCN": {
                    "breast_cancer_stage_ii": {
                        "recommended": ["surgery", "chemotherapy"],
                        "evidence_level": "Category 1",
                    }
                },
                "ESMO": {},
                "local_protocols": {},
            },
            "similar_cases": [
                {
                    "patient_profile": "65yo_female_breast_cancer_stage_ii",
                    "treatment_outcome": {"surgery": 0.85, "chemotherapy": 0.75},
                    "side_effects": {"surgery": "moderate", "chemotherapy": "severe"},
                }
            ],
        }

    def retrieve_relevant_knowledge(
        self,
        patient_state: PatientState,
        query_type: str,
        treatment_focus: TreatmentOption = None,
    ) -> Dict[str, Any]:
        """检索相关医学知识"""
        relevant_knowledge = {
            "guidelines": self._get_guidelines(patient_state, treatment_focus),
            "similar_cases": self._get_similar_cases(patient_state),
            "contraindications": self._get_contraindications(
                patient_state, treatment_focus
            ),
            "evidence_level": "A",
            "success_rates": self._get_success_rates(patient_state, treatment_focus),
            "side_effects": self._get_side_effects(treatment_focus),
        }

        return relevant_knowledge

    def _get_guidelines(
        self, patient_state: PatientState, treatment: TreatmentOption
    ) -> List[str]:
        """获取治疗指南"""
        guidelines = []
        if patient_state.diagnosis == "breast_cancer":
            if treatment == TreatmentOption.SURGERY:
                guidelines.append(
                    "NCCN recommends surgery for stage I-III breast cancer"
                )
                guidelines.append(
                    "Complete surgical resection is preferred when feasible"
                )
        return guidelines

    def _get_similar_cases(self, patient_state: PatientState) -> List[Dict]:
        """获取相似病例"""
        # 基于患者特征匹配相似病例
        return self.knowledge_base.get("similar_cases", [])

    def _get_contraindications(
        self, patient_state: PatientState, treatment: TreatmentOption
    ) -> List[str]:
        """获取禁忌症"""
        contraindications = []
        if treatment == TreatmentOption.SURGERY and patient_state.age > 80:
            contraindications.append("Advanced age increases surgical risk")
        if "cardiac_dysfunction" in patient_state.comorbidities:
            if treatment in [TreatmentOption.CHEMOTHERAPY, TreatmentOption.SURGERY]:
                contraindications.append("Cardiac dysfunction limits treatment options")
        return contraindications

    def _get_success_rates(
        self, patient_state: PatientState, treatment: TreatmentOption
    ) -> Dict[str, float]:
        """获取成功率数据"""
        # 模拟基于患者特征的成功率
        base_rates = {
            TreatmentOption.SURGERY: 0.85,
            TreatmentOption.CHEMOTHERAPY: 0.70,
            TreatmentOption.RADIOTHERAPY: 0.75,
        }

        # 根据患者特征调整
        if treatment in base_rates:
            rate = base_rates[treatment]
            if patient_state.age > 70:
                rate *= 0.9
            if len(patient_state.comorbidities) > 2:
                rate *= 0.85
            return {"5_year_survival": rate, "response_rate": rate * 0.9}

        return {}

    def _get_side_effects(self, treatment: TreatmentOption) -> Dict[str, str]:
        """获取副作用信息"""
        side_effects = {
            TreatmentOption.SURGERY: {
                "severity": "moderate",
                "duration": "short-term",
                "main_effects": "pain, infection risk",
            },
            TreatmentOption.CHEMOTHERAPY: {
                "severity": "severe",
                "duration": "treatment period",
                "main_effects": "nausea, fatigue, immunosuppression",
            },
            TreatmentOption.RADIOTHERAPY: {
                "severity": "mild",
                "duration": "treatment + 2 weeks",
                "main_effects": "skin irritation, fatigue",
            },
        }
        return side_effects.get(treatment, {})


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
                elif "strongly recommend" in message.content.lower() and my_stance < 0:
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
        communication_style = self.specialization.get(
            "communication_style", "professional"
        )

        # 构建基础立场
        if my_stance > 0.5:
            stance_phrase = "I strongly support"
        elif my_stance > 0:
            stance_phrase = "I cautiously recommend"
        elif my_stance < -0.5:
            stance_phrase = "I have serious concerns about"
        else:
            stance_phrase = "I have mixed feelings about"

        response_parts = []

        # 开场立场
        response_parts.append(f"{stance_phrase} {treatment.value} for this patient.")

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
            opposing_role = opposing_views[0]["role"].value
            response_parts.append(f"While I understand {opposing_role}'s concerns, ")
            counter_argument = self._generate_counter_argument(
                treatment, opposing_views[0], knowledge
            )
            response_parts.append(counter_argument)

        # 引用证据
        evidence = self._cite_relevant_evidence(treatment, knowledge)
        if evidence:
            response_parts.append(f"The evidence shows {evidence}")

        # 针对患者特异性考虑
        patient_specific = self._address_patient_specifics(patient_state, treatment)
        if patient_specific:
            response_parts.append(f"For this specific patient, {patient_specific}")

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
            RoleType.PSYCHOLOGIST: {
                TreatmentOption.SURGERY: "the patient seems psychologically prepared for surgical intervention",
                TreatmentOption.CHEMOTHERAPY: "we should consider the psychological burden of prolonged treatment",
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
            RoleType.RADIOLOGIST: "the imaging findings support the feasibility of this approach.",
        }

        return counter_arguments.get(
            self.role, "I believe the benefits justify proceeding."
        )

    def _cite_relevant_evidence(
        self, treatment: TreatmentOption, knowledge: Dict[str, Any]
    ) -> str:
        """引用相关证据"""
        success_rates = knowledge.get("success_rates", {})
        guidelines = knowledge.get("guidelines", [])

        evidence_parts = []

        if success_rates:
            survival_rate = success_rates.get("5_year_survival")
            if survival_rate:
                evidence_parts.append(f"a {survival_rate:.1%} five-year survival rate")

        if guidelines:
            evidence_parts.append(f"guidelines recommend this approach")

        return " and ".join(evidence_parts) if evidence_parts else ""

    def _address_patient_specifics(
        self, patient_state: PatientState, treatment: TreatmentOption
    ) -> str:
        """针对患者特异性的考虑"""
        considerations = []

        if patient_state.age > 70:
            considerations.append("given the patient's age")

        if len(patient_state.comorbidities) > 2:
            considerations.append("considering the multiple comorbidities")

        if patient_state.quality_of_life_score < 0.5:
            considerations.append("given the current quality of life concerns")

        return ", ".join(considerations) if considerations else ""

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
            citations.extend(knowledge.get("guidelines", [])[:2])  # 最多2个指南引用
        if "survival rate" in response_content.lower():
            citations.append("survival_statistics")
        return citations

    def update_stance_based_on_dialogue(
        self, dialogue_context: List[DialogueMessage]
    ) -> None:
        """基于对话更新立场"""
        # 简化的立场更新逻辑
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
        # 这里实现每个角色的具体评估逻辑
        base_scores = {
            RoleType.ONCOLOGIST: {
                TreatmentOption.SURGERY: 0.8,
                TreatmentOption.CHEMOTHERAPY: 0.7,
                TreatmentOption.RADIOTHERAPY: 0.6,
                TreatmentOption.IMMUNOTHERAPY: 0.5,
                TreatmentOption.PALLIATIVE_CARE: -0.2,
                TreatmentOption.WATCHFUL_WAITING: -0.5,
            }
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

        return base_score * age_factor * comorbidity_factor

    def _generate_reasoning(
        self, patient_state: PatientState, treatment_prefs: Dict[TreatmentOption, float]
    ) -> str:
        """生成决策推理"""
        # 在实际实现中，这里会调用LLM生成详细推理
        return f"{self.role.value} recommendation based on {patient_state.diagnosis}"

    def _calculate_confidence(self, patient_state: PatientState) -> float:
        """计算置信度"""
        # 基于数据完整性、病情复杂度等因素计算
        data_completeness = min(1.0, len(patient_state.lab_results) / 10)
        complexity_penalty = max(0.5, 1.0 - len(patient_state.comorbidities) * 0.1)

        return data_completeness * complexity_penalty


class MultiAgentDialogueManager:
    """多智能体对话管理器"""

    def __init__(self, rag_system: MedicalKnowledgeRAG):
        self.agents = {role: RoleAgent(role) for role in RoleType}
        self.rag_system = rag_system
        self.dialogue_rounds = []
        self.current_round = 0
        self.max_rounds = 5
        self.convergence_threshold = 0.8

    def conduct_mdt_discussion(self, patient_state: PatientState) -> Dict[str, Any]:
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
        # 获取相关医学知识
        initial_knowledge = self.rag_system.retrieve_relevant_knowledge(
            patient_state, "initial_assessment"
        )

        # 生成各角色的初始意见
        initial_round = DialogueRound(
            round_number=0,
            messages=[],
            focus_treatment=None,
            consensus_status="discussing",
        )

        for role, agent in self.agents.items():
            opinion = agent.generate_initial_opinion(patient_state, initial_knowledge)

            # 生成初始发言
            initial_message = self._create_initial_message(
                agent, opinion, patient_state
            )
            initial_round.messages.append(initial_message)

        self.dialogue_rounds.append(initial_round)

    def _create_initial_message(
        self, agent: RoleAgent, opinion: RoleOpinion, patient_state: PatientState
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

            # 简短停顿模拟思考时间
            logger.info(
                f"{role.value} responded to discussion on {current_round.focus_treatment.value}"
            )

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

        # 选择提及最多的治疗方案
        if treatment_mentions:
            return max(treatment_mentions.items(), key=lambda x: x[1])[0]

        return TreatmentOption.CHEMOTHERAPY  # 备选话题

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

        # 计算立场变化
        total_change = 0
        agent_count = 0

        for role, agent in self.agents.items():
            current_stances = agent.current_stance
            for treatment, score in current_stances.items():
                # 简化：如果立场变化小于阈值则认为收敛
                if abs(score) > 0.5:  # 只考虑有明确立场的情况
                    agent_count += 1

        # 如果大多数角色都有明确立场，认为已收敛
        return agent_count >= len(self.agents) * 0.8

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
                    f"Focusing next round on contentious treatment: {most_contentious[0].value}"
                )

    def _generate_final_consensus(self, patient_state: PatientState) -> Dict[str, Any]:
        """生成最终共识结果"""
        # 收集最终立场
        final_opinions = {}
        for role, agent in self.agents.items():
            final_opinion = RoleOpinion(
                role=role,
                treatment_preferences=agent.current_stance.copy(),
                reasoning=f"Final position after {self.current_round} rounds of discussion",
                confidence=self._calculate_final_confidence(agent),
                concerns=[],  # 简化
            )
            final_opinions[role] = final_opinion

        # 使用共识矩阵系统处理最终结果
        consensus_system = ConsensusMatrix()
        consensus_system.agents = self.agents  # 使用更新后的智能体

        # 生成详细的对话摘要
        dialogue_summary = self._generate_dialogue_summary()

        # 生成最终共识矩阵
        final_consensus = consensus_system.generate_consensus(patient_state)
        final_consensus["dialogue_summary"] = dialogue_summary
        final_consensus["total_rounds"] = self.current_round
        final_consensus["convergence_achieved"] = self._check_convergence()

        return final_consensus

    def _calculate_final_confidence(self, agent: RoleAgent) -> float:
        """计算最终置信度"""
        # 基于立场的明确性和一致性
        stances = list(agent.current_stance.values())
        if not stances:
            return 0.5

        # 立场越明确，置信度越高
        max_stance = max(abs(s) for s in stances)
        return min(1.0, 0.5 + max_stance * 0.5)

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


class ConsensusMatrix:
    """共识矩阵系统"""

    def __init__(self):
        self.agents = {role: RoleAgent(role) for role in RoleType}
        self.rag_system = MedicalKnowledgeRAG()

    def generate_consensus(self, patient_state: PatientState) -> Dict[str, Any]:
        """生成共识矩阵（集成对话结果）"""
        # 如果智能体已经经过对话更新，直接使用其当前立场
        role_opinions = {}
        for role, agent in self.agents.items():
            if hasattr(agent, "current_stance") and agent.current_stance:
                # 使用对话后的立场
                opinion = RoleOpinion(
                    role=role,
                    treatment_preferences=agent.current_stance,
                    reasoning="Post-dialogue consensus",
                    confidence=0.8,  # 经过对话后置信度较高
                    concerns=[],
                )
            else:
                # 生成初始意见
                relevant_knowledge = self.rag_system.retrieve_relevant_knowledge(
                    patient_state, "treatment_recommendation"
                )
                opinion = agent.generate_initial_opinion(
                    patient_state, relevant_knowledge
                )

            role_opinions[role] = opinion

        # 构建共识矩阵
        consensus_matrix = self._build_consensus_matrix(role_opinions)

        # 计算综合评分
        aggregated_scores = self._aggregate_scores(role_opinions)

        # 识别冲突与一致性
        conflicts = self._identify_conflicts(role_opinions)
        agreements = self._identify_agreements(role_opinions)

        return {
            "consensus_matrix": consensus_matrix,
            "role_opinions": role_opinions,
            "aggregated_scores": aggregated_scores,
            "conflicts": conflicts,
            "agreements": agreements,
            "timestamp": datetime.now(),
        }

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

        return pd.DataFrame(
            matrix_data,
            index=[t.value for t in treatments],
            columns=[r.value for r in roles],
        )

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
                # 加权平均
                weighted_score = np.average(scores, weights=weights)
                aggregated[treatment] = weighted_score

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
                    conflicts.append(
                        {
                            "treatment": treatment,
                            "variance": variance,
                            "min_score": min(scores),
                            "max_score": max(scores),
                            "conflicting_roles": self._find_conflicting_roles(
                                treatment, role_opinions
                            ),
                        }
                    )

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


class MDTReinforcementLearning:
    """医疗决策强化学习环境"""

    def __init__(self, consensus_system: ConsensusMatrix):
        self.consensus_system = consensus_system
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def create_state_vector(
        self, patient_state: PatientState, consensus_result: Dict[str, Any]
    ) -> np.ndarray:
        """创建状态向量"""
        # 患者特征
        patient_features = [
            patient_state.age / 100.0,  # 归一化年龄
            len(patient_state.comorbidities) / 10.0,  # 归一化并发症数量
            patient_state.quality_of_life_score,
        ]

        # 共识特征
        aggregated_scores = list(consensus_result["aggregated_scores"].values())
        conflict_count = len(consensus_result["conflicts"])
        agreement_count = len(consensus_result["agreements"])

        consensus_features = [
            np.mean(aggregated_scores),  # 平均共识得分
            np.std(aggregated_scores),  # 共识变异度
            conflict_count / len(TreatmentOption),  # 冲突比例
            agreement_count / len(TreatmentOption),  # 一致比例
        ]

        return np.array(patient_features + consensus_features + aggregated_scores)

    def calculate_reward(
        self,
        action: TreatmentOption,
        consensus_result: Dict[str, Any],
        patient_state: PatientState,
    ) -> float:
        """计算奖励函数"""
        # 基础奖励：共识得分
        consensus_score = consensus_result["aggregated_scores"].get(action, 0.0)

        # 一致性奖励
        agreements = consensus_result["agreements"]
        consensus_bonus = sum(
            agreement["consensus_score"] * agreement["agreement_strength"]
            for agreement in agreements
            if TreatmentOption(agreement["treatment"]) == action
        )

        # 冲突惩罚
        conflicts = consensus_result["conflicts"]
        conflict_penalty = sum(
            conflict["variance"]
            for conflict in conflicts
            if TreatmentOption(conflict["treatment"]) == action
        )

        # 患者适应性奖励
        patient_suitability = self._calculate_patient_suitability(action, patient_state)

        total_reward = (
            consensus_score
            + consensus_bonus * 0.5
            - conflict_penalty * 0.3
            + patient_suitability * 0.2
        )

        return total_reward

    def _calculate_patient_suitability(
        self, action: TreatmentOption, patient_state: PatientState
    ) -> float:
        """计算治疗方案对患者的适应性"""
        # 简化的适应性计算
        suitability_scores = {
            TreatmentOption.SURGERY: 1.0 - (patient_state.age / 100.0),
            TreatmentOption.CHEMOTHERAPY: patient_state.quality_of_life_score,
            TreatmentOption.PALLIATIVE_CARE: 1.0 - patient_state.quality_of_life_score,
        }

        return suitability_scores.get(action, 0.5)


def main():
    """主函数演示完整的MDT对话与共识系统"""
    print("=== 医疗多学科团队智能决策系统演示 ===\n")

    # 创建示例患者状态
    sample_patient = PatientState(
        patient_id="P001",
        age=65,
        diagnosis="breast_cancer",
        stage="II",
        lab_results={"creatinine": 1.2, "hemoglobin": 11.5, "cea": 3.5},
        vital_signs={"bp_systolic": 140, "heart_rate": 78, "weight": 70},
        symptoms=["fatigue", "pain", "anxiety"],
        comorbidities=["diabetes", "hypertension"],
        psychological_status="anxious",
        quality_of_life_score=0.7,
        timestamp=datetime.now(),
    )

    print(f"患者信息：{sample_patient.patient_id}, {sample_patient.age}岁")
    print(f"诊断：{sample_patient.diagnosis} ({sample_patient.stage}期)")
    print(f"并发症：{', '.join(sample_patient.comorbidities)}")
    print(f"生活质量评分：{sample_patient.quality_of_life_score}\n")

    # 创建RAG系统和对话管理器
    rag_system = MedicalKnowledgeRAG()
    dialogue_manager = MultiAgentDialogueManager(rag_system)

    print("=== 开始MDT多智能体对话 ===\n")

    # 进行MDT讨论
    mdt_result = dialogue_manager.conduct_mdt_discussion(sample_patient)

    # 展示对话结果
    print("=== 对话摘要 ===")
    dialogue_summary = mdt_result.get("dialogue_summary", {})
    print(f"总消息数：{dialogue_summary.get('total_messages', 0)}")
    print(f"讨论轮数：{mdt_result.get('total_rounds', 0)}")
    print(f"是否收敛：{mdt_result.get('convergence_achieved', False)}")

    key_topics = dialogue_summary.get("key_topics", [])
    if key_topics:
        print("\n主要讨论话题：")
        for i, (treatment, count) in enumerate(key_topics, 1):
            print(f"  {i}. {treatment.value} (提及{count}次)")

    evidence_cited = dialogue_summary.get("evidence_cited", [])
    if evidence_cited:
        print(f"\n引用证据：{', '.join(evidence_cited[:3])}")

    print("\n=== 最终共识矩阵 ===")
    consensus_matrix = mdt_result["consensus_matrix"]
    print(consensus_matrix.round(3))

    print("\n=== 治疗方案综合评分 ===")
    aggregated_scores = mdt_result["aggregated_scores"]
    sorted_treatments = sorted(
        aggregated_scores.items(), key=lambda x: x[1], reverse=True
    )

    for i, (treatment, score) in enumerate(sorted_treatments, 1):
        recommendation_level = get_recommendation_level(score)
        print(f"{i}. {treatment.value:<15} : {score:+.3f} ({recommendation_level})")

    print("\n=== 角色意见分析 ===")
    role_opinions = mdt_result["role_opinions"]
    for role, opinion in role_opinions.items():
        print(f"\n{role.value}:")
        top_preference = max(opinion.treatment_preferences.items(), key=lambda x: x[1])
        print(f"  首选方案: {top_preference[0].value} (评分: {top_preference[1]:+.2f})")
        print(f"  置信度: {opinion.confidence:.2f}")
        if opinion.concerns:
            print(f"  主要顾虑: {', '.join(opinion.concerns[:2])}")

    # 冲突分析
    conflicts = mdt_result["conflicts"]
    agreements = mdt_result["agreements"]

    if conflicts:
        print(f"\n=== 发现 {len(conflicts)} 个争议点 ===")
        for i, conflict in enumerate(conflicts, 1):
            treatment = conflict["treatment"]
            variance = conflict["variance"]
            print(f"{i}. {treatment.value}: 分歧程度 {variance:.3f}")
            print(
                f"   评分范围: {conflict['min_score']:+.2f} 到 {conflict['max_score']:+.2f}"
            )

    if agreements:
        print(f"\n=== 发现 {len(agreements)} 个一致意见 ===")
        for i, agreement in enumerate(agreements, 1):
            treatment = agreement["treatment"]
            consensus_score = agreement["consensus_score"]
            print(f"{i}. {treatment.value}: 共识评分 {consensus_score:+.3f}")
            if agreement["unanimous"]:
                print("   (全体一致)")

    print("\n=== 系统推荐 ===")
    best_treatment = sorted_treatments[0]
    print(f"推荐治疗方案：{best_treatment[0].value}")
    print(f"综合评分：{best_treatment[1]:+.3f}")

    # 生成推荐理由
    print("\n推荐理由：")
    if best_treatment[1] > 0.5:
        print("- 获得医疗团队多数支持")
    if (
        len(
            [
                a
                for a in agreements
                if TreatmentOption(a["treatment"]) == best_treatment[0]
            ]
        )
        > 0
    ):
        print("- 在该方案上达成了较好共识")
    if best_treatment[1] > 0:
        print("- 综合评估显示收益大于风险")

    print("\n=== 演示完成 ===")


def get_recommendation_level(score: float) -> str:
    """获取推荐等级描述"""
    if score > 0.7:
        return "强烈推荐"
    elif score > 0.3:
        return "推荐"
    elif score > 0:
        return "轻微推荐"
    elif score > -0.3:
        return "中性"
    elif score > -0.7:
        return "不推荐"
    else:
        return "强烈不推荐"


def demo_dialogue_system():
    """演示对话系统的详细过程"""
    print("=== 多智能体对话系统详细演示 ===\n")

    # 创建简单的患者案例
    patient = PatientState(
        patient_id="P002",
        age=45,
        diagnosis="breast_cancer",
        stage="I",
        lab_results={"creatinine": 0.9, "hemoglobin": 12.8},
        vital_signs={"bp_systolic": 120, "heart_rate": 72},
        symptoms=["mild_fatigue"],
        comorbidities=[],
        psychological_status="stable",
        quality_of_life_score=0.85,
        timestamp=datetime.now(),
    )

    # 创建对话管理器
    rag_system = MedicalKnowledgeRAG()
    dialogue_manager = MultiAgentDialogueManager(rag_system)

    # 逐步演示对话过程
    print("1. 初始化各角色立场...")
    dialogue_manager._initialize_discussion(patient)

    if dialogue_manager.dialogue_rounds:
        initial_round = dialogue_manager.dialogue_rounds[0]
        print(f"   各角色发表了 {len(initial_round.messages)} 条初始意见")

        for message in initial_round.messages[:2]:  # 显示前两个角色的发言
            print(f"   {message.role.value}: {message.content[:100]}...")

    print("\n2. 进行多轮对话...")
    dialogue_manager.current_round = 0  # 重置

    # 模拟2轮对话
    for round_num in range(1, 3):
        dialogue_manager.current_round = round_num
        round_data = dialogue_manager._conduct_dialogue_round(patient)
        dialogue_manager.dialogue_rounds.append(round_data)

        print(
            f"   第{round_num}轮: 讨论 {round_data.focus_treatment.value}, {len(round_data.messages)} 条回应"
        )

        # 显示一条示例回应
        if round_data.messages:
            sample_msg = round_data.messages[0]
            print(f"   示例回应({sample_msg.role.value}): {sample_msg.content[:80]}...")

    print("\n3. 生成最终共识...")
    final_result = dialogue_manager._generate_final_consensus(patient)

    print(f"   共识矩阵: {final_result['consensus_matrix'].shape}")
    print(f"   识别冲突: {len(final_result['conflicts'])} 个")
    print(f"   一致意见: {len(final_result['agreements'])} 个")

    print("\n=== 对话演示完成 ===")


if __name__ == "__main__":
    # 运行完整演示
    main()

    print("\n" + "=" * 60 + "\n")

    # 运行详细对话演示
    demo_dialogue_system()
