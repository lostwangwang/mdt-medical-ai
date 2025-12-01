"""
角色智能体模块
文件路径: src/consensus/role_agents.py
作者: 姚刚
功能: 实现医疗团队中不同角色的智能体
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import re
import json

from pandas import DataFrame

from ..utils.llm_interface import LLMConfig, LLMInterface

from ..core.data_models import (
    DialogueRound,
    RoleType,
    TreatmentOption,
    PatientState,
    RoleOpinion,
    DialogueMessage,
    MedicalEvent,
    ChatRole, QuestionOpinion, RoleRegistry,
)

from ..tools.fix_json import fix_and_parse_single_json
from experiments.medqa_types import MedicalQuestionState, QuestionOption

logger = logging.getLogger(__name__)


class RoleAgent:
    """角色智能体基类"""

    def __init__(
            self,
            role: Union[RoleType, RoleRegistry],
            llm_interface: Optional[LLMInterface] = None,
            llm_config: Optional[LLMConfig] = None,
    ):
        """初始化角色智能体
        Args:
            role: 角色类型
            llm_interface: LLM接口实例，用于与LLM交互
            llm_config: LLM配置实例，当llm_interface为None时使用
            默认为None，在初始化时会根据角色类型自动创建
        """
        self.role = role
        if llm_interface is not None:
            self.llm_interface = llm_interface
        else:
            cfg = llm_config or LLMConfig()
            self.llm_interface = LLMInterface(cfg)
        self.specialization = (
            self._get_specialization()
        )  # 通过内部方法获取实例的"专业领域"
        self.dialogue_history = []
        self.current_stance = {}  # 当前立场

        # RL指导相关属性
        # 用于存储RL模型的指导，包括推荐的治疗方案、置信度和解释
        # 默认为None，在RL训练完成后会被赋值
        self.rl_guidance = None
        # 用于存储RL模型对当前立场的影响强度
        # 默认为0.0，在RL训练完成后会被赋值
        self.rl_influence_strength = 0.0
        # 用于存储RL模型的指导历史记录
        # 默认为空列表，在RL训练完成后会被填充
        self.rl_guidance_history = []
        # 用于存储角色的原始偏好
        # 默认为空字典，在初始化时会被赋值
        self.original_preferences = {}  # 保存原始偏好

        logger.info(f"Initialized {role.value} agent")

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
            RoleType.NUTRITIONIST: {
                "primary_concerns": [
                    "nutritional_status",
                    "dietary_compliance",
                    "metabolic_support",
                ],
                "weight_factors": {
                    "nutritional_risk": 0.4,
                    "dietary_compliance": 0.3,
                    "metabolic_status": 0.3,
                },
                "expertise_areas": [
                    "nutritional_assessment",
                    "dietary_planning",
                    "metabolic_monitoring",
                ],
                "communication_style": "scientific, data-driven, individualized guidance",
            },
            RoleType.REHABILITATION_THERAPIST: {
                "primary_concerns": [
                    "functional_recovery",
                    "exercise_capacity",
                    "daily_living_independence",
                ],
                "weight_factors": {
                    "functional_recovery": 0.4,
                    "exercise_tolerance": 0.3,
                    "self_care_ability": 0.3,
                },
                "expertise_areas": [
                    "functional_assessment",
                    "exercise_prescription",
                    "rehabilitation_planning",
                ],
                "communication_style": "encouraging, goal-oriented, motivational",
            },
        }
        return specializations.get(self.role, {})

    def recurt_agents_medqa(
            self,
            question_state: MedicalQuestionState,
            question_options: List[QuestionOption]
    ):
        response = self.llm_interface.llm_recurt_agents_medqa(question_state, question_options)
        if isinstance(response, str):
            try:
                reasoning = fix_and_parse_single_json(response)
            except Exception as e:
                logger.warning(
                    f"json解析失败:{e}, 原始字符串:{reasoning}"
                )
        elif isinstance(response, dict):
            pass
        else:
            logger.warning(
                f"未知的返回类型:{type(response)}"
            )
            return None
        return reasoning


    def generate_mdt_leader_summary_dataset(
            self,
            question_state: MedicalQuestionState,
            question_options: List[QuestionOption],
            dialogue_rounds: DialogueRound,
            consensus_dict: Dict[str, Any],
            opinions_dict: Dict[Union[RoleType, RoleRegistry], Union[RoleOpinion, QuestionOpinion]] = None,
    ):
        current_round = dialogue_rounds.round_number
        if consensus_dict["consensus"] == False:
            mdt_leader_summary = self.llm_interface.llm_generate_mdt_leader_content(question_state, question_options,
                                                                                    dialogue_rounds,
                                                                                    consensus_dict)
        elif consensus_dict["consensus"] == True:
            response = self.llm_interface.llm_generate_final_mdt_leader_summary(question_state,
                                                                                          question_options,
                                                                                          dialogue_rounds,
                                                                                          consensus_dict,
                                                                                          opinions_dict)
            if isinstance(response, str):
                try:
                    reasoning = fix_and_parse_single_json(response)
                except Exception as e:
                    logger.warning(
                        f"json解析失败:{e}, 原始字符串:{reasoning}"
                    )
            elif isinstance(response, dict):
                pass
            else:
                logger.warning(
                    f"final_mdt_prompt未知的返回类型:{type(response)}"
                )
                return None
            mdt_leader_summary = reasoning
        logger.info(f"current_round:{current_round}, MDT_LEADER_SUMMARY: {mdt_leader_summary}")
        return mdt_leader_summary

    def _update_agent_opinions_and_scores_medqa(
            self,
            question_state: MedicalQuestionState,
            current_round: DialogueRound,
            previous_opinion: Union[RoleOpinion, QuestionOpinion],
            question_options: List[QuestionOption],
            mdt_leader_summary: str,
            dataset_name: str = None,
    ) -> Union[RoleOpinion, QuestionOpinion]:
        """根据当前轮次的对话内容,更新角色的医疗问题偏好和医疗问题意见以及置信度"""
        reasoning = self._generate_update_agent_opinions_reasoning_medqa(
            question_state, current_round, previous_opinion, question_options, mdt_leader_summary, dataset_name
        )
        logger.debug(f"{self.role.value}更新医疗偏好:{reasoning}")
        print(f"{self.role.value}更新医疗偏好:{reasoning}")
        if isinstance(reasoning, str):
            try:
                reasoning = fix_and_parse_single_json(reasoning)
            except Exception as e:
                logger.warning(
                    f"json解析失败:{e}, 原始字符串:{reasoning}"
                )
        elif isinstance(reasoning, dict):
            pass
        else:
            logger.error(f"未知的推理类型:{type(reasoning)}, 原始字符串:{reasoning}")
            return None

        role_opinion = QuestionOpinion(
            role=self.role.value,
            scores=reasoning["scores"],
            reasoning=reasoning["reasoning"],
            evidence_strength=reasoning["evidence_strength"],
            evidences=reasoning["evidences"],
        )
        # 更新当前角色的 TreatmentPreferences
        self.current_stance = role_opinion.scores
        return role_opinion

    def _update_agent_opinions_and_preferences(
            self,
            patient_state: PatientState,
            current_round: DialogueRound,
            previous_opinion: RoleOpinion,
            treatment_options: List[TreatmentOption],
    ) -> RoleOpinion:
        """根据当前轮次的对话内容,更新角色的治疗偏好和治疗意见以及置信度"""
        reasoning = self._generate_update_agent_opinions_reasoning(
            patient_state, current_round, previous_opinion, treatment_options
        )
        reasoning = json.loads(reasoning)
        role_option = RoleOpinion(
            role=self.role.value,
            treatment_preferences=reasoning["treatment_preferences"],
            reasoning=reasoning["reasoning"],
            confidence=reasoning["confidence"],
            concerns=reasoning["concerns"],
        )

        self.current_stance = role_option.treatment_preferences
        return role_option

    def _generate_update_agent_opinions_reasoning_medqa(
            self,
            question_state: MedicalQuestionState,
            current_round: DialogueRound,
            previous_opinion: Union[RoleOpinion, QuestionOpinion],
            question_options: List[QuestionOption],
            mdt_leader_summary: str,
            dataset_name: str = None,
    ):
        """根据当前轮次的对话内容,生成更新角色医疗问题意见的推理"""
        # 如果有LLM接口，使用智能推理
        if self.llm_interface:
            try:
                # 使用LLM生成专业推理
                reasoning = self.llm_interface.generate_update_agent_opinions_reasoning_medqa(
                    question_state=question_state,
                    role=self.role,
                    current_round=current_round,
                    previous_opinion=previous_opinion,
                    question_options=question_options,
                    mdt_leader_summary=mdt_leader_summary,
                    dataset_name=dataset_name,
                )
                if reasoning and len(reasoning.strip()) > 0:
                    logger.debug(
                        f"[更新立场生成推理]Generated LLM reasoning for {self.role.value}: {reasoning}..."
                    )

                return reasoning

            except Exception as e:
                logger.warning(
                    f"[更新立场生成推理] LLM reasoning generation failed for {self.role}: {e}"
                )

    def _generate_update_agent_opinions_reasoning(
            self,
            patient_state: PatientState,
            current_round: DialogueRound,
            previous_opinion: RoleOpinion,
            treatment_options: List[TreatmentOption],
    ):
        """根据当前轮次的对话内容,生成更新角色意见的推理"""

        # 如果有LLM接口，使用智能推理
        if self.llm_interface:
            try:
                # 使用LLM生成专业推理
                reasoning = self.llm_interface.generate_update_agent_opinions_reasoning(
                    patient_state=patient_state,
                    role=self.role,
                    current_round=current_round,
                    previous_opinion=previous_opinion,
                    treatment_options=treatment_options,
                )
                if reasoning and len(reasoning.strip()) > 0:
                    logger.debug(
                        f"[更新立场生成推理]Generated LLM reasoning for {self.role.value}: {reasoning}..."
                    )

                    return reasoning

            except Exception as e:
                logger.warning(
                    f"[更新立场生成推理] LLM reasoning generation failed for {self.role}: {e}"
                )

    def generate_initial_opinion_medqa(
            self,
            question_state: MedicalQuestionState,
            question_options: List[QuestionOption],
            dataset_name: str = None
    ) -> QuestionOpinion:
        """生成初始意见 - 专为MedQA场景设计"""
        reasoning = self._generate_reasoning_medqa(
            question_state,
            question_options,
            dataset_name
        )
        if isinstance(reasoning, str):
            try:
                reasoning = fix_and_parse_single_json(reasoning)
            except Exception as e:
                logger.warning(
                    f"json解析失败:{e}, 原始字符串:{reasoning}"
                )
        elif isinstance(reasoning, dict):
            pass
        else:
            logger.error(f"未知的推理类型:{type(reasoning)}, 原始字符串:{reasoning}")
            return None

        role_opinion = QuestionOpinion(
            role=self.role,
            scores=reasoning["scores"],
            reasoning=reasoning["reasoning"],
            evidence_strength=reasoning["evidence_strength"],
            evidences=reasoning["evidences"],
        )
        self.current_stance = role_opinion.scores

        return role_opinion

    def _generate_reasoning_medqa(
            self,
            question_state: MedicalQuestionState,
            question_options: List[QuestionOption],
            dataset_name: str = None
    ) -> str:
        """生成决策推理 - 专为MedQA场景设计"""
        # 如果有LLM接口，使用智能推理
        if self.llm_interface:
            try:
                # 使用LLM生成专业推理
                reasoning = self.llm_interface.generate_treatment_reasoning_medqa(
                    question_state=question_state,
                    role=self.role,
                    question_options=question_options,
                    dataset_name=dataset_name
                )
                if reasoning and len(reasoning.strip()) > 0:
                    return reasoning

            except Exception as e:
                logger.warning(f"LLM reasoning generation failed for {self.role}: {e}")

        # 降级到改进的模板化推理
        template_reasoning = f"Based on the patient's condition and relevant knowledge, the recommended treatment"

        logger.debug(f"Using template reasoning for {self.role.value}")
        return template_reasoning

    def generate_initial_opinion(
            self,
            patient_state: PatientState,
            knowledge: Dict[str, Any],
            treatment_options: List[TreatmentOption],
    ) -> RoleOpinion:
        """生成初始意见"""
        # 我可以调用大模型吗? 可以，基于患者状态和相关知识，调用llm_interface.generate_text方法生成初始意见
        # 这个初始意见是怎么生成的? 基于患者状态和相关知识，调用_calculate_treatment_preferences和_generate_reasoning方法
        # treatment_prefs是什么？ 它是一个字典，键是治疗选项，值是该选项的偏好评分
        treatment_prefs = self._calculate_treatment_preferences(
            patient_state, knowledge
        )
        logger.debug(
            f"[{self.role.value}] Generated initial treatment preferences: {treatment_prefs}"
        )

        # reasoning是什么？ 它是一个字符串，描述了基于患者状态和治疗偏好的推理过程
        reasoning = self._generate_reasoning(
            patient_state,
            treatment_prefs,
            knowledge,
            treatment_options=treatment_options,
        )
        reasoning = json.loads(reasoning)
        role_option = RoleOpinion(
            role=self.role.value,
            treatment_preferences=reasoning["treatment_preferences"],
            reasoning=reasoning["reasoning"],
            confidence=reasoning["confidence"],
            concerns=reasoning["concerns"],
        )

        self.current_stance = role_option.treatment_preferences
        # 更新共识矩阵的置信度

        return role_option

    def _calculate_treatment_preferences_medqa(
            self, patient_state: PatientState, knowledge: Dict[str, Any]
    ) -> Dict[TreatmentOption, float]:
        """计算治疗偏好评分 - 专为MedQA场景设计"""
        prefs = {}

        for treatment in TreatmentOption:
            # 基于角色特征计算偏好评分
            score = self._evaluate_treatment_for_role_medqa(treatment, patient_state)
            prefs[treatment] = np.clip(score, -1.0, 1.0)

        return prefs

    def _evaluate_treatment_for_role_medqa(
            self, treatment: TreatmentOption, patient_state: PatientState
    ) -> float:
        """角色特异性治疗评估 - 专为MedQA场景设计"""
        # 基础评分表
        base_scores = {
            RoleType.ONCOLOGIST: {},
            # 其他角色的评分保持不变
        }

        role_scores = base_scores.get(self.role, {})
        base_score = role_scores.get(treatment, 0.0)

        # 根据患者状态调整评分
        adjusted_score = self._adjust_score_by_patient_state(
            base_score, treatment, patient_state
        )

        return adjusted_score

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
            RoleType.NUTRITIONIST: {
                TreatmentOption.SURGERY: 0.3,  # 关注术后营养恢复
                TreatmentOption.CHEMOTHERAPY: 0.5,  # 重视营养支持减轻副作用
                TreatmentOption.RADIOTHERAPY: 0.4,  # 关注放疗期间营养维持
                TreatmentOption.IMMUNOTHERAPY: 0.6,  # 强调免疫营养支持
                TreatmentOption.PALLIATIVE_CARE: 0.7,  # 重视营养舒适护理
                TreatmentOption.WATCHFUL_WAITING: 0.2,  # 预防性营养干预
            },
            RoleType.REHABILITATION_THERAPIST: {
                TreatmentOption.SURGERY: 0.6,  # 重视术后功能恢复
                TreatmentOption.CHEMOTHERAPY: 0.2,  # 关注体能维持
                TreatmentOption.RADIOTHERAPY: 0.4,  # 关注功能保持
                TreatmentOption.IMMUNOTHERAPY: 0.5,  # 支持免疫治疗期间活动
                TreatmentOption.PALLIATIVE_CARE: 0.8,  # 强调功能维持和生活质量
                TreatmentOption.WATCHFUL_WAITING: 0.3,  # 预防性康复训练
            },
            RoleType.RADIOLOGIST: {
                TreatmentOption.SURGERY: 0.4,
                TreatmentOption.CHEMOTHERAPY: 0.3,
                TreatmentOption.RADIOTHERAPY: 0.6,
                TreatmentOption.IMMUNOTHERAPY: 0.4,
                TreatmentOption.PALLIATIVE_CARE: 0.3,
                TreatmentOption.WATCHFUL_WAITING: 0.2,
            },
            RoleType.PATIENT_ADVOCATE: {
                TreatmentOption.SURGERY: 0.5,
                TreatmentOption.CHEMOTHERAPY: 0.3,
                TreatmentOption.RADIOTHERAPY: 0.4,
                TreatmentOption.IMMUNOTHERAPY: 0.6,
                TreatmentOption.PALLIATIVE_CARE: 0.9,
                TreatmentOption.WATCHFUL_WAITING: 0.9,
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
            treatment_options: List[TreatmentOption],
    ) -> str:
        """生成决策推理"""
        # 找到最推荐的治疗
        best_treatment = max(treatment_prefs.items(), key=lambda x: x[1])

        # 如果有LLM接口，使用智能推理
        if self.llm_interface:
            try:
                # 使用LLM生成专业推理
                reasoning = self.llm_interface.generate_treatment_reasoning(
                    patient_state=patient_state,
                    role=self.role,
                    treatment_option=best_treatment[0],
                    knowledge_context=knowledge,
                    treatment_options=treatment_options,
                )
                if reasoning and len(reasoning.strip()) > 0:
                    logger.debug(
                        f"Generated LLM reasoning for {self.role.value}: {reasoning}..."
                    )

                    return reasoning

            except Exception as e:
                logger.warning(f"LLM reasoning generation failed for {self.role}: {e}")

        # 降级到改进的模板化推理
        reasoning_templates = {
            RoleType.ONCOLOGIST: f"Based on clinical evidence and {patient_state.diagnosis} stage {patient_state.stage}, considering the patient's age {patient_state.age} and comorbidities {patient_state.comorbidities}, {best_treatment[0].value} offers the best survival benefit with acceptable risk profile.",
            RoleType.NURSE: f"From a care management perspective, {best_treatment[0].value} is feasible and manageable for this patient profile. Considering quality of life score {patient_state.quality_of_life_score} and current symptoms {patient_state.symptoms}, we can provide appropriate supportive care.",
            RoleType.PSYCHOLOGIST: f"Considering the patient's psychological status ({patient_state.psychological_status}) and overall well-being, {best_treatment[0].value} provides the best balance of treatment efficacy and mental health impact. The patient's coping ability should be supported throughout treatment.",
            RoleType.RADIOLOGIST: f"Based on imaging findings and anatomical considerations for {patient_state.diagnosis}, {best_treatment[0].value} is technically feasible and appropriate for this patient's disease stage and overall condition.",
            RoleType.PATIENT_ADVOCATE: f"In terms of patient autonomy and quality of life, {best_treatment[0].value} aligns with typical patient preferences while respecting the patient's values and maintaining dignity throughout the treatment process.",
            RoleType.NUTRITIONIST: f"From a nutritional perspective, {best_treatment[0].value} is optimal considering the patient's current nutritional status and metabolic needs. With quality of life score {patient_state.quality_of_life_score} and comorbidities {patient_state.comorbidities}, targeted nutritional support can enhance treatment tolerance and recovery outcomes.",
            RoleType.REHABILITATION_THERAPIST: f"Regarding functional recovery and physical rehabilitation, {best_treatment[0].value} provides the best opportunity for maintaining and improving the patient's functional capacity. Considering age {patient_state.age} and current symptoms {patient_state.symptoms}, a comprehensive rehabilitation plan can optimize treatment outcomes and daily living independence.",
        }

        template_reasoning = reasoning_templates.get(
            self.role,
            f"From {self.role.value} perspective, {best_treatment[0].value} is the recommended approach for this patient",
        )

        logger.debug(f"Using template reasoning for {self.role.value}")
        return template_reasoning

    def _calculate_confidence(self, patient_state: PatientState) -> float:
        """计算置信度"""
        # 基于数据完整性、病情复杂度等因素计算
        # 确保data_completeness至少为0.3，避免置信度为0
        data_completeness = max(0.3, min(1.0, len(patient_state.lab_results) / 5))
        complexity_penalty = max(0.5, 1.0 - len(patient_state.comorbidities) * 0.1)

        # 角色专业相关性
        role_relevance = {
            RoleType.ONCOLOGIST: 0.9,
            RoleType.RADIOLOGIST: 0.8,
            RoleType.NURSE: 0.7,
            RoleType.PSYCHOLOGIST: 0.6,
            RoleType.PATIENT_ADVOCATE: 0.7,
            RoleType.NUTRITIONIST: 0.8,  # 营养师在治疗支持中的重要性
            RoleType.REHABILITATION_THERAPIST: 0.8,  # 康复师在功能恢复中的重要性
        }

        base_confidence = role_relevance.get(self.role, 0.7)

        return base_confidence * data_completeness * complexity_penalty

    def generate_dialogue_response_medqa(
            self,
            question_state: MedicalQuestionState,
            question_options: List[QuestionOption],
            opinions_dict: Dict[Union[RoleType, RoleRegistry], Union[RoleOpinion,QuestionOpinion]] = None,
            last_round_messages: List[DialogueMessage] = None,
            mdt_leader_summary: str = None,
            dataset_name: str = None
    ):
        # 生成回应内容
        response_content = self._construct_response_medqa(
            question_state,
            question_options,
            opinions_dict=opinions_dict,
            last_round_messages=last_round_messages,
            mdt_leader_summary=mdt_leader_summary,
            dataset_name=dataset_name
        )

        return DialogueMessage(
            role=self.role,
            content=response_content,
            timestamp=datetime.now(),
            message_type="response",
        )

    def generate_dialogue_response(
            self,
            patient_state: PatientState,
            knowledge: Dict[str, Any],
            dialogue_context: List[DialogueMessage],
            target_treatment: TreatmentOption,
            opinions_dict: Dict[RoleType, RoleOpinion],
            last_round_messages: List[DialogueMessage],
    ) -> DialogueMessage:
        """生成对话回应"""

        # 生成回应内容
        response_content = self._construct_response(
            patient_state,
            knowledge,
            target_treatment,
            full_dialogue_context=dialogue_context,
            opinions_dict=opinions_dict,
            last_round_messages=last_round_messages,
        )

        return DialogueMessage(
            role=self.role,
            content=response_content,
            timestamp=datetime.now(),
            message_type="response",
            treatment_focus=target_treatment,
        )

    def _construct_response_medqa(
            self,
            question_state: MedicalQuestionState,
            question_options: List[QuestionOption],
            opinions_dict: Dict[Union[RoleType, RoleRegistry], Union[RoleOpinion, QuestionOpinion]] = None,
            last_round_messages: List[DialogueMessage] = None,
            mdt_leader_summary: str = None,
            dataset_name: str = None
    ) -> str:
        """构建回应内容 - 基于患者状态、知识、治疗选项、对话上下文、立场和上一轮对话, 要用"""
        logger.info(f"开始构建{self.role.value}的回应内容")
        # 优先使用LLM生成自然对话
        if self.llm_interface:
            try:
                current_opinion = opinions_dict[self.role.value]
                dialogue_history = self._get_last_round_history(
                    last_round_messages
                )
                # 使用LLM生成自然对话回应
                response = self.llm_interface.generate_dialogue_response_medqa(
                    question_state=question_state,
                    question_options=question_options,
                    role=self.role,
                    current_opinion=current_opinion,
                    dialogue_history=dialogue_history,
                    mdt_leader_summary=mdt_leader_summary,
                    dataset_name=dataset_name
                )

                if response and len(response.strip()) > 0:
                    logger.debug(
                        f"Generated LLM response for {self.role.value}: {response}..."
                    )
                    return response

            except Exception as e:
                logger.warning(f"LLM response generation failed for {self.role}: {e}")

    def _construct_response(
            self,
            patient_state: PatientState,
            knowledge: Dict[str, Any],
            treatment: TreatmentOption,
            full_dialogue_context: Optional[List[DialogueMessage]] = None,
            opinions_dict: Dict[RoleType, RoleOpinion] = None,
            last_round_messages: List[DialogueMessage] = None,
    ) -> str:
        """构建回应内容 - 基于患者状态、知识、治疗选项、对话上下文、立场和上一轮对话, 要用"""
        logger.info(f"开始构建{self.role.value}的回应内容")
        # 优先使用LLM生成自然对话
        if self.llm_interface:
            try:
                dialogue_context = ""
                current_stance = opinions_dict[self.role.value]
                logger.info(f"当前{self.role.value}立场: {current_stance}")

                # 使用LLM生成自然对话回应
                response = self.llm_interface.generate_dialogue_response(
                    patient_state=patient_state,
                    role=self.role,
                    treatment_option=treatment,
                    discussion_context=dialogue_context,
                    knowledge_context=knowledge,
                    current_stance=current_stance,
                    dialogue_history=self._get_recent_dialogue_history(
                        last_round_messages
                    ),
                )

                if response and len(response.strip()) > 0:
                    logger.debug(
                        f"Generated LLM response for {self.role.value}: {response}..."
                    )
                    return response

            except Exception as e:
                logger.warning(f"LLM response generation failed for {self.role}: {e}")

    def _get_last_round_history(
            self, last_round_messages: List[DialogueMessage]
    ) -> List[Dict[str, Any]]:
        """提取上一轮非当前角色的对话历史"""
        last_round_history: List[Dict[str, Any]] = []
        for item in last_round_messages:
            if item.role != self.role:
                last_round_history.append(
                    {"role": item.role.value, "content": item.content}
                )
        return last_round_history

    def _get_recent_dialogue_history(
            self, messages: Optional[List[DialogueMessage]] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """提取最近对话历史供LLM提示使用（仅需role与content）"""
        history_list: List[Dict[str, Any]] = []
        source = (
            messages if messages is not None else getattr(self, "dialogue_history", [])
        )
        if not source:
            return history_list
        recent = source[-limit:] if len(source) > limit else source
        for item in recent:
            if isinstance(item, dict):
                role_name = item.get("role", "Unknown")
                content = item.get("content", "")
            else:
                role_name = (
                    item.role.value
                    if hasattr(item, "role") and hasattr(item.role, "value")
                    else str(getattr(item, "role", "Unknown"))
                )
                content = getattr(item, "content", "")
            if not content:
                continue
            history_list.append({"role": role_name, "content": content})
        return history_list
