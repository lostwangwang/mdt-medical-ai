"""
角色智能体模块
文件路径: src/consensus/role_agents.py
作者: 姚刚
功能: 实现医疗团队中不同角色的智能体
"""

import numpy as np
from typing import Dict, List, Any, Optional
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
        """初始化角色智能体"""
        self.role = role
        self.llm_interface = llm_interface
        self.specialization = self._get_specialization()
        self.dialogue_history = []
        self.current_stance = {}  # 当前立场
        
        # RL指导相关属性
        self.rl_guidance = None
        self.rl_influence_strength = 0.0
        self.rl_guidance_history = []
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
        }
        return specializations.get(self.role, {})

    def generate_initial_opinion(
        self, patient_state: PatientState, knowledge: Dict[str, Any]
    ) -> RoleOpinion:
        """生成初始意见"""
        # 我可以调用大模型吗? 可以，基于患者状态和相关知识，调用llm_interface.generate_text方法生成初始意见
        # 这个初始意见是怎么生成的? 基于患者状态和相关知识，调用_calculate_treatment_preferences和_generate_reasoning方法
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
        
        # 如果有LLM接口，使用智能推理
        if self.llm_interface:
            try:
                # 使用LLM生成专业推理
                reasoning = self.llm_interface.generate_treatment_reasoning(
                    patient_state=patient_state,
                    role=self.role,
                    treatment_option=best_treatment[0],
                    knowledge_context=knowledge
                )
                
                if reasoning and len(reasoning.strip()) > 0:
                    logger.debug(f"Generated LLM reasoning for {self.role.value}: {reasoning[:100]}...")
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
        }

        template_reasoning = reasoning_templates.get(
            self.role, f"From {self.role.value} perspective, {best_treatment[0].value} is the recommended approach for this patient"
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
        
        # 如果有LLM接口，使用智能对话生成
        if self.llm_interface:
            try:
                # 构建对话上下文
                context_summary = ""
                if opposing_views:
                    context_summary += f"Opposing views: {opposing_views[0].get('content', '')[:100]}... "
                if supporting_views:
                    context_summary += f"Supporting views: {supporting_views[0].get('content', '')[:100]}... "
                
                # 使用LLM生成自然对话回应
                response = self.llm_interface.generate_dialogue_response(
                    patient_state=patient_state,
                    role=self.role,
                    treatment_option=treatment,
                    discussion_context=context_summary,
                    knowledge_context=knowledge,
                    current_stance=self.current_stance
                )
                
                if response and len(response.strip()) > 0:
                    logger.debug(f"Generated LLM response for {self.role.value}: {response[:100]}...")
                    return response
                    
            except Exception as e:
                logger.warning(f"LLM response generation failed for {self.role}: {e}")

        # 降级到改进的模板化回应
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

        # 添加角色特定的开场白
        role_openings = {
            RoleType.ONCOLOGIST: "From an oncological perspective,",
            RoleType.NURSE: "As the care coordinator,",
            RoleType.PSYCHOLOGIST: "Considering the psychological aspects,",
            RoleType.RADIOLOGIST: "Based on imaging findings,",
            RoleType.PATIENT_ADVOCATE: "Speaking for the patient's interests,",
        }
        
        opening = role_openings.get(self.role, f"As the {self.role.value},")
        response_parts = [opening, f"{stance_phrase} {treatment.value} for this patient."]

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
            response_parts.append(f"However, {counter_argument}")

        logger.debug(f"Using template response for {self.role.value}")
        return " ".join(response_parts)

    def _get_professional_reasoning(
        self,
        patient_state: PatientState,
        treatment: TreatmentOption,
        knowledge: Dict[str, Any],
    ) -> str:
        """获取专业推理"""
        # 如果有LLM接口，使用智能专业推理生成
        if self.llm_interface:
            try:
                # 使用LLM生成角色特异性的专业推理
                reasoning = self.llm_interface.generate_professional_reasoning(
                    patient_state=patient_state,
                    role=self.role,
                    treatment_option=treatment,
                    knowledge_context=knowledge
                )
                
                if reasoning and len(reasoning.strip()) > 0:
                    logger.debug(f"Generated LLM professional reasoning for {self.role.value}: {reasoning[:100]}...")
                    return reasoning
                    
            except Exception as e:
                logger.warning(f"LLM professional reasoning generation failed for {self.role}: {e}")
        
        # 降级到扩展的模板化专业推理
        reasoning_map = {
            RoleType.ONCOLOGIST: {
                TreatmentOption.SURGERY: f"surgical resection offers the best chance for cure in {patient_state.diagnosis} stage {patient_state.stage}, with acceptable operative risk given patient's age {patient_state.age}",
                TreatmentOption.CHEMOTHERAPY: f"systemic therapy is essential for micrometastatic disease control in {patient_state.diagnosis}, considering the patient's performance status and comorbidities {patient_state.comorbidities}",
                TreatmentOption.RADIOTHERAPY: f"radiation therapy provides excellent local control for {patient_state.diagnosis} with minimal systemic toxicity",
                TreatmentOption.IMMUNOTHERAPY: f"immunotherapy shows promising results for {patient_state.diagnosis} with manageable side effect profile",
                TreatmentOption.PALLIATIVE_CARE: f"palliative care is appropriate given the advanced stage and patient's quality of life priorities"
            },
            RoleType.NURSE: {
                TreatmentOption.SURGERY: f"this patient appears capable of handling the post-operative care requirements based on current functional status and support system",
                TreatmentOption.CHEMOTHERAPY: f"we need to carefully monitor for side effects and ensure compliance, considering the patient's current symptoms {patient_state.symptoms}",
                TreatmentOption.RADIOTHERAPY: f"radiation therapy is well-tolerated with manageable daily treatment schedule",
                TreatmentOption.IMMUNOTHERAPY: f"immunotherapy requires careful monitoring but has fewer immediate side effects than traditional chemotherapy",
                TreatmentOption.PALLIATIVE_CARE: f"palliative care focuses on comfort and quality of life, which aligns with comprehensive nursing care goals"
            },
            RoleType.PSYCHOLOGIST: {
                TreatmentOption.SURGERY: f"surgery may cause anxiety but offers hope for cure, which can positively impact psychological well-being",
                TreatmentOption.CHEMOTHERAPY: f"chemotherapy's side effects may impact mood and cognition, requiring psychological support throughout treatment",
                TreatmentOption.RADIOTHERAPY: f"radiation therapy's daily schedule provides routine while minimizing psychological burden",
                TreatmentOption.IMMUNOTHERAPY: f"immunotherapy's novel approach may provide hope while requiring adjustment to new treatment paradigm",
                TreatmentOption.PALLIATIVE_CARE: f"palliative care addresses psychological distress and helps with acceptance and coping"
            },
            RoleType.RADIOLOGIST: {
                TreatmentOption.SURGERY: f"imaging findings support surgical feasibility with clear anatomical landmarks for resection",
                TreatmentOption.CHEMOTHERAPY: f"baseline imaging establishes measurable disease for treatment response monitoring",
                TreatmentOption.RADIOTHERAPY: f"imaging-guided radiation planning ensures optimal target coverage while sparing critical structures",
                TreatmentOption.IMMUNOTHERAPY: f"imaging will be crucial for monitoring immune-related response patterns",
                TreatmentOption.PALLIATIVE_CARE: f"imaging can guide palliative interventions for symptom management"
            },
            RoleType.PATIENT_ADVOCATE: {
                TreatmentOption.SURGERY: f"surgery aligns with patient's desire for aggressive treatment while respecting informed consent",
                TreatmentOption.CHEMOTHERAPY: f"chemotherapy provides active treatment option while maintaining patient autonomy in decision-making",
                TreatmentOption.RADIOTHERAPY: f"radiation therapy offers effective treatment with preserved quality of life",
                TreatmentOption.IMMUNOTHERAPY: f"immunotherapy represents cutting-edge care that patients often prefer when available",
                TreatmentOption.PALLIATIVE_CARE: f"palliative care honors patient's values and preferences for comfort-focused care"
            },
        }

        role_reasoning = reasoning_map.get(self.role, {})
        template_reasoning = role_reasoning.get(treatment, f"from {self.role.value} perspective, {treatment.value} is a reasonable treatment option for this patient")
        
        logger.debug(f"Using template professional reasoning for {self.role.value}")
        return template_reasoning

    def _generate_counter_argument(
        self,
        treatment: TreatmentOption,
        opposing_view: Dict[str, Any],
        knowledge: Dict[str, Any],
    ) -> str:
        """生成反驳论据"""
        # 如果有LLM接口，使用智能反驳论据生成
        if self.llm_interface:
            try:
                # 提取反对观点的内容
                opposing_content = opposing_view.get('content', '') if opposing_view else ''
                opposing_role = opposing_view.get('role', '') if opposing_view else ''
                
                # 使用LLM生成有说服力的反驳论据
                counter_argument = self.llm_interface.generate_counter_argument(
                    role=self.role,
                    treatment_option=treatment,
                    opposing_view=opposing_content,
                    opposing_role=opposing_role,
                    knowledge_context=knowledge
                )
                
                if counter_argument and len(counter_argument.strip()) > 0:
                    logger.debug(f"Generated LLM counter argument for {self.role.value}: {counter_argument[:100]}...")
                    return counter_argument
                    
            except Exception as e:
                logger.warning(f"LLM counter argument generation failed for {self.role}: {e}")
        
        # 降级到扩展的模板化反驳论据
        opposing_content = opposing_view.get('content', '') if opposing_view else ''
        
        # 基于治疗类型和角色的反驳论据
        counter_arguments = {
            RoleType.ONCOLOGIST: {
                TreatmentOption.SURGERY: f"the survival benefit and potential for cure with {treatment.value} outweighs the surgical risks in this case, especially given the patient's overall condition",
                TreatmentOption.CHEMOTHERAPY: f"the systemic control achieved with {treatment.value} is essential for long-term outcomes, and side effects can be managed with modern supportive care",
                TreatmentOption.RADIATION: f"the precision of modern {treatment.value} techniques minimizes toxicity while maximizing local control",
                TreatmentOption.IMMUNOTHERAPY: f"the durable responses seen with {treatment.value} justify the treatment approach, with manageable immune-related adverse events",
                TreatmentOption.PALLIATIVE_CARE: f"focusing on {treatment.value} doesn't preclude future treatment options and prioritizes patient comfort and dignity"
            },
            RoleType.NURSE: {
                TreatmentOption.SURGERY: f"with proper pre-operative preparation and post-operative monitoring, the concerns about {treatment.value} can be effectively managed",
                TreatmentOption.CHEMOTHERAPY: f"our nursing protocols for {treatment.value} ensure patient safety and quality of life throughout treatment",
                TreatmentOption.RADIATION: f"the daily treatment schedule for {treatment.value} is manageable with appropriate patient education and support",
                TreatmentOption.IMMUNOTHERAPY: f"nursing surveillance for {treatment.value} side effects allows for early intervention and optimal outcomes",
                TreatmentOption.PALLIATIVE_CARE: f"comprehensive nursing care in {treatment.value} addresses both physical and emotional needs effectively"
            },
            RoleType.PSYCHOLOGIST: {
                TreatmentOption.SURGERY: f"while {treatment.value} may cause initial anxiety, the hope for cure provides significant psychological benefit",
                TreatmentOption.CHEMOTHERAPY: f"psychological support during {treatment.value} can help patients cope with side effects and maintain treatment adherence",
                TreatmentOption.RADIOTHERAPY: f"the routine of {treatment.value} can provide structure and hope during a difficult time",
                TreatmentOption.IMMUNOTHERAPY: f"patients often feel empowered by choosing cutting-edge {treatment.value}, which positively impacts their mental state",
                TreatmentOption.PALLIATIVE_CARE: f"{treatment.value} allows patients to focus on meaningful relationships and experiences, reducing psychological distress"
            },
            RoleType.RADIOLOGIST: {
                TreatmentOption.SURGERY: f"imaging findings clearly support the feasibility of {treatment.value} with acceptable anatomical considerations",
                TreatmentOption.CHEMOTHERAPY: f"baseline imaging provides excellent measurable targets for monitoring {treatment.value} response",
                TreatmentOption.RADIOTHERAPY: f"advanced imaging guidance ensures precise {treatment.value} delivery while protecting normal tissues",
                TreatmentOption.IMMUNOTHERAPY: f"imaging can effectively monitor the unique response patterns associated with {treatment.value}",
                TreatmentOption.PALLIATIVE_CARE: f"imaging-guided interventions can enhance the effectiveness of {treatment.value} approaches"
            },
            RoleType.PATIENT_ADVOCATE: {
                TreatmentOption.SURGERY: f"the patient has expressed preference for aggressive treatment, and {treatment.value} respects their autonomous decision-making",
                TreatmentOption.CHEMOTHERAPY: f"patients deserve access to active treatment options like {treatment.value} when medically appropriate",
                TreatmentOption.RADIOTHERAPY: f"{treatment.value} offers effective treatment while preserving quality of life, aligning with patient values",
                TreatmentOption.IMMUNOTHERAPY: f"patients should have access to innovative treatments like {treatment.value} when they meet criteria",
                TreatmentOption.PALLIATIVE_CARE: f"{treatment.value} honors the patient's right to comfort-focused care and personal values"
            },
        }

        role_arguments = counter_arguments.get(self.role, {})
        template_argument = role_arguments.get(treatment, f"I believe the benefits of {treatment.value} justify proceeding, considering the patient's overall situation")
        
        logger.debug(f"Using template counter argument for {self.role.value}")
        return template_argument

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
    
    # RL指导相关方法
    def set_rl_guidance(self, rl_guidance, influence_strength: float):
        """设置RL指导信息"""
        self.rl_guidance = rl_guidance
        self.rl_influence_strength = influence_strength
        self.rl_guidance_history.append({
            'guidance': rl_guidance,
            'influence_strength': influence_strength,
            'timestamp': datetime.now()
        })
        logger.debug(f"{self.role.value} received RL guidance with strength {influence_strength:.3f}")
    
    def generate_rl_influenced_opinion(
        self, 
        patient_state: PatientState, 
        knowledge: Dict[str, Any]
    ) -> RoleOpinion:
        """生成受RL影响的初始意见"""
        # 首先生成原始意见
        original_opinion = self.generate_initial_opinion(patient_state, knowledge)
        self.original_preferences = original_opinion.treatment_preferences.copy()
        
        if self.rl_guidance is None or self.rl_influence_strength == 0.0:
            return original_opinion
        
        # 应用RL影响
        influenced_preferences = self._apply_rl_influence_to_preferences(
            original_opinion.treatment_preferences
        )
        
        # 更新推荐治疗
        influenced_treatment = max(influenced_preferences.items(), key=lambda x: x[1])[0]
        
        # 调整置信度
        influenced_confidence = self._calculate_rl_influenced_confidence(
            original_opinion.confidence, influenced_treatment
        )
        
        # 生成新的推理
        influenced_reasoning = self._generate_rl_influenced_reasoning(
            original_opinion.reasoning, influenced_treatment
        )
        
        return RoleOpinion(
            role=self.role,
            treatment_preferences=influenced_preferences,
            reasoning=influenced_reasoning,
            confidence=influenced_confidence,
            concerns=original_opinion.concerns,
        )
    
    def _apply_rl_influence_to_preferences(
        self, 
        original_preferences: Dict[TreatmentOption, float]
    ) -> Dict[TreatmentOption, float]:
        """将RL影响应用到治疗偏好"""
        influenced_preferences = original_preferences.copy()
        
        if self.rl_guidance is None:
            return influenced_preferences
        
        # 获取RL推荐的治疗方案
        rl_recommended = self.rl_guidance.recommended_treatment
        
        # 计算影响因子
        influence_factor = self.rl_influence_strength * self.rl_guidance.confidence
        
        # 对RL推荐的治疗方案增加偏好
        if rl_recommended in influenced_preferences:
            original_score = influenced_preferences[rl_recommended]
            boost = influence_factor * (1.0 - abs(original_score))  # 避免过度增强
            influenced_preferences[rl_recommended] = np.clip(
                original_score + boost, -1.0, 1.0
            )
        
        # 基于RL价值估计调整其他治疗方案
        for treatment, rl_value in self.rl_guidance.value_estimates.items():
            if treatment != rl_recommended and treatment in influenced_preferences:
                original_score = influenced_preferences[treatment]
                
                # 根据RL价值估计调整
                value_adjustment = (rl_value - 0.5) * influence_factor * 0.3
                influenced_preferences[treatment] = np.clip(
                    original_score + value_adjustment, -1.0, 1.0
                )
        
        return influenced_preferences
    
    def _calculate_rl_influenced_confidence(
        self, 
        original_confidence: float, 
        influenced_treatment: TreatmentOption
    ) -> float:
        """计算受RL影响的置信度"""
        if self.rl_guidance is None:
            return original_confidence
        
        # 如果选择了RL推荐的治疗，增加置信度
        if influenced_treatment == self.rl_guidance.recommended_treatment:
            rl_confidence_boost = self.rl_guidance.confidence * self.rl_influence_strength * 0.2
            return np.clip(original_confidence + rl_confidence_boost, 0.0, 1.0)
        
        # 如果选择了其他治疗，根据RL价值估计调整置信度
        rl_value = self.rl_guidance.value_estimates.get(influenced_treatment, 0.5)
        value_factor = (rl_value - 0.5) * 2  # 转换到[-1, 1]范围
        confidence_adjustment = value_factor * self.rl_influence_strength * 0.1
        
        return np.clip(original_confidence + confidence_adjustment, 0.0, 1.0)
    
    def _generate_rl_influenced_reasoning(
        self, 
        original_reasoning: str, 
        influenced_treatment: TreatmentOption
    ) -> str:
        """生成受RL影响的推理"""
        if self.rl_guidance is None or self.rl_influence_strength < 0.1:
            return original_reasoning
        
        rl_reasoning_parts = []
        
        # 如果选择了RL推荐的治疗
        if influenced_treatment == self.rl_guidance.recommended_treatment:
            rl_reasoning_parts.append(
                f"数据分析支持{influenced_treatment.value}，"
                f"RL系统基于历史案例显示此方案具有较高价值（置信度{self.rl_guidance.confidence:.2f}）。"
            )
        else:
            # 如果选择了其他治疗，解释为什么
            rl_value = self.rl_guidance.value_estimates.get(influenced_treatment, 0.5)
            rl_reasoning_parts.append(
                f"虽然RL系统推荐{self.rl_guidance.recommended_treatment.value}，"
                f"但从{self.role.value}专业角度，{influenced_treatment.value}更适合"
                f"（RL价值评估：{rl_value:.2f}）。"
            )
        
        # 添加RL推理到原始推理
        if self.rl_guidance.reasoning:
            rl_reasoning_parts.append(f"RL分析要点：{self.rl_guidance.reasoning}")
        
        # 组合推理
        combined_reasoning = original_reasoning
        if rl_reasoning_parts:
            combined_reasoning += " " + " ".join(rl_reasoning_parts)
        
        return combined_reasoning
    
    def generate_rl_guided_response(
        self,
        patient_state: PatientState,
        knowledge: Dict[str, Any],
        dialogue_rounds: List,
        focus_treatment: TreatmentOption,
        rl_guidance
    ) -> Dict[str, Any]:
        """生成RL指导的对话响应"""
        # 基础响应生成
        base_message = self.generate_dialogue_response(
            patient_state, knowledge, 
            [msg for round_data in dialogue_rounds for msg in round_data.messages],
            focus_treatment
        )
        
        # 如果没有RL指导，返回基础响应
        if rl_guidance is None or self.rl_influence_strength < 0.1:
            return {
                "content": base_message.content,
                "treatment_focus": base_message.treatment_focus,
                "referenced_roles": base_message.referenced_roles,
                "evidence_cited": base_message.evidence_cited
            }
        
        # 生成RL增强的内容
        rl_enhanced_content = self._enhance_content_with_rl(
            base_message.content, focus_treatment, rl_guidance
        )
        
        return {
            "content": rl_enhanced_content,
            "treatment_focus": base_message.treatment_focus,
            "referenced_roles": base_message.referenced_roles,
            "evidence_cited": base_message.evidence_cited
        }
    
    def _enhance_content_with_rl(
        self, 
        base_content: str, 
        focus_treatment: TreatmentOption, 
        rl_guidance
    ) -> str:
        """用RL信息增强内容"""
        enhanced_parts = [base_content]
        
        # 添加RL相关的观点
        if focus_treatment == rl_guidance.recommended_treatment:
            enhanced_parts.append(
                f"这与数据驱动的分析一致，RL系统基于大量历史案例支持此选择。"
            )
        elif rl_guidance.confidence > 0.7:
            rl_value = rl_guidance.value_estimates.get(focus_treatment, 0.5)
            enhanced_parts.append(
                f"虽然数据分析倾向于{rl_guidance.recommended_treatment.value}，"
                f"但{focus_treatment.value}在当前情况下仍有其价值（评估值：{rl_value:.2f}）。"
            )
        
        return " ".join(enhanced_parts)
    
    def get_rl_influence_summary(self) -> Dict[str, Any]:
        """获取RL影响摘要"""
        return {
            "current_rl_influence": self.rl_influence_strength,
            "has_rl_guidance": self.rl_guidance is not None,
            "guidance_history_count": len(self.rl_guidance_history),
            "original_vs_current": {
                "original_preferences": self.original_preferences,
                "current_stance": self.current_stance
            } if self.original_preferences else None
        }
