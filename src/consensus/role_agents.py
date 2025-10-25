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
import re

from ..utils.llm_interface import LLMConfig, LLMInterface

from ..core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    RoleOpinion,
    DialogueMessage,
    MedicalEvent,
    ChatRole,
)

logger = logging.getLogger(__name__)


class RoleAgent:
    """角色智能体基类"""

    def __init__(self, role: RoleType, llm_interface: Optional[LLMInterface] = None, llm_config: Optional[LLMConfig] = None):
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
        self.specialization = self._get_specialization() # 通过内部方法获取实例的"专业领域"
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

    def generate_initial_opinion(
        self, patient_state: PatientState, knowledge: Dict[str, Any]
    ) -> RoleOpinion:
        """生成初始意见"""
        
        # 我可以调用大模型吗? 可以，基于患者状态和相关知识，调用llm_interface.generate_text方法生成初始意见
        # 这个初始意见是怎么生成的? 基于患者状态和相关知识，调用_calculate_treatment_preferences和_generate_reasoning方法
        # treatment_prefs是什么？ 它是一个字典，键是治疗选项，值是该选项的偏好评分
        treatment_prefs = self._calculate_treatment_preferences(
            patient_state, knowledge
        )
        logger.debug(f"[{self.role.value}] Generated initial treatment preferences: {treatment_prefs}")
        
        # reasoning是什么？ 它是一个字符串，描述了基于患者状态和治疗偏好的推理过程
        reasoning = self._generate_reasoning(patient_state, treatment_prefs, knowledge)
        logger.debug(f"[{self.role.value}] Generated initial reasoning: {reasoning}")

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
                    logger.debug(f"Generated LLM reasoning for {self.role.value}: {reasoning}...")
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
            RoleType.NUTRITIONIST: 0.8,  # 营养师在治疗支持中的重要性
            RoleType.REHABILITATION_THERAPIST: 0.8,  # 康复师在功能恢复中的重要性
        }

        base_confidence = role_relevance.get(self.role, 0.7)

        return base_confidence * data_completeness * complexity_penalty

    def _identify_concerns(self, patient_state: PatientState) -> List[str]:
        """识别关注点"""
        concerns = []

        # 通用关注点
        if patient_state.age > 70:
            concerns.append("高龄患者")

        if len(patient_state.comorbidities) > 2:
            concerns.append("合并症较多")

        if patient_state.quality_of_life_score < 0.5:
            concerns.append("生活质量较差")

        # 角色特异性关注点（中文）
        role_specific_concerns = {
            RoleType.ONCOLOGIST: ["疾病进展", "治疗耐药性"],
            RoleType.NURSE: ["患者依从性", "护理复杂度"],
            RoleType.PSYCHOLOGIST: ["心理负担", "家庭压力"],
            RoleType.RADIOLOGIST: ["解剖结构限制", "技术可行性"],
            RoleType.PATIENT_ADVOCATE: ["患者自主权", "知情同意"],
            RoleType.NUTRITIONIST: ["营养不良", "饮食管理难度"],
            RoleType.REHABILITATION_THERAPIST: ["术后康复挑战", "功能障碍风险"],
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
            patient_state, knowledge, target_treatment, opposing_views, supporting_views, full_dialogue_context=dialogue_context
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
        logger.debug(f"分析反对观点： {self.role.value}_stance: {my_stance}")

        for message in dialogue_context:
            if message.role != self.role and message.treatment_focus == treatment:
                # 更稳健的反对判定：中英双语关键词 + 无担忧排除
                content_lower = (message.content or "").lower()

                no_concern_markers = (
                    # 英文
                    "no concern", "not a concern", "no significant concern", "little concern", "low concern",
                    # 中文
                    "无担忧", "不是担忧", "没有显著担忧", "担忧较少", "担忧低",
                    "无顾虑", "不成问题", "没有明显问题"
                )
                negative_markers = (
                    # 英文
                    "not recommend", "contraindicated", "oppose", "against", "avoid",
                    "do not", "not advised", "no benefit", "concern", "high risk", "risk outweighs",
                    # 中文
                    "不推荐", "禁忌", "反对", "不支持", "避免", "不要",
                    "不建议", "无获益", "担忧", "高风险", "风险大于获益", "风险超过获益",
                    "不宜", "不适合"
                )

                has_no_concern = any(k in content_lower for k in no_concern_markers)
                is_negative = any(k in content_lower for k in negative_markers) and not has_no_concern

                if is_negative and my_stance > 0:
                    opposing_views.append(
                        {
                            "role": message.role,
                            "content": message.content,
                            "stance": "opposing",
                        }
                    )
        logger.debug(f"[{self.role.value}] 分析反对观点： {opposing_views}")
        # logger.debug(f"Detected opposing view: {message.content}...")   
        return opposing_views

    def _analyze_supporting_views(
        self, dialogue_context: List[DialogueMessage], treatment: TreatmentOption
    ) -> List[Dict[str, Any]]:
        """分析支持观点"""
        supporting_views = []
        my_stance = self.current_stance.get(treatment, 0)

        for message in dialogue_context:
            if message.role != self.role and message.treatment_focus == treatment:
                content_lower = (message.content or "").lower()

                # 负向关键词（中英双语）
                negative_markers = (
                    # 英文
                    "not recommend", "contraindicated", "oppose", "against", "avoid",
                    "do not", "not advised", "no benefit", "concern", "high risk", "risk outweighs",
                    # 中文
                    "不推荐", "禁忌", "反对", "不支持", "避免", "不要",
                    "不建议", "无获益", "担忧", "高风险", "风险大于获益", "风险超过获益",
                    "不宜", "不适合"
                )

                # 正向关键词（中英双语）
                positive_markers = (
                    # 英文
                    "recommend", "suggest", "advocate", "support", "favor", "prefer", "indicated",
                    # 中文
                    "推荐", "建议", "倡导", "支持", "赞成", "偏好", "适应证", "指征"
                )

                has_negative = any(k in content_lower for k in negative_markers)
                has_positive = any(k in content_lower for k in positive_markers)

                # 保证不包含负向词，且出现正向词，并且本角色当前是正向立场
                if has_positive and not has_negative and my_stance > 0:
                    supporting_views.append(
                        {
                            "role": message.role,
                            "content": message.content,
                            "stance": "supporting",
                        }
                    )
        logger.debug(f"分析支持观点： my_stance: {my_stance}")
        logger.debug(f"supporting_views: {supporting_views}")

        return supporting_views

    def _construct_response(
        self,
        patient_state: PatientState,
        knowledge: Dict[str, Any],
        treatment: TreatmentOption,
        opposing_views: List[Dict],
        supporting_views: List[Dict],
        full_dialogue_context: Optional[List[DialogueMessage]] = None,
    ) -> str:
        """构建回应内容 - 增强版本，减少模板化"""
        
        # 优先使用LLM生成自然对话
        if self.llm_interface:
            try:
                # 构建丰富的对话上下文
                dialogue_context = self._build_rich_dialogue_context(
                    patient_state, opposing_views, supporting_views, knowledge
                )
                
                # 使用LLM生成自然对话回应
                response = self.llm_interface.generate_dialogue_response(
                    patient_state=patient_state,
                    role=self.role,
                    treatment_option=treatment,
                    discussion_context=dialogue_context,
                    knowledge_context=knowledge,
                    current_stance=self.current_stance,
                    dialogue_history=self._get_recent_dialogue_history(full_dialogue_context)
                )
                
                if response and len(response.strip()) > 0:
                    logger.debug(f"Generated LLM response for {self.role.value}: {response}...")
                    return response
                    
            except Exception as e:
                logger.warning(f"LLM response generation failed for {self.role}: {e}")

        # 智能化的降级方案 - 动态构建回应
        return self._construct_intelligent_fallback_response(
            patient_state, treatment, opposing_views, supporting_views, knowledge
        )
    
    def _build_rich_dialogue_context(
        self,
        patient_state: PatientState,
        opposing_views: List[Dict],
        supporting_views: List[Dict],
        knowledge: Dict[str, Any]
    ) -> str:
        """构建丰富的对话上下文（中英双语与更稳健的关键词标签）"""
        context_parts = []

        # 患者关键信息（中英双语标签）
        try:
            age_en = f"{patient_state.age}y"
            age_zh = f"{patient_state.age}岁"
            diagnosis = patient_state.diagnosis
            stage = patient_state.stage
            context_parts.append(f"Patient/患者: {age_en} {diagnosis} stage/分期 {stage} ({age_zh})")
        except Exception:
            context_parts.append("Patient/患者: N/A")

        # 关键临床指标（QoL与合并症的双语提示）
        try:
            if patient_state.quality_of_life_score < 0.5:
                context_parts.append("Poor QoL/生活质量较差")
        except Exception:
            pass
        try:
            if len(getattr(patient_state, 'comorbidities', []) or []) > 2:
                context_parts.append("Multiple comorbidities/多合并症")
        except Exception:
            pass

        # 对话历史摘要（中英双语标签）
        if opposing_views:
            opposing_summary = self._summarize_viewpoints(opposing_views, "opposing")
            if opposing_summary:
                context_parts.append(f"Opposition/反对: {opposing_summary}")

        if supporting_views:
            supporting_summary = self._summarize_viewpoints(supporting_views, "supporting")
            if supporting_summary:
                context_parts.append(f"Support/支持: {supporting_summary}")

        return " | ".join(context_parts)

    def _summarize_viewpoints(self, views: List[Dict], view_type: str) -> str:
        """总结观点（中英双语，否定识别与关键词更稳健）"""
        if not views:
            return ""

        key_concerns = []

        # 关键词与否定模式（英中混合）
        risk_patterns = [
            r"\brisk\b", r"\bconcern(s)?\b", r"\bworry(ing)?\b",
            r"adverse", r"side effect(s)?", r"toxicit(y|ies)",
            r"风险", r"担忧", r"顾虑", r"副作用", r"不良", r"毒性"
        ]
        benefit_patterns = [
            r"\bbenefit(s)?\b", r"\beffective(ness)?\b", r"\befficacy\b",
            r"\bresponse\b", r"\bimprov(e|ement)\b", r"\bsurvival\b",
            r"获益", r"有效", r"疗效", r"反应", r"改善", r"生存"
        ]
        qol_patterns = [
            r"quality of life", r"\bqol\b", r"life quality",
            r"生活质量", r"生存质量"
        ]
        safety_patterns = [
            r"\bsafety\b", r"\bsafe\b", r"\btolerable\b",
            r"安全性", r"安全", r"耐受", r"可耐受"
        ]
        negation_patterns = [
            r"no\s+(risk|concern|worr(y|ies)|adverse|side effect(s)?|toxicit(y|ies))",
            r"not\s+(concerned|worried)",
            r"without\s+(risk|concern|toxicity|adverse)",
            r"free of\s+(risk|toxicit(y|ies))",
            r"lack of\s+concern",
            r"无风险", r"不担心", r"无担忧", r"无需担忧", r"无顾虑", r"无明显风险"
        ]
        low_risk_patterns = [
            r"low\s+risk", r"风险较低", r"风险不高"
        ]

        def any_match(patterns: List[str], text: str) -> bool:
            return any(re.search(p, text) for p in patterns)

        for view in views[:2]:  # 只看前两个观点
            content_raw = view.get('content', '')
            if not isinstance(content_raw, str):
                continue
            content = content_raw.lower()
            role = view.get('role', '')

            # 优先识别否定或安全倾向
            if any_match(negation_patterns, content) or any_match(safety_patterns, content):
                key_concerns.append(f"{role} indicates safety/表示安全性较好")
                if any_match(benefit_patterns, content):
                    key_concerns.append(f"{role} sees benefits/看到获益")
                if any_match(qol_patterns, content):
                    key_concerns.append(f"{role} focuses on QoL/关注生活质量")
                continue

            # 风险识别（排除明确否定）
            if any_match(risk_patterns, content) and not any_match(negation_patterns, content):
                key_concerns.append(f"{role} raises risks/提出风险")

            # 获益
            if any_match(benefit_patterns, content):
                key_concerns.append(f"{role} sees benefits/看到获益")

            # 生活质量
            if any_match(qol_patterns, content):
                key_concerns.append(f"{role} focuses on QoL/关注生活质量")

            # 低风险提示（作为补充）
            if any_match(low_risk_patterns, content):
                key_concerns.append(f"{role} notes low risk/风险较低")

        # 去重并截断
        unique = []
        for item in key_concerns:
            if item not in unique:
                unique.append(item)
        return ", ".join(unique[:3])  # 最多3个关注点
    
    def _construct_intelligent_fallback_response(
        self,
        patient_state: PatientState,
        treatment: TreatmentOption,
        opposing_views: List[Dict],
        supporting_views: List[Dict],
        knowledge: Dict[str, Any],
        dialogue_patterns: Dict[str, Any] = None
    ) -> str:
        """构建智能化的降级回应 - 减少模板化"""
        
        # 动态分析当前讨论焦点
        discussion_focus = self._analyze_discussion_focus(opposing_views, supporting_views)
        
        # 考虑对话模式，避免重复表达
        avoid_phrases = []
        if dialogue_patterns and 'role_patterns' in dialogue_patterns:
            role_pattern = dialogue_patterns['role_patterns'].get(self.role.value, {})
            avoid_phrases = role_pattern.get('phrases', [])[:5]  # 避免最近使用的短语
        
        # 基于角色专业性和患者特征生成个性化开场
        opening = self._generate_contextual_opening(
            patient_state, treatment, discussion_focus, avoid_phrases
        )
        
        # 生成基于证据的核心论点
        core_argument = self._generate_evidence_based_argument(
            patient_state, treatment, knowledge, discussion_focus
        )
        
        # 如果有反对意见，生成针对性回应
        response_to_opposition = ""
        if opposing_views:
            response_to_opposition = self._generate_targeted_response(
                opposing_views, treatment, patient_state
            )
        
        # 组合回应，避免固定模板
        response_parts = [opening, core_argument]
        if response_to_opposition:
            response_parts.append(response_to_opposition)
        
        # 添加角色特异性的结论
        conclusion = self._generate_role_specific_conclusion(patient_state, treatment)
        if conclusion:
            response_parts.append(conclusion)
        
        logger.debug(f"Using intelligent fallback response for {self.role.value}")
        return " ".join(response_parts)
    
    def _analyze_discussion_focus(self, opposing_views: List[Dict], supporting_views: List[Dict]) -> str:
        """分析当前讨论焦点"""
        focus_keywords = []
        
        all_views = opposing_views + supporting_views
        for view in all_views:
            content = view.get('content', '').lower()
            
            # 识别讨论焦点
            if any(word in content for word in ['risk', 'safety', 'complication']):
                focus_keywords.append('safety')
            if any(word in content for word in ['efficacy', 'survival', 'outcome']):
                focus_keywords.append('efficacy')
            if any(word in content for word in ['quality', 'comfort', 'symptom']):
                focus_keywords.append('quality_of_life')
            if any(word in content for word in ['cost', 'resource', 'feasible']):
                focus_keywords.append('feasibility')
        
        # 返回最主要的讨论焦点
        if focus_keywords:
            return max(set(focus_keywords), key=focus_keywords.count)
        return 'general'
    
    def _generate_contextual_opening(
        self, patient_state: PatientState, treatment: TreatmentOption, focus: str, avoid_phrases: List[str] = None
    ) -> str:
        """生成基于上下文的动态开场白 - 大幅减少模板化"""
        
        import random
        
        # 基于患者特征的动态要素
        age_factor = "年轻" if patient_state.age < 50 else "年长" if patient_state.age > 70 else ""
        stage_factor = "早期" if patient_state.stage in ["I", "II"] else "晚期" if patient_state.stage in ["III", "IV"] else ""
        qol_factor = "生活质量较差" if patient_state.quality_of_life_score < 0.5 else "状态良好" if patient_state.quality_of_life_score > 0.8 else ""
        
        # 角色特异性的多样化开场方式
        role_openings = {
            RoleType.ONCOLOGIST: [
                f"这位{age_factor}患者的{patient_state.diagnosis}",
                f"针对{stage_factor}{patient_state.diagnosis}",
                f"从肿瘤学角度看这个病例",
                f"考虑到患者的临床特征",
                f"基于{patient_state.diagnosis}的治疗指南"
            ],
            RoleType.NURSE: [
                f"从护理实践来看",
                f"考虑到患者的日常护理需求",
                f"在实际照护中",
                f"从患者舒适度角度",
                f"护理团队的经验显示"
            ],
            RoleType.PSYCHOLOGIST: [
                f"从心理适应角度",
                f"考虑患者的心理状态",
                f"心理评估显示",
                f"从情感支持层面",
                f"患者的心理承受能力"
            ],
            RoleType.RADIOLOGIST: [
                f"影像学评估结果",
                f"从放射学角度",
                f"技术可行性方面",
                f"基于影像学发现",
                f"放射治疗的精准性"
            ],
            RoleType.PATIENT_ADVOCATE: [
                f"站在患者立场",
                f"从患者权益角度",
                f"考虑患者的价值观",
                f"尊重患者的选择",
                f"患者的最佳利益"
            ]
        }
        
        # 随机选择开场方式，增加多样性，避免重复
        openings = role_openings.get(self.role, [f"作为{self.role.value}"])
        
        # 过滤掉最近使用过的短语
        if avoid_phrases:
            filtered_openings = []
            for opening in openings:
                # 检查是否包含需要避免的短语
                contains_avoid_phrase = any(phrase in opening for phrase in avoid_phrases if phrase)
                if not contains_avoid_phrase:
                    filtered_openings.append(opening)
            openings = filtered_openings if filtered_openings else openings
        
        base_opening = random.choice(openings)
        
        # 根据讨论焦点添加特定关注点
        focus_additions = {
            'safety': ["安全性是关键考虑", "风险评估很重要", "需要谨慎评估风险"],
            'efficacy': ["疗效是核心问题", "治疗效果值得期待", "疗效数据支持"],
            'quality_of_life': ["生活质量不容忽视", "患者感受很重要", "舒适度是关键"],
            'feasibility': ["实施可行性", "操作层面的考虑", "实际执行中"]
        }
        
        # 30%概率添加焦点相关内容
        if focus in focus_additions and random.random() < 0.3:
            focus_addition = random.choice(focus_additions[focus])
            return f"{base_opening}，{focus_addition}，"
        
        return f"{base_opening}，"
    
    def _generate_evidence_based_argument(
        self,
        patient_state: PatientState,
        treatment: TreatmentOption,
        knowledge: Dict[str, Any],
        focus: str
    ) -> str:
        """生成基于证据的核心论点"""
        
        # 获取角色立场
        my_stance = self.current_stance.get(treatment, 0)
        
        # 基于立场和焦点生成论点
        if my_stance > 0.5:
            return self._generate_strong_support_argument(patient_state, treatment, focus)
        elif my_stance > 0:
            return self._generate_cautious_support_argument(patient_state, treatment, focus)
        elif my_stance < -0.5:
            return self._generate_strong_concern_argument(patient_state, treatment, focus)
        else:
            return self._generate_balanced_argument(patient_state, treatment, focus)
    
    def _generate_strong_support_argument(self, patient_state: PatientState, treatment: TreatmentOption, focus: str) -> str:
        """生成强烈支持的论点 - 动态化和个性化"""
        import random
        
        # 基于患者特征的动态要素
        age_consideration = f"{patient_state.age}岁的年龄" if patient_state.age > 70 or patient_state.age < 40 else ""
        stage_consideration = f"{patient_state.stage}期" if patient_state.stage else ""
        comorbidity_consideration = "合并症情况" if patient_state.comorbidities else ""
        
        # 角色特异性的支持论点模板池
        role_arguments = {
            RoleType.ONCOLOGIST: {
                'efficacy': [
                    f"{treatment.value}在{patient_state.diagnosis}治疗中显示出显著疗效",
                    f"临床数据支持{treatment.value}对此类患者的治疗效果",
                    f"基于循证医学，{treatment.value}是最佳选择",
                    f"多项研究证实{treatment.value}的生存获益"
                ],
                'safety': [
                    f"患者的临床状况适合{treatment.value}治疗",
                    f"风险效益比分析支持{treatment.value}",
                    f"安全性数据令人放心",
                    f"副作用可控且可管理"
                ]
            },
            RoleType.NURSE: {
                'feasibility': [
                    f"我们的护理团队完全有能力支持{treatment.value}",
                    f"从护理角度看，{treatment.value}是可行的",
                    f"患者的功能状态支持这个治疗方案",
                    f"护理流程已经很成熟"
                ],
                'quality_of_life': [
                    f"{treatment.value}有助于维持患者的生活质量",
                    f"从患者舒适度考虑，这是好选择",
                    f"能够保持患者的独立性",
                    f"日常生活影响相对较小"
                ]
            },
            RoleType.PSYCHOLOGIST: {
                'quality_of_life': [
                    f"从心理角度，{treatment.value}给患者带来希望",
                    f"患者的心理状态能够承受这个治疗",
                    f"心理适应性良好",
                    f"有助于患者保持积极心态"
                ]
            },
            RoleType.RADIOLOGIST: {
                'safety': [
                    f"影像学评估支持{treatment.value}的安全性",
                    f"技术条件完全满足要求",
                    f"解剖结构适合这种治疗方式",
                    f"精准度可以得到保证"
                ]
            },
            RoleType.PATIENT_ADVOCATE: {
                'general': [
                    f"{treatment.value}符合患者的最佳利益",
                    f"这个选择尊重了患者的价值观",
                    f"从患者权益角度完全支持",
                    f"患者有权获得最佳治疗"
                ]
            }
        }
        
        # 获取角色特定的论点
        role_args = role_arguments.get(self.role, {})
        focus_args = role_args.get(focus, role_args.get('general', []))
        
        if focus_args:
            base_argument = random.choice(focus_args)
        else:
            # 通用支持论点
            generic_supports = [
                f"{treatment.value}是这个患者的最佳选择",
                f"我强烈推荐{treatment.value}",
                f"综合考虑，{treatment.value}最合适",
                f"专业判断支持{treatment.value}"
            ]
            base_argument = random.choice(generic_supports)
        
        # 随机添加患者特征相关的补充说明
        supplements = []
        if age_consideration and random.random() < 0.4:
            supplements.append(f"考虑到{age_consideration}")
        if stage_consideration and random.random() < 0.4:
            supplements.append(f"在{stage_consideration}的情况下")
        if comorbidity_consideration and random.random() < 0.3:
            supplements.append(f"尽管有{comorbidity_consideration}")
        
        if supplements:
            supplement = random.choice(supplements)
            return f"{supplement}，{base_argument}。"
        
        return f"{base_argument}。"
    
    def _generate_cautious_support_argument(self, patient_state: PatientState, treatment: TreatmentOption, focus: str) -> str:
        """生成谨慎支持的论点 - 动态化"""
        import random
        
        cautious_expressions = [
            f"{treatment.value}有一定优势，但需要密切关注",
            f"我倾向于支持{treatment.value}，不过要谨慎监测",
            f"{treatment.value}可以考虑，但要做好风险管理",
            f"支持{treatment.value}，同时要充分准备应对并发症"
        ]
        
        risk_factors = []
        if patient_state.age > 70:
            risk_factors.append("高龄因素")
        if patient_state.comorbidities:
            risk_factors.append("合并症负担")
        if patient_state.quality_of_life_score < 0.6:
            risk_factors.append("生活质量状况")
        
        base_statement = random.choice(cautious_expressions)
        if risk_factors:
            risk_mention = f"特别是{random.choice(risk_factors)}"
            return f"{base_statement}，{risk_mention}。"
        
        return f"{base_statement}。"
    
    def _generate_strong_concern_argument(self, patient_state: PatientState, treatment: TreatmentOption, focus: str) -> str:
        """生成强烈关注的论点 - 动态化"""
        import random
        
        concern_expressions = [
            f"我对{treatment.value}有较大担忧",
            f"{treatment.value}的风险可能过高",
            f"需要重新考虑{treatment.value}的适用性",
            f"我不太赞成{treatment.value}这个方案"
        ]
        
        specific_concerns = {
            'safety': [
                f"安全性风险不容忽视",
                f"副作用可能难以承受",
                f"风险效益比不理想"
            ],
            'quality_of_life': [
                f"对生活质量的影响太大",
                f"患者的舒适度会严重下降",
                f"生活质量评分已经不高"
            ],
            'efficacy': [
                f"疗效可能不如预期",
                f"获益有限",
                f"效果存在不确定性"
            ]
        }
        
        base_concern = random.choice(concern_expressions)
        specific_concern = random.choice(specific_concerns.get(focus, specific_concerns['safety']))
        
        # 添加患者特征相关的具体担忧
        patient_factors = []
        if patient_state.age > 75:
            patient_factors.append(f"{patient_state.age}岁的高龄")
        if len(patient_state.comorbidities) > 2:
            patient_factors.append("多重合并症")
        if patient_state.quality_of_life_score < 0.4:
            patient_factors.append("生活质量已经很差")
        
        if patient_factors:
            factor = random.choice(patient_factors)
            return f"{base_concern}，{specific_concern}，尤其是{factor}。"
        
        return f"{base_concern}，{specific_concern}。"
    
    def _generate_balanced_argument(self, patient_state: PatientState, treatment: TreatmentOption, focus: str) -> str:
        """生成平衡的论点 - 动态化"""
        import random
        
        balanced_expressions = [
            f"{treatment.value}有利有弊，需要仔细权衡",
            f"对于{treatment.value}，我持中性态度",
            f"{treatment.value}值得考虑，但要全面评估",
            f"需要更多信息来判断{treatment.value}的适用性"
        ]
        
        considerations = [
            "需要综合考虑各种因素",
            "建议多学科讨论",
            "患者的个人意愿也很重要",
            "可以进一步评估后决定"
        ]
        
        base_statement = random.choice(balanced_expressions)
        consideration = random.choice(considerations)
        
        return f"{base_statement}，{consideration}。"
    
    def _generate_targeted_response(
        self, opposing_views: List[Dict], treatment: TreatmentOption, patient_state: PatientState
    ) -> str:
        """生成针对性回应 - 动态化"""
        if not opposing_views:
            return ""
        
        import random
        
        # 分析反对意见的核心关注点
        main_concern = self._extract_main_concern(opposing_views[0])
        opposing_role = opposing_views[0].get('role', 'colleague')
        
        # 动态回应模板
        response_templates = {
            'risk': [
                f"我理解{opposing_role}对风险的担忧，但我认为通过适当的监测可以管理这些风险",
                f"虽然{opposing_role}提到了风险问题，但获益可能更大",
                f"关于{opposing_role}的风险顾虑，我们有相应的预防措施",
                f"我同意{opposing_role}的风险评估，但这些风险是可控的"
            ],
            'efficacy': [
                f"我对{opposing_role}的疗效评估有不同看法，类似病例的证据显示效果不错",
                f"虽然{opposing_role}质疑疗效，但我认为值得一试",
                f"关于{opposing_role}提到的疗效问题，我们可以密切观察",
                f"我理解{opposing_role}的疑虑，但现有证据还是支持的"
            ],
            'quality_of_life': [
                f"{opposing_role}的生活质量担忧很有道理，我们可以通过支持治疗来改善",
                f"我同意{opposing_role}的观点，生活质量确实重要，但我们有办法平衡",
                f"关于{opposing_role}提到的生活质量问题，我们会特别关注",
                f"虽然{opposing_role}担心生活质量，但长远来看可能是有益的"
            ],
            'general': [
                f"我很赞赏{opposing_role}的观点，但我们也要考虑其他因素",
                f"感谢{opposing_role}的提醒，不过我认为还有其他角度需要考虑",
                f"我理解{opposing_role}的立场，但我们需要综合评估",
                f"{opposing_role}说得有道理，但我们也要看到积极的一面"
            ]
        }
        
        # 随机选择回应模板
        templates = response_templates.get(main_concern, response_templates['general'])
        return random.choice(templates)
    
    def _extract_main_concern(self, opposing_view: Dict) -> str:
        """提取主要关注点"""
        content = opposing_view.get('content', '').lower()
        
        if any(word in content for word in ['risk', 'danger', 'complication']):
            return 'risk'
        elif any(word in content for word in ['ineffective', 'limited benefit', 'poor outcome']):
            return 'efficacy'
        elif any(word in content for word in ['quality of life', 'comfort', 'suffering']):
            return 'quality_of_life'
        
        return 'general'
    
    def _generate_role_specific_conclusion(self, patient_state: PatientState, treatment: TreatmentOption) -> str:
        """生成角色特异性结论 - 动态化"""
        import random
        
        role_conclusions = {
            RoleType.ONCOLOGIST: [
                f"从肿瘤学角度，{treatment.value}是当前最佳治疗选择",
                f"基于肿瘤特征，我推荐{treatment.value}",
                f"考虑到患者的肿瘤分期，{treatment.value}符合指南推荐",
                f"从抗肿瘤效果来看，{treatment.value}具有良好前景"
            ],
            RoleType.NURSE: [
                f"从护理角度，我们能够确保{treatment.value}的安全实施",
                f"护理团队有信心配合{treatment.value}的治疗",
                f"我们会全程关注患者在{treatment.value}期间的护理需求",
                f"从护理管理角度，{treatment.value}是可行的"
            ],
            RoleType.PSYCHOLOGIST: [
                f"从心理学角度，{treatment.value}考虑了患者的心理承受能力",
                f"这个方案有助于维护患者的心理健康",
                f"从心理支持角度，{treatment.value}是合适的选择",
                f"考虑到患者的心理状态，{treatment.value}是可以接受的"
            ],
            RoleType.RADIOLOGIST: [
                f"影像学检查支持{treatment.value}的选择",
                f"从影像学角度，{treatment.value}有充分的依据",
                f"影像学表现提示{treatment.value}是合理的",
                f"基于影像学评估，{treatment.value}具有可行性"
            ],
            RoleType.PATIENT_ADVOCATE: [
                f"这个决定充分尊重了患者的价值观和偏好",
                f"从患者权益角度，{treatment.value}是合适的",
                f"我们充分考虑了患者的意愿和需求",
                f"这个选择体现了以患者为中心的理念"
            ]
        }
        
        # 添加患者特征相关的个性化元素
        patient_factors = []
        if patient_state.age > 70:
            patient_factors.append("考虑到患者的年龄因素")
        if patient_state.comorbidities:
            patient_factors.append("结合患者的合并症情况")
        if patient_state.quality_of_life_score < 0.6:
            patient_factors.append("重视患者的生活质量")
        
        base_conclusions = role_conclusions.get(self.role, [f"从专业角度，{treatment.value}是合理的选择"])
        base_conclusion = random.choice(base_conclusions)
        
        if patient_factors and random.random() < 0.7:  # 70%概率添加患者因素
            factor = random.choice(patient_factors)
            return f"{base_conclusion}，{factor}。"
        
        return f"{base_conclusion}。"

    def _get_recent_dialogue_history(
        self,
        messages: Optional[List[DialogueMessage]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """提取最近对话历史供LLM提示使用（仅需role与content）"""
        history_list: List[Dict[str, Any]] = []
        source = messages if messages is not None else getattr(self, "dialogue_history", [])
        if not source:
            return history_list
        recent = source[-limit:] if len(source) > limit else source
        for item in recent:
            if isinstance(item, dict):
                role_name = item.get("role", "Unknown")
                content = item.get("content", "")
            else:
                role_name = item.role.value if hasattr(item, "role") and hasattr(item.role, "value") else str(getattr(item, "role", "Unknown"))
                content = getattr(item, "content", "")
            if not content:
                continue
            history_list.append({"role": role_name, "content": content})
        return history_list
    
    def _get_dialogue_history(self, discussion_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """获取对话历史，增强上下文感知"""
        dialogue_history = []
        
        # 从讨论上下文中提取历史对话
        if 'previous_rounds' in discussion_context:
            for round_data in discussion_context['previous_rounds']:
                if 'responses' in round_data:
                    for response in round_data['responses']:
                        dialogue_history.append({
                            'role': response.get('role', 'unknown'),
                            'content': response.get('content', ''),
                            'stance': response.get('stance', 'neutral'),
                            'round': round_data.get('round_number', 0)
                        })
        
        # 限制历史长度，保留最近的对话
        return dialogue_history[-10:] if len(dialogue_history) > 10 else dialogue_history
    
    def _analyze_dialogue_patterns(self, dialogue_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析对话模式，避免重复"""
        patterns = {
            'repeated_phrases': [],
            'common_stances': {},
            'role_patterns': {},
            'recent_topics': []
        }
        
        # 分析角色发言模式
        for entry in dialogue_history:
            role = entry['role']
            content = entry['content']
            stance = entry['stance']
            
            # 统计角色立场
            if role not in patterns['role_patterns']:
                patterns['role_patterns'][role] = {'stances': [], 'phrases': []}
            patterns['role_patterns'][role]['stances'].append(stance)
            
            # 提取常用短语（简化版）
            words = content.split()
            if len(words) > 3:
                patterns['role_patterns'][role]['phrases'].extend(words[:3])
        
        return patterns

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
                    logger.debug(f"Generated LLM professional reasoning for {self.role.value}: {reasoning}...")
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
                    logger.debug(f"Generated LLM counter argument for {self.role.value}: {counter_argument}...")
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

    # def update_stance_based_on_dialogue(
    #     self, dialogue_context: List[DialogueMessage]
    # ) -> None:
    #     """基于对话更新立场"""
    #     for treatment in TreatmentOption:
    #         supporting_messages = [
    #             m
    #             for m in dialogue_context
    #             if m.treatment_focus == treatment
    #             and m.role != self.role
    #             and "recommend" in m.content.lower()
    #         ]

    #         if len(supporting_messages) >= 2:  # 多数支持时轻微调整立场
    #             current_stance = self.current_stance.get(treatment, 0)
    #             self.current_stance[treatment] = min(1.0, current_stance + 0.1)
    
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
    
    def update_stance_based_on_dialogue(self, messages: List[DialogueMessage]) -> None:
        """根据对话内容更新立场（改进版：否定优先、证据加权、去重、防御空值）"""
        if not messages:
            return

        # 同一（发言角色, 治疗）仅影响一次，避免重复累积
        seen_pairs = set()

        for message in messages:
            # 防御空字段
            if not hasattr(message, "role") or not hasattr(message, "treatment_focus"):
                continue

            # 忽略非医疗对话角色（USER/SYSTEM）
            if isinstance(message.role, ChatRole):
                continue

            # 跳过自己的消息
            if message.role == self.role:
                continue

            treatment = message.treatment_focus
            if not treatment or treatment not in self.current_stance:
                continue

            # 去重：同一角色-同一治疗只应用一次影响
            pair = (message.role, treatment)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            content = (getattr(message, "content", "") or "").lower()

            # 证据强度加权（中英双语关键词）
            influence = 0.06  # 默认影响步长
            strong_evidence_markers = (
                # 英文
                "rct", "randomized", "randomised", "meta-analysis",
                "systematic review", "guideline", "practice guideline", "cochrane",
                # 中文
                "随机对照试验", "随机化", "随机研究", "meta分析",
                "系统综述", "指南", "实践指南", "循证指南",
                "科克伦", "考科兰", "cochrane系统综述"
            )
            moderate_markers = (
                # 英文
                "evidence", "study", "trial", "data",
                # 中文
                "证据", "研究", "试验", "数据", "临床研究", "临床试验"
            )
            case_markers = (
                # 英文
                "case report", "case series",
                # 中文
                "病例报告", "病例系列", "个案报告"
            )

            if any(k in content for k in strong_evidence_markers):
                influence = 0.15
            elif any(k in content for k in moderate_markers):
                influence = 0.10
            elif any(k in content for k in case_markers):
                influence = 0.08

            # 否定优先识别（中英双语），避免"不推荐/not recommend"被误判为正向
            no_concern_markers = (
                # 英文
                "no concern", "not a concern", "no significant concern", "little concern", "low concern",
                # 中文
                "无担忧", "不是担忧", "没有显著担忧", "担忧较少", "担忧低",
                "无顾虑", "不成问题", "没有明显问题"
            )
            has_no_concern = any(k in content for k in no_concern_markers)

            negative_markers = (
                # 英文
                "not recommend", "contraindicated", "oppose", "against", "avoid",
                "do not", "not advised", "no benefit", "concern", "high risk", "risk outweighs",
                # 中文
                "不推荐", "禁忌", "反对", "不支持", "避免", "不要",
                "不建议", "无获益", "担忧", "高风险", "风险大于获益",
                "风险超过获益", "不宜", "不适合"
            )
            positive_markers = (
                # 英文
                "recommend", "suggest", "advocate", "support", "favor", "prefer", "indicated",
                # 中文
                "推荐", "建议", "倡导", "支持", "赞成", "偏好", "适应证", "指征"
            )

            is_negative = any(k in content for k in negative_markers) and not has_no_concern
            # 更稳健的正向判定：不包含任何负向词，且出现任一正向词
            is_positive = (not any(k in content for k in negative_markers)) and any(k in content for k in positive_markers)

            if is_negative:
                self.current_stance[treatment] = max(-1.0, self.current_stance[treatment] - influence)
            elif is_positive:
                self.current_stance[treatment] = min(1.0, self.current_stance[treatment] + influence)
            else:
                # 无明确倾向，保持不变
                continue

        # 记录立场更新
        logger.debug(f"{self.role.value} updated stance based on dialogue: {self.current_stance}")
