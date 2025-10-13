"""
医学知识检索增强生成系统
文件路径: src/knowledge/rag_system.py
作者: 团队共同维护 (姚刚主要贡献)
功能: 实现医学知识的检索和上下文增强
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

from ..core.data_models import PatientState, TreatmentOption

logger = logging.getLogger(__name__)


class MedicalKnowledgeRAG:
    """医学知识检索增强生成系统"""

    def __init__(self, knowledge_base_path: str = None):
        self.knowledge_base = self._initialize_knowledge_base(knowledge_base_path)
        self.embedding_cache = {}

    def _initialize_knowledge_base(
        self, knowledge_base_path: Optional[str]
    ) -> Dict[str, Any]:
        """初始化医学知识库"""
        # 在实际项目中，这里会加载真实的医学知识库
        # 现在使用模拟的知识结构

        knowledge_base = {
            "disease_profiles": {
                "breast_cancer": {
                    "stage_i": {
                        "description": "Early stage breast cancer confined to breast tissue",
                        "standard_treatments": ["surgery", "adjuvant_therapy"],
                        "survival_rates": {"5_year": 0.99, "10_year": 0.95},
                        "contraindications": [],
                    },
                    "stage_ii": {
                        "description": "Locally advanced breast cancer",
                        "standard_treatments": [
                            "surgery",
                            "chemotherapy",
                            "radiotherapy",
                        ],
                        "survival_rates": {"5_year": 0.93, "10_year": 0.85},
                        "contraindications": ["severe_cardiac_dysfunction"],
                    },
                    "stage_iii": {
                        "description": "Regional lymph node involvement",
                        "standard_treatments": [
                            "neoadjuvant_chemotherapy",
                            "surgery",
                            "radiotherapy",
                        ],
                        "survival_rates": {"5_year": 0.72, "10_year": 0.58},
                        "contraindications": ["multiple_organ_dysfunction"],
                    },
                    "stage_iv": {
                        "description": "Metastatic breast cancer",
                        "standard_treatments": ["systemic_therapy", "palliative_care"],
                        "survival_rates": {"5_year": 0.29, "10_year": 0.13},
                        "contraindications": [],
                    },
                }
            },
            "treatment_guidelines": {
                "NCCN": {
                    "breast_cancer": {
                        "surgery": {
                            "indications": ["stage_i", "stage_ii", "stage_iii"],
                            "contraindications": [
                                "metastatic_disease",
                                "poor_performance_status",
                            ],
                            "evidence_level": "Category 1",
                            "recommendations": "Complete surgical resection when feasible",
                        },
                        "chemotherapy": {
                            "indications": ["stage_ii", "stage_iii", "stage_iv"],
                            "contraindications": [
                                "severe_organ_dysfunction",
                                "poor_performance_status",
                            ],
                            "evidence_level": "Category 1",
                            "recommendations": "Adjuvant or neoadjuvant based on stage",
                        },
                        "radiotherapy": {
                            "indications": ["post_surgical", "locally_advanced"],
                            "contraindications": ["prior_radiation", "pregnancy"],
                            "evidence_level": "Category 1",
                            "recommendations": "Standard after breast-conserving surgery",
                        },
                        "immunotherapy": {
                            "indications": ["triple_negative", "her2_positive"],
                            "contraindications": ["autoimmune_disorders"],
                            "evidence_level": "Category 2A",
                            "recommendations": "Consider in appropriate subgroups",
                        },
                    }
                },
                "ESMO": {
                    "breast_cancer": {
                        "general": "Similar to NCCN with European modifications",
                        "palliative_care": {
                            "indications": ["advanced_disease", "poor_prognosis"],
                            "evidence_level": "I-A",
                            "recommendations": "Integrate early in advanced disease",
                        },
                    }
                },
            },
            "drug_interactions": {
                "chemotherapy_diabetes": {
                    "interaction": "Chemotherapy may worsen glucose control",
                    "management": "Close glucose monitoring required",
                    "severity": "moderate",
                },
                "surgery_hypertension": {
                    "interaction": "Perioperative blood pressure management critical",
                    "management": "Optimize BP control preoperatively",
                    "severity": "moderate",
                },
            },
            "comorbidity_considerations": {
                "diabetes": {
                    "surgery": {"risk_increase": 1.3, "precautions": "glucose_control"},
                    "chemotherapy": {
                        "risk_increase": 1.2,
                        "precautions": "infection_monitoring",
                    },
                    "radiotherapy": {
                        "risk_increase": 1.1,
                        "precautions": "wound_healing",
                    },
                },
                "hypertension": {
                    "surgery": {"risk_increase": 1.2, "precautions": "bp_optimization"},
                    "chemotherapy": {
                        "risk_increase": 1.1,
                        "precautions": "cardiac_monitoring",
                    },
                },
                "cardiac_dysfunction": {
                    "surgery": {
                        "risk_increase": 2.0,
                        "precautions": "cardiac_clearance",
                    },
                    "chemotherapy": {
                        "risk_increase": 1.8,
                        "precautions": "cardio_oncology",
                    },
                },
            },
            "similar_cases": [
                {
                    "patient_profile": {
                        "age": 65,
                        "stage": "II",
                        "comorbidities": ["diabetes", "hypertension"],
                    },
                    "treatment_given": "surgery_plus_chemotherapy",
                    "outcome": {"survival_months": 48, "quality_of_life": 0.7},
                    "complications": ["wound_healing_delay", "neuropathy"],
                },
                {
                    "patient_profile": {
                        "age": 78,
                        "stage": "III",
                        "comorbidities": ["cardiac_dysfunction"],
                    },
                    "treatment_given": "palliative_care",
                    "outcome": {"survival_months": 12, "quality_of_life": 0.8},
                    "complications": [],
                },
            ],
        }

        logger.info("Medical knowledge base initialized")
        return knowledge_base

    def retrieve_relevant_knowledge(
        self,
        patient_state: PatientState,
        query_type: str,
        treatment_focus: TreatmentOption = None,
    ) -> Dict[str, Any]:
        """检索相关医学知识"""

        logger.debug(
            f"Retrieving knowledge for {query_type}, patient {patient_state.patient_id}"
        )

        relevant_knowledge = {
            "guidelines": self._get_treatment_guidelines(
                patient_state, treatment_focus
            ),
            "similar_cases": self._get_similar_cases(patient_state),
            "contraindications": self._get_contraindications(
                patient_state, treatment_focus
            ),
            "evidence_level": self._get_evidence_level(treatment_focus),
            "success_rates": self._get_success_rates(patient_state, treatment_focus),
            "side_effects": self._get_side_effects(treatment_focus),
            "comorbidity_considerations": self._get_comorbidity_considerations(
                patient_state
            ),
            "drug_interactions": self._get_drug_interactions(
                patient_state, treatment_focus
            ),
        }

        # 根据查询类型过滤信息
        if query_type == "initial_assessment":
            relevant_knowledge["priority"] = "comprehensive_evaluation"
        elif query_type == "treatment_discussion":
            relevant_knowledge["priority"] = "treatment_specific_details"
        elif query_type == "safety_review":
            relevant_knowledge["priority"] = "contraindications_and_risks"

        return relevant_knowledge

    def _get_treatment_guidelines(
        self, patient_state: PatientState, treatment: TreatmentOption = None
    ) -> List[str]:
        """获取治疗指南"""
        guidelines = []

        disease_key = patient_state.diagnosis
        stage_key = f"stage_{patient_state.stage.lower()}"

        # NCCN指南
        nccn_guidelines = self.knowledge_base.get("treatment_guidelines", {}).get(
            "NCCN", {}
        )
        disease_guidelines = nccn_guidelines.get(disease_key, {})

        if treatment:
            treatment_key = treatment.value
            if treatment_key in disease_guidelines:
                treatment_info = disease_guidelines[treatment_key]
                recommendation = treatment_info.get("recommendations", "")
                evidence_level = treatment_info.get("evidence_level", "")
                guidelines.append(f"NCCN {evidence_level}: {recommendation}")
        else:
            # 返回疾病阶段的一般指南
            disease_profile = self.knowledge_base.get("disease_profiles", {}).get(
                disease_key, {}
            )
            stage_info = disease_profile.get(stage_key, {})
            if stage_info:
                standard_treatments = stage_info.get("standard_treatments", [])
                guidelines.append(
                    f"Standard treatments for {stage_key}: {', '.join(standard_treatments)}"
                )

        return guidelines

    def _get_similar_cases(self, patient_state: PatientState) -> List[Dict[str, Any]]:
        """获取相似病例"""
        similar_cases = []

        patient_age = patient_state.age
        patient_stage = patient_state.stage
        patient_comorbidities = set(patient_state.comorbidities)

        for case in self.knowledge_base.get("similar_cases", []):
            case_profile = case["patient_profile"]

            # 计算相似度
            age_similarity = 1.0 - abs(patient_age - case_profile["age"]) / 50.0
            stage_similarity = 1.0 if patient_stage == case_profile["stage"] else 0.5

            case_comorbidities = set(case_profile.get("comorbidities", []))
            comorbidity_similarity = len(
                patient_comorbidities & case_comorbidities
            ) / max(1, len(patient_comorbidities | case_comorbidities))

            overall_similarity = (
                age_similarity + stage_similarity + comorbidity_similarity
            ) / 3.0

            if overall_similarity > 0.6:  # 相似度阈值
                case_info = case.copy()
                case_info["similarity_score"] = overall_similarity
                similar_cases.append(case_info)

        # 按相似度排序
        similar_cases.sort(key=lambda x: x["similarity_score"], reverse=True)

        return similar_cases[:3]  # 返回最相似的3个案例

    def _get_contraindications(
        self, patient_state: PatientState, treatment: TreatmentOption = None
    ) -> List[str]:
        """获取禁忌症"""
        contraindications = []

        if not treatment:
            return contraindications

        # 从疾病profile获取禁忌症
        disease_profile = self.knowledge_base.get("disease_profiles", {}).get(
            patient_state.diagnosis, {}
        )
        stage_info = disease_profile.get(f"stage_{patient_state.stage.lower()}", {})
        stage_contraindications = stage_info.get("contraindications", [])

        # 从治疗指南获取禁忌症
        nccn_guidelines = self.knowledge_base.get("treatment_guidelines", {}).get(
            "NCCN", {}
        )
        treatment_guidelines = nccn_guidelines.get(patient_state.diagnosis, {}).get(
            treatment.value, {}
        )
        guideline_contraindications = treatment_guidelines.get("contraindications", [])

        all_contraindications = stage_contraindications + guideline_contraindications

        # 检查患者是否有这些禁忌症
        for contraindication in all_contraindications:
            if self._check_contraindication(patient_state, contraindication):
                contraindications.append(contraindication)

        # 基于患者特征的动态禁忌症
        if treatment == TreatmentOption.SURGERY and patient_state.age > 85:
            contraindications.append("advanced_age_surgical_risk")

        if (
            treatment == TreatmentOption.CHEMOTHERAPY
            and "cardiac_dysfunction" in patient_state.comorbidities
        ):
            contraindications.append("cardiac_dysfunction_chemotherapy_risk")

        return contraindications

    def _check_contraindication(
        self, patient_state: PatientState, contraindication: str
    ) -> bool:
        """检查患者是否有特定禁忌症"""
        contraindication_checks = {
            "severe_cardiac_dysfunction": "cardiac_dysfunction"
            in patient_state.comorbidities,
            "multiple_organ_dysfunction": len(patient_state.comorbidities) > 3,
            "poor_performance_status": patient_state.quality_of_life_score < 0.3,
            "metastatic_disease": patient_state.stage == "IV",
            "pregnancy": False,  # 需要从其他字段获取
        }

        return contraindication_checks.get(contraindication, False)

    def _get_evidence_level(self, treatment: TreatmentOption = None) -> str:
        """获取证据等级"""
        if not treatment:
            return "N/A"

        # 从NCCN指南获取证据等级
        nccn_guidelines = self.knowledge_base.get("treatment_guidelines", {}).get(
            "NCCN", {}
        )
        breast_cancer_guidelines = nccn_guidelines.get("breast_cancer", {})
        treatment_info = breast_cancer_guidelines.get(treatment.value, {})

        return treatment_info.get("evidence_level", "Category 2B")

    def _get_success_rates(
        self, patient_state: PatientState, treatment: TreatmentOption = None
    ) -> Dict[str, float]:
        """获取成功率数据"""
        if not treatment:
            # 返回疾病阶段的一般生存率
            disease_profile = self.knowledge_base.get("disease_profiles", {}).get(
                patient_state.diagnosis, {}
            )
            stage_info = disease_profile.get(f"stage_{patient_state.stage.lower()}", {})
            return stage_info.get("survival_rates", {})

        # 基于治疗方案和患者特征调整成功率
        base_rates = self._get_base_success_rates(patient_state, treatment)

        # 根据患者特征调整
        adjusted_rates = self._adjust_success_rates_by_patient(
            base_rates, patient_state
        )

        return adjusted_rates

    def _get_base_success_rates(
        self, patient_state: PatientState, treatment: TreatmentOption
    ) -> Dict[str, float]:
        """获取基础成功率"""
        # 从疾病profile获取基础生存率
        disease_profile = self.knowledge_base.get("disease_profiles", {}).get(
            patient_state.diagnosis, {}
        )
        stage_info = disease_profile.get(f"stage_{patient_state.stage.lower()}", {})
        base_survival = stage_info.get("survival_rates", {"5_year": 0.7})

        # 根据治疗方案调整
        treatment_modifiers = {
            TreatmentOption.SURGERY: 1.1,
            TreatmentOption.CHEMOTHERAPY: 1.05,
            TreatmentOption.RADIOTHERAPY: 1.03,
            TreatmentOption.IMMUNOTHERAPY: 1.0,
            TreatmentOption.PALLIATIVE_CARE: 0.5,
            TreatmentOption.WATCHFUL_WAITING: 0.8,
        }

        modifier = treatment_modifiers.get(treatment, 1.0)

        adjusted_rates = {}
        for timepoint, rate in base_survival.items():
            adjusted_rates[timepoint] = min(1.0, rate * modifier)

        return adjusted_rates

    def _adjust_success_rates_by_patient(
        self, base_rates: Dict[str, float], patient_state: PatientState
    ) -> Dict[str, float]:
        """根据患者特征调整成功率"""
        adjusted_rates = base_rates.copy()

        # 年龄调整
        age_factor = 1.0
        if patient_state.age > 70:
            age_factor = 0.9
        elif patient_state.age > 80:
            age_factor = 0.8

        # 并发症调整
        comorbidity_factor = 1.0
        for comorbidity in patient_state.comorbidities:
            comorbidity_impact = self.knowledge_base.get(
                "comorbidity_considerations", {}
            ).get(comorbidity, {})
            if comorbidity_impact:
                # 取各治疗方案风险增加的平均值作为整体影响
                avg_risk_increase = np.mean(
                    [
                        info.get("risk_increase", 1.0)
                        for info in comorbidity_impact.values()
                    ]
                )
                comorbidity_factor *= 1.0 / avg_risk_increase

        # 生活质量调整
        qol_factor = 0.8 + patient_state.quality_of_life_score * 0.4  # 0.8-1.2的范围

        # 应用所有调整因子
        total_factor = age_factor * comorbidity_factor * qol_factor

        for timepoint, rate in adjusted_rates.items():
            adjusted_rates[timepoint] = max(0.1, min(1.0, rate * total_factor))

        return adjusted_rates

    def _get_side_effects(self, treatment: TreatmentOption = None) -> Dict[str, Any]:
        """获取副作用信息"""
        if not treatment:
            return {}

        side_effects_db = {
            TreatmentOption.SURGERY: {
                "severity": "moderate",
                "duration": "acute",
                "main_effects": ["pain", "infection_risk", "scarring"],
                "serious_complications": ["bleeding", "anesthesia_complications"],
                "frequency": {"common": 0.8, "serious": 0.05},
            },
            TreatmentOption.CHEMOTHERAPY: {
                "severity": "severe",
                "duration": "treatment_period_plus_recovery",
                "main_effects": ["nausea", "fatigue", "hair_loss", "immunosuppression"],
                "serious_complications": ["severe_neutropenia", "cardiac_toxicity"],
                "frequency": {"common": 0.9, "serious": 0.15},
            },
            TreatmentOption.RADIOTHERAPY: {
                "severity": "mild_to_moderate",
                "duration": "treatment_plus_2_weeks",
                "main_effects": ["skin_irritation", "fatigue", "local_swelling"],
                "serious_complications": ["radiation_pneumonitis", "cardiac_injury"],
                "frequency": {"common": 0.7, "serious": 0.02},
            },
            TreatmentOption.IMMUNOTHERAPY: {
                "severity": "variable",
                "duration": "variable",
                "main_effects": ["fatigue", "skin_reactions", "diarrhea"],
                "serious_complications": ["autoimmune_reactions", "severe_colitis"],
                "frequency": {"common": 0.6, "serious": 0.08},
            },
            TreatmentOption.PALLIATIVE_CARE: {
                "severity": "minimal",
                "duration": "ongoing",
                "main_effects": ["symptom_focused"],
                "serious_complications": [],
                "frequency": {"common": 0.1, "serious": 0.0},
            },
            TreatmentOption.WATCHFUL_WAITING: {
                "severity": "none",
                "duration": "none",
                "main_effects": ["anxiety", "uncertainty"],
                "serious_complications": ["disease_progression"],
                "frequency": {"common": 0.3, "serious": 0.0},
            },
        }

        return side_effects_db.get(treatment, {})

    def _get_comorbidity_considerations(
        self, patient_state: PatientState
    ) -> Dict[str, Any]:
        """获取并发症考虑"""
        considerations = {}

        for comorbidity in patient_state.comorbidities:
            comorbidity_info = self.knowledge_base.get(
                "comorbidity_considerations", {}
            ).get(comorbidity, {})
            if comorbidity_info:
                considerations[comorbidity] = comorbidity_info

        return considerations

    def _get_drug_interactions(
        self, patient_state: PatientState, treatment: TreatmentOption = None
    ) -> List[Dict[str, Any]]:
        """获取药物相互作用"""
        interactions = []

        if not treatment:
            return interactions

        # 检查治疗与并发症的相互作用
        for comorbidity in patient_state.comorbidities:
            interaction_key = f"{treatment.value}_{comorbidity}"
            interaction_info = self.knowledge_base.get("drug_interactions", {}).get(
                interaction_key
            )

            if interaction_info:
                interactions.append(interaction_info)

        return interactions

    def search_knowledge(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索知识库"""
        # 简化的文本搜索实现
        # 在实际项目中会使用向量搜索或更复杂的检索方法

        results = []
        query_lower = query.lower()

        # 搜索治疗指南
        for guideline_source, guidelines in self.knowledge_base.get(
            "treatment_guidelines", {}
        ).items():
            for disease, treatments in guidelines.items():
                for treatment, info in treatments.items():
                    if isinstance(info, dict):
                        recommendation = info.get("recommendations", "")
                        if query_lower in recommendation.lower():
                            results.append(
                                {
                                    "source": f"{guideline_source}_{disease}_{treatment}",
                                    "content": recommendation,
                                    "evidence_level": info.get("evidence_level", ""),
                                    "type": "guideline",
                                }
                            )

        # 搜索疾病资料
        for disease, profiles in self.knowledge_base.get(
            "disease_profiles", {}
        ).items():
            for stage, info in profiles.items():
                description = info.get("description", "")
                if query_lower in description.lower():
                    results.append(
                        {
                            "source": f"{disease}_{stage}",
                            "content": description,
                            "survival_rates": info.get("survival_rates", {}),
                            "type": "disease_profile",
                        }
                    )

        return results[:max_results]

    def update_knowledge_base(self, new_knowledge: Dict[str, Any]) -> None:
        """更新知识库"""
        # 深度合并新知识到现有知识库
        self._deep_merge(self.knowledge_base, new_knowledge)
        logger.info("Knowledge base updated")

    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """深度合并字典"""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
