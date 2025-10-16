"""
智能体角色工厂模块
文件路径: src/consensus/role_factory.py
作者: 姚刚
功能: 根据患者病情动态选择和配置合适的角色组合
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import logging
import re

from .enhanced_role_definitions import ExtendedRoleType, EnhancedRoleDefinitions, RoleCharacteristics
from ..core.data_models import PatientState, RoleType

logger = logging.getLogger(__name__)


@dataclass
class RoleSelectionCriteria:
    """角色选择标准"""
    
    required_roles: Set[ExtendedRoleType]  # 必需角色
    optional_roles: Set[ExtendedRoleType]  # 可选角色
    max_team_size: int  # 最大团队规模
    priority_weights: Dict[ExtendedRoleType, float]  # 优先级权重
    condition_relevance: Dict[str, float]  # 病情相关性


class IntelligentRoleFactory:
    """智能角色工厂"""
    
    def __init__(self):
        """初始化角色工厂"""
        self.role_definitions = EnhancedRoleDefinitions()
        self.condition_patterns = self._initialize_condition_patterns()
        self.role_compatibility_matrix = self._initialize_compatibility_matrix()
        
    def _initialize_condition_patterns(self) -> Dict[str, List[str]]:
        """初始化疾病模式匹配"""
        return {
            "cancer": [
                "cancer", "tumor", "malignancy", "carcinoma", "sarcoma", 
                "lymphoma", "leukemia", "oncology", "neoplasm"
            ],
            "kidney_disease": [
                "kidney", "renal", "nephro", "dialysis", "creatinine",
                "glomerular", "proteinuria", "uremia"
            ],
            "cardiac": [
                "heart", "cardiac", "cardio", "coronary", "myocardial",
                "arrhythmia", "hypertension", "valve"
            ],
            "diabetes": [
                "diabetes", "diabetic", "glucose", "insulin", "glycemic",
                "hyperglycemia", "hypoglycemia"
            ],
            "mental_health": [
                "depression", "anxiety", "psychiatric", "psychological",
                "bipolar", "schizophrenia", "ptsd"
            ],
            "respiratory": [
                "lung", "pulmonary", "respiratory", "pneumonia", "asthma",
                "copd", "bronchial"
            ],
            "neurological": [
                "brain", "neurological", "stroke", "seizure", "dementia",
                "parkinson", "alzheimer", "epilepsy"
            ],
            "surgical": [
                "surgery", "surgical", "operation", "procedure", "resection",
                "reconstruction", "transplant"
            ]
        }
    
    def _initialize_compatibility_matrix(self) -> Dict[ExtendedRoleType, Set[ExtendedRoleType]]:
        """初始化角色兼容性矩阵"""
        return {
            ExtendedRoleType.ONCOLOGIST: {
                ExtendedRoleType.PATHOLOGIST, ExtendedRoleType.RADIOLOGIST,
                ExtendedRoleType.SURGEON, ExtendedRoleType.CLINICAL_PHARMACIST
            },
            ExtendedRoleType.SURGEON: {
                ExtendedRoleType.ANESTHESIOLOGIST, ExtendedRoleType.RADIOLOGIST,
                ExtendedRoleType.PATHOLOGIST, ExtendedRoleType.NURSE
            },
            ExtendedRoleType.NEPHROLOGIST: {
                ExtendedRoleType.CARDIOLOGIST, ExtendedRoleType.ENDOCRINOLOGIST,
                ExtendedRoleType.CLINICAL_PHARMACIST, ExtendedRoleType.NUTRITIONIST
            },
            ExtendedRoleType.CARDIOLOGIST: {
                ExtendedRoleType.NEPHROLOGIST, ExtendedRoleType.ENDOCRINOLOGIST,
                ExtendedRoleType.ANESTHESIOLOGIST, ExtendedRoleType.CLINICAL_PHARMACIST
            },
            ExtendedRoleType.PSYCHOLOGIST: {
                ExtendedRoleType.PSYCHIATRIST, ExtendedRoleType.SOCIAL_WORKER,
                ExtendedRoleType.PATIENT_ADVOCATE, ExtendedRoleType.FAMILY_REPRESENTATIVE
            }
        }
    
    def analyze_patient_conditions(self, patient_state: PatientState) -> Dict[str, float]:
        """分析患者病情，返回各疾病类型的相关性得分"""
        
        condition_scores = {}
        
        # 分析主要诊断
        primary_diagnosis = patient_state.diagnosis.lower()
        
        # 分析合并症
        comorbidities_text = " ".join(patient_state.comorbidities).lower()
        
        # 分析症状
        symptoms_text = " ".join(patient_state.symptoms).lower()
        
        # 合并所有文本
        all_text = f"{primary_diagnosis} {comorbidities_text} {symptoms_text}"
        
        # 计算各疾病类型的匹配得分
        for condition_type, patterns in self.condition_patterns.items():
            score = 0.0
            for pattern in patterns:
                # 主要诊断匹配权重更高
                if pattern in primary_diagnosis:
                    score += 3.0
                # 合并症匹配
                elif pattern in comorbidities_text:
                    score += 2.0
                # 症状匹配
                elif pattern in symptoms_text:
                    score += 1.0
            
            # 标准化得分
            condition_scores[condition_type] = min(score / len(patterns), 1.0)
        
        logger.info(f"患者病情分析结果: {condition_scores}")
        return condition_scores
    
    def select_optimal_team(
        self, 
        patient_state: PatientState, 
        max_team_size: int = 6,
        include_core_roles: bool = True
    ) -> Tuple[Set[ExtendedRoleType], Dict[str, float]]:
        """选择最优团队组合"""
        
        # 分析患者病情
        condition_scores = self.analyze_patient_conditions(patient_state)
        
        # 核心角色（总是包含）
        core_roles = {
            ExtendedRoleType.NURSE,
            ExtendedRoleType.PATIENT_ADVOCATE
        } if include_core_roles else set()
        
        # 根据病情选择专科角色
        specialist_roles = set()
        
        # 肿瘤相关
        if condition_scores.get("cancer", 0) > 0.3:
            specialist_roles.update({
                ExtendedRoleType.ONCOLOGIST,
                ExtendedRoleType.PATHOLOGIST,
                ExtendedRoleType.RADIOLOGIST
            })
        
        # 肾病相关
        if condition_scores.get("kidney_disease", 0) > 0.3:
            specialist_roles.add(ExtendedRoleType.NEPHROLOGIST)
        
        # 心脏病相关
        if condition_scores.get("cardiac", 0) > 0.3:
            specialist_roles.add(ExtendedRoleType.CARDIOLOGIST)
        
        # 糖尿病相关
        if condition_scores.get("diabetes", 0) > 0.3:
            specialist_roles.add(ExtendedRoleType.ENDOCRINOLOGIST)
        
        # 心理健康相关
        if (condition_scores.get("mental_health", 0) > 0.2 or 
            patient_state.psychological_status in ["anxious", "depressed", "distressed"]):
            specialist_roles.add(ExtendedRoleType.PSYCHOLOGIST)
        
        # 手术相关
        if condition_scores.get("surgical", 0) > 0.3:
            specialist_roles.update({
                ExtendedRoleType.SURGEON,
                ExtendedRoleType.ANESTHESIOLOGIST
            })
        
        # 药物管理（复杂病例总是需要）
        if len(patient_state.comorbidities) > 2:
            specialist_roles.add(ExtendedRoleType.CLINICAL_PHARMACIST)
        
        # 合并核心角色和专科角色
        selected_roles = core_roles.union(specialist_roles)
        
        # 如果团队过大，进行优化
        if len(selected_roles) > max_team_size:
            selected_roles = self._optimize_team_size(
                selected_roles, condition_scores, max_team_size
            )
        
        logger.info(f"选择的团队角色: {[role.value for role in selected_roles]}")
        return selected_roles, condition_scores
    
    def _optimize_team_size(
        self, 
        roles: Set[ExtendedRoleType], 
        condition_scores: Dict[str, float],
        max_size: int
    ) -> Set[ExtendedRoleType]:
        """优化团队规模"""
        
        # 计算每个角色的重要性得分
        role_importance = {}
        
        for role in roles:
            importance = 0.0
            
            # 基于病情相关性计算重要性
            if role == ExtendedRoleType.ONCOLOGIST:
                importance += condition_scores.get("cancer", 0) * 3.0
            elif role == ExtendedRoleType.NEPHROLOGIST:
                importance += condition_scores.get("kidney_disease", 0) * 3.0
            elif role == ExtendedRoleType.CARDIOLOGIST:
                importance += condition_scores.get("cardiac", 0) * 3.0
            elif role == ExtendedRoleType.SURGEON:
                importance += condition_scores.get("surgical", 0) * 3.0
            elif role == ExtendedRoleType.PSYCHOLOGIST:
                importance += condition_scores.get("mental_health", 0) * 2.0
            elif role in [ExtendedRoleType.NURSE, ExtendedRoleType.PATIENT_ADVOCATE]:
                importance += 2.0  # 核心角色基础分
            else:
                importance += 1.0  # 其他角色基础分
            
            role_importance[role] = importance
        
        # 按重要性排序并选择前max_size个
        sorted_roles = sorted(role_importance.items(), key=lambda x: x[1], reverse=True)
        optimized_roles = {role for role, _ in sorted_roles[:max_size]}
        
        logger.info(f"团队优化: 从{len(roles)}个角色优化到{len(optimized_roles)}个")
        return optimized_roles
    
    def create_role_selection_criteria(
        self, 
        patient_state: PatientState
    ) -> RoleSelectionCriteria:
        """创建角色选择标准"""
        
        selected_roles, condition_scores = self.select_optimal_team(patient_state)
        
        # 区分必需和可选角色
        required_roles = {
            ExtendedRoleType.NURSE,
            ExtendedRoleType.PATIENT_ADVOCATE
        }
        
        # 根据病情严重程度确定必需的专科角色
        for condition, score in condition_scores.items():
            if score > 0.5:  # 高相关性
                if condition == "cancer":
                    required_roles.add(ExtendedRoleType.ONCOLOGIST)
                elif condition == "kidney_disease":
                    required_roles.add(ExtendedRoleType.NEPHROLOGIST)
                elif condition == "cardiac":
                    required_roles.add(ExtendedRoleType.CARDIOLOGIST)
        
        optional_roles = selected_roles - required_roles
        
        # 计算优先级权重
        priority_weights = {}
        for role in selected_roles:
            if role in required_roles:
                priority_weights[role] = 1.0
            else:
                priority_weights[role] = 0.7
        
        return RoleSelectionCriteria(
            required_roles=required_roles,
            optional_roles=optional_roles,
            max_team_size=6,
            priority_weights=priority_weights,
            condition_relevance=condition_scores
        )
    
    def get_role_interaction_recommendations(
        self, 
        selected_roles: Set[ExtendedRoleType]
    ) -> Dict[str, List[str]]:
        """获取角色交互建议"""
        
        recommendations = {
            "high_priority_interactions": [],
            "collaboration_suggestions": [],
            "communication_protocols": []
        }
        
        # 高优先级交互
        for role in selected_roles:
            compatible_roles = self.role_compatibility_matrix.get(role, set())
            for compatible_role in compatible_roles:
                if compatible_role in selected_roles:
                    interaction = f"{role.value} <-> {compatible_role.value}"
                    if interaction not in recommendations["high_priority_interactions"]:
                        recommendations["high_priority_interactions"].append(interaction)
        
        # 协作建议
        if ExtendedRoleType.ONCOLOGIST in selected_roles and ExtendedRoleType.PATHOLOGIST in selected_roles:
            recommendations["collaboration_suggestions"].append(
                "肿瘤科医生与病理科医生应就分子标志物和治疗靶点进行深入讨论"
            )
        
        if ExtendedRoleType.NEPHROLOGIST in selected_roles and ExtendedRoleType.CLINICAL_PHARMACIST in selected_roles:
            recommendations["collaboration_suggestions"].append(
                "肾脏科医生与临床药师应协作制定肾功能相关的用药方案"
            )
        
        # 沟通协议
        recommendations["communication_protocols"].append(
            "每位专家应明确表达其专业观点和关注点"
        )
        recommendations["communication_protocols"].append(
            "护士应从实用性角度评估治疗方案的可行性"
        )
        recommendations["communication_protocols"].append(
            "患者代表应确保患者意愿和偏好得到充分考虑"
        )
        
        return recommendations
    
    def convert_to_legacy_roles(
        self, 
        extended_roles: Set[ExtendedRoleType]
    ) -> Set[RoleType]:
        """转换为原有的角色类型（向后兼容）"""
        
        conversion_map = {
            ExtendedRoleType.ONCOLOGIST: RoleType.ONCOLOGIST,
            ExtendedRoleType.RADIOLOGIST: RoleType.RADIOLOGIST,
            ExtendedRoleType.NURSE: RoleType.NURSE,
            ExtendedRoleType.PSYCHOLOGIST: RoleType.PSYCHOLOGIST,
            ExtendedRoleType.PATIENT_ADVOCATE: RoleType.PATIENT_ADVOCATE
        }
        
        legacy_roles = set()
        for extended_role in extended_roles:
            legacy_role = conversion_map.get(extended_role)
            if legacy_role:
                legacy_roles.add(legacy_role)
        
        # 确保至少有基本角色
        if not legacy_roles:
            legacy_roles = {
                RoleType.ONCOLOGIST,
                RoleType.NURSE,
                RoleType.PATIENT_ADVOCATE
            }
        
        return legacy_roles


# 使用示例和测试函数
def demonstrate_role_factory():
    """演示角色工厂的使用"""
    
    logger.info("=== 智能角色工厂演示 ===")
    
    # 创建角色工厂
    factory = IntelligentRoleFactory()
    
    # 模拟患者状态（肾病患者）
    from ..core.data_models import PatientState
    from datetime import datetime
    
    patient_state = PatientState(
        patient_id="demo_patient",
        age=65,
        diagnosis="chronic_kidney_disease",
        stage="stage_4",
        lab_results={"creatinine": 3.2, "gfr": 25},
        vital_signs={"bp": "150/90", "hr": 80},
        symptoms=["fatigue", "swelling", "shortness_of_breath"],
        comorbidities=["diabetes", "hypertension", "anemia"],
        psychological_status="anxious",
        quality_of_life_score=0.4,
        timestamp=datetime.now()
    )
    
    # 分析病情
    condition_scores = factory.analyze_patient_conditions(patient_state)
    logger.info(f"病情分析: {condition_scores}")
    
    # 选择团队
    selected_roles, _ = factory.select_optimal_team(patient_state)
    logger.info(f"选择的角色: {[role.value for role in selected_roles]}")
    
    # 获取交互建议
    recommendations = factory.get_role_interaction_recommendations(selected_roles)
    logger.info(f"交互建议: {recommendations}")
    
    # 转换为原有角色类型
    legacy_roles = factory.convert_to_legacy_roles(selected_roles)
    logger.info(f"兼容角色: {[role.value for role in legacy_roles]}")


if __name__ == "__main__":
    demonstrate_role_factory()