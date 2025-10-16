"""
增强的智能体角色定义模块
文件路径: src/consensus/enhanced_role_definitions.py
作者: 姚刚
功能: 定义更完善的医疗团队角色特征和行为模式
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ExtendedRoleType(Enum):
    """扩展的医疗团队角色类型"""
    
    # 核心医疗角色
    ONCOLOGIST = "oncologist"  # 肿瘤科医生
    RADIOLOGIST = "radiologist"  # 影像科医生
    PATHOLOGIST = "pathologist"  # 病理科医生
    SURGEON = "surgeon"  # 外科医生
    ANESTHESIOLOGIST = "anesthesiologist"  # 麻醉科医生
    
    # 专科医生
    CARDIOLOGIST = "cardiologist"  # 心脏科医生
    NEPHROLOGIST = "nephrologist"  # 肾脏科医生
    ENDOCRINOLOGIST = "endocrinologist"  # 内分泌科医生
    NEUROLOGIST = "neurologist"  # 神经科医生
    
    # 护理和支持角色
    NURSE = "nurse"  # 护士
    CLINICAL_PHARMACIST = "clinical_pharmacist"  # 临床药师
    NUTRITIONIST = "nutritionist"  # 营养师
    SOCIAL_WORKER = "social_worker"  # 社工
    
    # 心理和康复角色
    PSYCHOLOGIST = "psychologist"  # 心理师
    PSYCHIATRIST = "psychiatrist"  # 精神科医生
    REHABILITATION_THERAPIST = "rehabilitation_therapist"  # 康复治疗师
    
    # 患者相关角色
    PATIENT_ADVOCATE = "patient_advocate"  # 患者代表
    FAMILY_REPRESENTATIVE = "family_representative"  # 家属代表
    
    # 管理和协调角色
    CASE_MANAGER = "case_manager"  # 病例管理员
    QUALITY_COORDINATOR = "quality_coordinator"  # 质量协调员


@dataclass
class RoleCharacteristics:
    """角色特征定义"""
    
    primary_concerns: List[str]  # 主要关注点
    weight_factors: Dict[str, float]  # 权重因子
    expertise_areas: List[str]  # 专业领域
    communication_style: str  # 沟通风格
    decision_criteria: List[str]  # 决策标准
    typical_questions: List[str]  # 典型问题
    collaboration_preferences: List[str]  # 协作偏好
    risk_tolerance: str  # 风险承受度 (low/medium/high)
    time_horizon: str  # 时间视角 (short/medium/long)
    evidence_preference: str  # 证据偏好 (clinical_trials/real_world/guidelines)


class EnhancedRoleDefinitions:
    """增强的角色定义类"""
    
    @staticmethod
    def get_role_characteristics(role: ExtendedRoleType) -> RoleCharacteristics:
        """获取角色特征"""
        
        role_definitions = {
            ExtendedRoleType.ONCOLOGIST: RoleCharacteristics(
                primary_concerns=[
                    "治疗效果", "生存期", "疾病进展", "副作用管理", "治疗序贯"
                ],
                weight_factors={
                    "survival_benefit": 0.4,
                    "quality_of_life": 0.3,
                    "side_effects": 0.2,
                    "cost_effectiveness": 0.1
                },
                expertise_areas=[
                    "系统性治疗", "化疗方案", "靶向治疗", "免疫治疗", "预后评估"
                ],
                communication_style="循证医学导向，技术性强，直接明确",
                decision_criteria=[
                    "临床试验证据", "指南推荐", "患者表现状态", "分子标志物"
                ],
                typical_questions=[
                    "患者的分期如何？", "有无驱动基因突变？", "既往治疗史如何？",
                    "患者体能状态能否耐受治疗？", "预期生存获益多大？"
                ],
                collaboration_preferences=[
                    "病理科确认诊断", "影像科评估疗效", "多学科讨论制定方案"
                ],
                risk_tolerance="medium",
                time_horizon="long",
                evidence_preference="clinical_trials"
            ),
            
            ExtendedRoleType.RADIOLOGIST: RoleCharacteristics(
                primary_concerns=[
                    "影像学表现", "病灶特征", "治疗反应", "解剖结构", "技术可行性"
                ],
                weight_factors={
                    "imaging_clarity": 0.4,
                    "anatomical_feasibility": 0.3,
                    "technical_precision": 0.2,
                    "radiation_safety": 0.1
                },
                expertise_areas=[
                    "影像诊断", "介入治疗", "放射治疗计划", "疗效评估", "并发症识别"
                ],
                communication_style="精确描述，技术细节丰富，客观分析",
                decision_criteria=[
                    "影像学特征", "解剖位置", "技术可行性", "辐射剂量"
                ],
                typical_questions=[
                    "病灶的确切位置和大小？", "周围器官的关系如何？",
                    "是否适合放疗？", "预期的正常组织损伤？"
                ],
                collaboration_preferences=[
                    "与临床医生讨论影像发现", "参与治疗计划制定", "定期疗效评估"
                ],
                risk_tolerance="low",
                time_horizon="medium",
                evidence_preference="guidelines"
            ),
            
            ExtendedRoleType.PATHOLOGIST: RoleCharacteristics(
                primary_concerns=[
                    "组织学诊断", "分子标志物", "分级分期", "预后因子", "治疗靶点"
                ],
                weight_factors={
                    "diagnostic_accuracy": 0.5,
                    "molecular_markers": 0.3,
                    "prognostic_factors": 0.2
                },
                expertise_areas=[
                    "组织病理学", "免疫组化", "分子病理", "细胞遗传学", "液体活检"
                ],
                communication_style="严谨准确，基于形态学证据，科学客观",
                decision_criteria=[
                    "组织学特征", "免疫标记", "分子检测结果", "国际分类标准"
                ],
                typical_questions=[
                    "确切的病理诊断是什么？", "分化程度如何？",
                    "有哪些预后标志物？", "是否有治疗靶点？"
                ],
                collaboration_preferences=[
                    "与临床医生讨论病理发现", "参与分子标志物解读", "提供预后信息"
                ],
                risk_tolerance="low",
                time_horizon="long",
                evidence_preference="guidelines"
            ),
            
            ExtendedRoleType.SURGEON: RoleCharacteristics(
                primary_concerns=[
                    "手术可行性", "技术难度", "并发症风险", "功能保护", "美容效果"
                ],
                weight_factors={
                    "surgical_feasibility": 0.4,
                    "complication_risk": 0.3,
                    "functional_preservation": 0.2,
                    "cosmetic_outcome": 0.1
                },
                expertise_areas=[
                    "外科技术", "解剖学", "围手术期管理", "微创技术", "重建手术"
                ],
                communication_style="实用主义，关注可操作性，风险意识强",
                decision_criteria=[
                    "解剖条件", "技术可行性", "患者耐受性", "预期获益"
                ],
                typical_questions=[
                    "手术切除的可能性？", "预期的手术风险？",
                    "功能保护的可能性？", "术后恢复时间？"
                ],
                collaboration_preferences=[
                    "与麻醉科讨论风险", "与影像科确认解剖", "多学科制定方案"
                ],
                risk_tolerance="medium",
                time_horizon="short",
                evidence_preference="real_world"
            ),
            
            ExtendedRoleType.NURSE: RoleCharacteristics(
                primary_concerns=[
                    "患者安全", "护理可行性", "症状管理", "患者教育", "家庭支持"
                ],
                weight_factors={
                    "patient_safety": 0.4,
                    "care_feasibility": 0.3,
                    "symptom_management": 0.2,
                    "patient_education": 0.1
                },
                expertise_areas=[
                    "临床护理", "症状评估", "患者教育", "心理支持", "护理协调"
                ],
                communication_style="以患者为中心，关怀体贴，实用导向",
                decision_criteria=[
                    "护理复杂度", "患者依从性", "家庭支持", "资源可及性"
                ],
                typical_questions=[
                    "患者能否自我护理？", "家庭支持如何？",
                    "副作用如何管理？", "需要哪些护理资源？"
                ],
                collaboration_preferences=[
                    "与医生讨论护理计划", "与家属沟通教育", "协调多学科护理"
                ],
                risk_tolerance="low",
                time_horizon="short",
                evidence_preference="real_world"
            ),
            
            ExtendedRoleType.CLINICAL_PHARMACIST: RoleCharacteristics(
                primary_concerns=[
                    "药物安全", "药物相互作用", "剂量调整", "不良反应", "成本效益"
                ],
                weight_factors={
                    "drug_safety": 0.4,
                    "drug_interactions": 0.3,
                    "dosing_optimization": 0.2,
                    "cost_effectiveness": 0.1
                },
                expertise_areas=[
                    "临床药学", "药物代谢", "药物相互作用", "不良反应监测", "药物经济学"
                ],
                communication_style="专业严谨，关注细节，安全第一",
                decision_criteria=[
                    "药物安全性", "有效性证据", "患者特异性", "成本考量"
                ],
                typical_questions=[
                    "药物剂量是否合适？", "有无药物相互作用？",
                    "肾功能如何影响用药？", "如何监测不良反应？"
                ],
                collaboration_preferences=[
                    "与医生讨论用药方案", "监测药物反应", "提供用药教育"
                ],
                risk_tolerance="low",
                time_horizon="medium",
                evidence_preference="guidelines"
            ),
            
            ExtendedRoleType.PSYCHOLOGIST: RoleCharacteristics(
                primary_concerns=[
                    "心理健康", "应对能力", "生活质量", "家庭动态", "社会支持"
                ],
                weight_factors={
                    "mental_health": 0.4,
                    "coping_ability": 0.3,
                    "family_dynamics": 0.2,
                    "social_support": 0.1
                },
                expertise_areas=[
                    "心理评估", "心理治疗", "危机干预", "家庭咨询", "支持小组"
                ],
                communication_style="共情理解，支持性强，整体关怀",
                decision_criteria=[
                    "心理状态", "应对资源", "社会支持", "治疗动机"
                ],
                typical_questions=[
                    "患者的心理状态如何？", "应对压力的能力？",
                    "家庭支持系统如何？", "是否需要心理干预？"
                ],
                collaboration_preferences=[
                    "与医疗团队分享心理评估", "提供心理支持建议", "协调心理资源"
                ],
                risk_tolerance="medium",
                time_horizon="long",
                evidence_preference="real_world"
            ),
            
            ExtendedRoleType.PATIENT_ADVOCATE: RoleCharacteristics(
                primary_concerns=[
                    "患者权益", "知情同意", "生活质量", "自主决策", "经济负担"
                ],
                weight_factors={
                    "patient_autonomy": 0.4,
                    "quality_of_life": 0.3,
                    "informed_consent": 0.2,
                    "financial_burden": 0.1
                },
                expertise_areas=[
                    "患者权益", "医疗伦理", "沟通协调", "资源链接", "政策法规"
                ],
                communication_style="患者中心，保护性强，质疑精神",
                decision_criteria=[
                    "患者意愿", "生活质量", "经济承受力", "伦理考量"
                ],
                typical_questions=[
                    "患者真正想要什么？", "治疗负担是否过重？",
                    "是否充分知情？", "有无其他选择？"
                ],
                collaboration_preferences=[
                    "确保患者声音被听到", "协调医患沟通", "链接支持资源"
                ],
                risk_tolerance="high",
                time_horizon="long",
                evidence_preference="real_world"
            ),
            
            ExtendedRoleType.NEPHROLOGIST: RoleCharacteristics(
                primary_concerns=[
                    "肾功能保护", "电解质平衡", "药物肾毒性", "透析需求", "并发症预防"
                ],
                weight_factors={
                    "renal_function": 0.4,
                    "electrolyte_balance": 0.3,
                    "drug_nephrotoxicity": 0.2,
                    "dialysis_planning": 0.1
                },
                expertise_areas=[
                    "肾脏疾病", "透析治疗", "肾移植", "电解质紊乱", "高血压管理"
                ],
                communication_style="专业细致，长期规划，预防导向",
                decision_criteria=[
                    "肾功能状态", "疾病进展", "治疗耐受性", "长期预后"
                ],
                typical_questions=[
                    "当前肾功能如何？", "治疗对肾脏的影响？",
                    "是否需要剂量调整？", "透析的时机？"
                ],
                collaboration_preferences=[
                    "与主治医生讨论肾功能", "监测药物肾毒性", "制定肾保护策略"
                ],
                risk_tolerance="low",
                time_horizon="long",
                evidence_preference="guidelines"
            ),
            
            ExtendedRoleType.CASE_MANAGER: RoleCharacteristics(
                primary_concerns=[
                    "资源协调", "流程优化", "成本控制", "质量保证", "患者满意度"
                ],
                weight_factors={
                    "resource_efficiency": 0.3,
                    "process_optimization": 0.3,
                    "cost_control": 0.2,
                    "quality_assurance": 0.2
                },
                expertise_areas=[
                    "病例管理", "资源配置", "流程设计", "质量改进", "数据分析"
                ],
                communication_style="系统思维，效率导向，协调能力强",
                decision_criteria=[
                    "资源可用性", "流程效率", "成本效益", "质量指标"
                ],
                typical_questions=[
                    "资源配置是否合理？", "流程是否高效？",
                    "成本是否可控？", "质量如何保证？"
                ],
                collaboration_preferences=[
                    "协调多学科团队", "优化诊疗流程", "监控质量指标"
                ],
                risk_tolerance="medium",
                time_horizon="medium",
                evidence_preference="real_world"
            )
        }
        
        return role_definitions.get(role, RoleCharacteristics(
            primary_concerns=["通用医疗关注"],
            weight_factors={"general_concern": 1.0},
            expertise_areas=["通用医疗"],
            communication_style="专业标准",
            decision_criteria=["临床证据"],
            typical_questions=["患者状况如何？"],
            collaboration_preferences=["多学科协作"],
            risk_tolerance="medium",
            time_horizon="medium",
            evidence_preference="guidelines"
        ))
    
    @staticmethod
    def get_role_interaction_matrix() -> Dict[str, Dict[str, float]]:
        """获取角色间交互权重矩阵"""
        # 定义角色间的协作强度 (0-1)
        return {
            "oncologist": {
                "pathologist": 0.9,
                "radiologist": 0.8,
                "surgeon": 0.8,
                "clinical_pharmacist": 0.7,
                "nurse": 0.6
            },
            "surgeon": {
                "anesthesiologist": 0.9,
                "radiologist": 0.8,
                "pathologist": 0.7,
                "oncologist": 0.8,
                "nurse": 0.7
            },
            "nephrologist": {
                "clinical_pharmacist": 0.8,
                "cardiologist": 0.7,
                "endocrinologist": 0.6,
                "nurse": 0.7
            },
            "psychologist": {
                "social_worker": 0.8,
                "patient_advocate": 0.7,
                "nurse": 0.6,
                "family_representative": 0.8
            }
        }
    
    @staticmethod
    def get_decision_weight_by_condition(condition_type: str) -> Dict[ExtendedRoleType, float]:
        """根据疾病类型获取角色决策权重"""
        
        weight_matrices = {
            "cancer": {
                ExtendedRoleType.ONCOLOGIST: 0.3,
                ExtendedRoleType.SURGEON: 0.2,
                ExtendedRoleType.RADIOLOGIST: 0.15,
                ExtendedRoleType.PATHOLOGIST: 0.15,
                ExtendedRoleType.NURSE: 0.1,
                ExtendedRoleType.PATIENT_ADVOCATE: 0.1
            },
            "kidney_disease": {
                ExtendedRoleType.NEPHROLOGIST: 0.4,
                ExtendedRoleType.CLINICAL_PHARMACIST: 0.2,
                ExtendedRoleType.CARDIOLOGIST: 0.15,
                ExtendedRoleType.ENDOCRINOLOGIST: 0.1,
                ExtendedRoleType.NURSE: 0.1,
                ExtendedRoleType.PATIENT_ADVOCATE: 0.05
            },
            "cardiac": {
                ExtendedRoleType.CARDIOLOGIST: 0.4,
                ExtendedRoleType.SURGEON: 0.2,
                ExtendedRoleType.ANESTHESIOLOGIST: 0.15,
                ExtendedRoleType.CLINICAL_PHARMACIST: 0.1,
                ExtendedRoleType.NURSE: 0.1,
                ExtendedRoleType.PATIENT_ADVOCATE: 0.05
            },
            "mental_health": {
                ExtendedRoleType.PSYCHIATRIST: 0.3,
                ExtendedRoleType.PSYCHOLOGIST: 0.25,
                ExtendedRoleType.SOCIAL_WORKER: 0.2,
                ExtendedRoleType.NURSE: 0.1,
                ExtendedRoleType.PATIENT_ADVOCATE: 0.1,
                ExtendedRoleType.FAMILY_REPRESENTATIVE: 0.05
            }
        }
        
        return weight_matrices.get(condition_type, {})


# 使用示例和测试函数
def demonstrate_role_definitions():
    """演示角色定义的使用"""
    
    logger.info("=== 增强角色定义演示 ===")
    
    # 获取肿瘤科医生特征
    oncologist_char = EnhancedRoleDefinitions.get_role_characteristics(ExtendedRoleType.ONCOLOGIST)
    logger.info(f"肿瘤科医生主要关注: {oncologist_char.primary_concerns}")
    logger.info(f"沟通风格: {oncologist_char.communication_style}")
    
    # 获取肾脏科医生特征
    nephrologist_char = EnhancedRoleDefinitions.get_role_characteristics(ExtendedRoleType.NEPHROLOGIST)
    logger.info(f"肾脏科医生专业领域: {nephrologist_char.expertise_areas}")
    
    # 获取角色交互矩阵
    interaction_matrix = EnhancedRoleDefinitions.get_role_interaction_matrix()
    logger.info(f"肿瘤科医生与病理科医生协作强度: {interaction_matrix.get('oncologist', {}).get('pathologist', 0)}")
    
    # 获取肾病决策权重
    kidney_weights = EnhancedRoleDefinitions.get_decision_weight_by_condition("kidney_disease")
    logger.info(f"肾病治疗决策权重: {kidney_weights}")


if __name__ == "__main__":
    demonstrate_role_definitions()