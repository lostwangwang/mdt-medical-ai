#!/usr/bin/env python3
"""
治疗方案生成演示脚本
文件路径: scripts/generate_treatment_plan.py
作者: AI Assistant
功能: 为患者10037928生成治疗方案的演示脚本
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.data_models import TreatmentOption, RoleType, PatientState
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplifiedTreatmentGenerator:
    """简化的治疗方案生成器"""
    
    def __init__(self):
        """初始化生成器"""
        self.rag_system = MedicalKnowledgeRAG()
        self.dialogue_manager = MultiAgentDialogueManager(self.rag_system)
        logger.info("SimplifiedTreatmentGenerator initialized")
    
    def load_patient_data(self, patient_id: str) -> PatientState:
        """加载患者数据"""
        try:
            # 从JSON文件加载患者数据
            json_file_path = project_root / "data" / "clinical_memory" / f"{patient_id}_clinical_memory.json"
            
            if not json_file_path.exists():
                raise FileNotFoundError(f"Patient data file not found: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
            
            # 解析患者数据
            patient_state = self._parse_patient_data(patient_data)
            logger.info(f"Successfully loaded patient {patient_id}")
            return patient_state
            
        except Exception as e:
            logger.error(f"Failed to load patient {patient_id}: {e}")
            raise
    
    def _parse_patient_data(self, patient_data: dict) -> PatientState:
        """解析患者数据"""
        try:
            # 提取基本信息
            subject_id = patient_data.get("subject_id", "unknown")
            gender = patient_data.get("gender", "unknown")
            age = patient_data.get("anchor_age", patient_data.get("age", 0))
            
            # 提取慢性疾病
            chronic_diseases_raw = patient_data.get("chronic_diseases", [])
            chronic_diseases = []
            for disease in chronic_diseases_raw:
                if isinstance(disease, dict) and "desc" in disease:
                    chronic_diseases.append(disease["desc"])
                elif isinstance(disease, str):
                    chronic_diseases.append(disease)
            
            # 提取实验室结果
            baseline_labs = patient_data.get("baseline_labs", {})
            lab_results = {}
            for lab_name, lab_value in baseline_labs.items():
                if isinstance(lab_value, (int, float)):
                    lab_results[lab_name.lower().replace(" ", "_")] = lab_value
            
            # 提取生命体征
            baseline_vitals = patient_data.get("baseline_vitals", {})
            vital_signs = {}
            for vital_name, vital_value in baseline_vitals.items():
                if isinstance(vital_value, (int, float)):
                    vital_signs[vital_name.lower().replace(" ", "_")] = vital_value
            
            # 提取出院药物
            discharge_medications = patient_data.get("discharge_medications", [])
            medications = []
            for med in discharge_medications:
                if isinstance(med, dict) and "medication" in med:
                    medications.append(med["medication"])
                elif isinstance(med, str):
                    medications.append(med)
            
            # 从慢性疾病中提取症状
            symptoms = []
            psychological_status = "stable"
            
            # 根据疾病推断症状
            for disease in chronic_diseases:
                if isinstance(disease, str):
                    disease_lower = disease.lower()
                    if "anxiety" in disease_lower:
                        symptoms.append("anxiety")
                        psychological_status = "anxious"
                    elif "depression" in disease_lower:
                        symptoms.append("depression")
                        psychological_status = "depressed"
                    elif "pain" in disease_lower:
                        symptoms.append("pain")
                    elif "anemia" in disease_lower:
                        symptoms.append("fatigue")
            
            # 根据慢性疾病推断主要诊断
            primary_diagnosis = "multiple_chronic_conditions"
            chronic_diseases_text = " ".join(chronic_diseases).lower()
            if "kidney" in chronic_diseases_text:
                primary_diagnosis = "chronic_kidney_disease"
            elif "diabetes" in chronic_diseases_text:
                primary_diagnosis = "diabetes_mellitus"
            
            # 创建PatientState对象
            patient_state = PatientState(
                patient_id=str(subject_id),
                age=age,
                diagnosis=primary_diagnosis,
                stage="unknown",  # 数据中没有分期信息
                lab_results=lab_results,
                vital_signs=vital_signs,
                symptoms=symptoms,
                comorbidities=chronic_diseases,
                psychological_status=psychological_status,
                quality_of_life_score=0.6,  # 基于多种慢性疾病的估计
                timestamp=datetime.now()
            )
            
            return patient_state
            
        except Exception as e:
            logger.error(f"Failed to parse patient data: {e}")
            raise
    
    def generate_treatment_plan(self, patient_id: str) -> dict:
        """生成治疗方案"""
        try:
            logger.info(f"Starting treatment plan generation for patient {patient_id}")
            
            # 1. 加载患者数据
            patient_state = self.load_patient_data(patient_id)
            
            print(f"\n=== 患者信息 ===")
            print(f"患者ID: {patient_state.patient_id}")
            print(f"年龄: {patient_state.age}岁")
            print(f"主要诊断: {patient_state.diagnosis}")
            print(f"慢性疾病: {', '.join(patient_state.comorbidities[:5])}...")
            print(f"当前症状: {', '.join(patient_state.symptoms)}")
            print(f"心理状态: {patient_state.psychological_status}")
            
            # 2. 进行MDT多智能体讨论
            print(f"\n=== 开始MDT多智能体讨论 ===")
            mdt_result = self.dialogue_manager.conduct_mdt_discussion(patient_state)
            
            # 3. 分析讨论结果
            print(f"\n=== MDT讨论结果 ===")
            print(f"是否达成共识: {mdt_result.convergence_achieved}")
            print(f"讨论轮数: {mdt_result.total_rounds}")
            
            # 4. 获取各专家意见
            expert_opinions = []
            for role, opinion in mdt_result.role_opinions.items():
                expert_opinion = {
                    "role": role.value,
                    "confidence": opinion.confidence,
                    "reasoning": opinion.reasoning[:200] + "..." if len(opinion.reasoning) > 200 else opinion.reasoning
                }
                expert_opinions.append(expert_opinion)
                
                print(f"\n{role.value}:")
                print(f"  置信度: {opinion.confidence:.2f}")
                print(f"  理由: {expert_opinion['reasoning']}")
            
            # 5. 确定最终推荐
            # 获取推荐得分最高的治疗方案
            treatment_scores = {}
            for role, opinion in mdt_result.role_opinions.items():
                for treatment, score in opinion.treatment_preferences.items():
                    if treatment not in treatment_scores:
                        treatment_scores[treatment] = []
                    treatment_scores[treatment].append(score)
            
            # 计算平均得分
            avg_scores = {}
            for treatment, scores in treatment_scores.items():
                avg_scores[treatment] = sum(scores) / len(scores)
            
            # 获取最高得分的治疗方案
            recommended_treatment = max(avg_scores.items(), key=lambda x: x[1])
            
            print(f"\n=== 最终推荐 ===")
            print(f"推荐治疗: {recommended_treatment[0].value}")
            print(f"平均得分: {recommended_treatment[1]:.3f}")
            
            # 6. 生成详细治疗计划
            detailed_plan = self._generate_detailed_plan(patient_state, recommended_treatment[0])
            
            # 7. 组装完整结果
            treatment_plan = {
                "patient_info": {
                    "patient_id": patient_state.patient_id,
                    "age": patient_state.age,
                    "primary_diagnosis": patient_state.diagnosis,
                    "comorbidities": patient_state.comorbidities,
                    "symptoms": patient_state.symptoms,
                    "psychological_status": patient_state.psychological_status
                },
                "mdt_discussion": {
                    "consensus_achieved": mdt_result.convergence_achieved,
                    "total_rounds": mdt_result.total_rounds,
                    "expert_opinions": expert_opinions
                },
                "recommended_treatment": {
                    "primary_recommendation": recommended_treatment[0].value,
                    "confidence_score": recommended_treatment[1],
                    "alternative_options": [
                        treatment.value for treatment, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[1:4]
                    ]
                },
                "detailed_plan": detailed_plan,
                "generated_at": datetime.now().isoformat()
            }
            
            return treatment_plan
            
        except Exception as e:
            logger.error(f"Failed to generate treatment plan: {e}")
            raise
    
    def _generate_detailed_plan(self, patient_state: PatientState, recommended_treatment: TreatmentOption) -> dict:
        """生成详细治疗计划"""
        
        # 基于患者状态和推荐治疗生成详细计划
        plan = {
            "medication_management": [],
            "lifestyle_interventions": [],
            "monitoring_schedule": [],
            "follow_up_plan": []
        }
        
        # 药物管理
        if "kidney" in patient_state.diagnosis.lower():
            plan["medication_management"].extend([
                "ACE抑制剂或ARB类药物控制血压",
                "磷结合剂控制血磷",
                "EPO治疗贫血",
                "维生素D补充"
            ])
        
        if any("diabetes" in disease.lower() for disease in patient_state.comorbidities):
            plan["medication_management"].extend([
                "胰岛素或口服降糖药",
                "血糖监测设备",
                "低血糖急救药物"
            ])
        
        # 生活方式干预
        plan["lifestyle_interventions"].extend([
            "低盐低磷饮食",
            "适量蛋白质摄入",
            "规律轻度运动",
            "戒烟限酒",
            "心理健康支持"
        ])
        
        # 监测计划
        plan["monitoring_schedule"].extend([
            "每周血压监测",
            "每月肾功能检查",
            "每3个月糖化血红蛋白检测",
            "定期电解质平衡检查"
        ])
        
        # 随访计划
        plan["follow_up_plan"].extend([
            "1周后门诊随访",
            "1个月后实验室检查",
            "3个月后全面评估",
            "紧急情况随时就诊"
        ])
        
        return plan
    
    def save_treatment_plan(self, treatment_plan: dict, output_path: str):
        """保存治疗方案"""
        try:
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(treatment_plan, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Treatment plan saved to {output_path}")
            print(f"\n治疗方案已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save treatment plan: {e}")
            raise


def main():
    """主函数"""
    try:
        print("=== MDT医疗智能体治疗方案生成系统 ===")
        print("正在为患者10037928生成个性化治疗方案...\n")
        
        # 创建治疗方案生成器
        generator = SimplifiedTreatmentGenerator()
        
        # 生成治疗方案
        patient_id = "10037928"
        treatment_plan = generator.generate_treatment_plan(patient_id)
        
        # 保存结果
        output_path = project_root / "results" / f"treatment_plan_{patient_id}.json"
        generator.save_treatment_plan(treatment_plan, str(output_path))
        
        print(f"\n=== 治疗方案生成完成 ===")
        print(f"患者ID: {patient_id}")
        print(f"推荐治疗: {treatment_plan['recommended_treatment']['primary_recommendation']}")
        print(f"置信度: {treatment_plan['recommended_treatment']['confidence_score']:.3f}")
        print(f"MDT讨论轮数: {treatment_plan['mdt_discussion']['total_rounds']}")
        print(f"是否达成共识: {treatment_plan['mdt_discussion']['consensus_achieved']}")
        
        return treatment_plan
        
    except Exception as e:
        print(f"生成治疗方案时出错: {e}")
        logger.error(f"Treatment plan generation failed: {e}")
        return None


if __name__ == "__main__":
    main()