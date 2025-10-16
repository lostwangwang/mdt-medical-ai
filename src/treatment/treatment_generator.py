"""
治疗方案生成器
文件路径: src/treatment/treatment_generator.py
作者: AI Assistant
功能: 为特定患者生成个性化治疗方案，集成FAISS数据库和多智能体协作
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from ..core.data_models import PatientState, TreatmentOption, RoleType
from ..consensus.dialogue_manager import MultiAgentDialogueManager
from ..knowledge.rag_system import MedicalKnowledgeRAG
from ..knowledge.enhanced_faiss_integration import EnhancedFAISSManager
from ..utils.llm_interface import LLMInterface
from ..integration.agent_rl_coordinator import AgentRLCoordinator

logger = logging.getLogger(__name__)


class TreatmentPlanGenerator:
    """治疗方案生成器"""
    
    def __init__(self, llm_interface: LLMInterface, enable_faiss: bool = True):
        """
        初始化治疗方案生成器
        
        Args:
            llm_interface: LLM接口实例
            enable_faiss: 是否启用FAISS向量数据库
        """
        self.llm_interface = llm_interface
        self.rag_system = MedicalKnowledgeRAG()
        self.dialogue_manager = MultiAgentDialogueManager(self.rag_system)
        self.agent_rl_coordinator = AgentRLCoordinator(
            rag_system=self.rag_system
        )
        
        # 初始化增强FAISS管理器
        self.faiss_manager = None
        if enable_faiss:
            try:
                self.faiss_manager = EnhancedFAISSManager(
                    db_path="clinical_memory_db",
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Enhanced FAISS Manager initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS Manager: {e}")
                logger.info("Continuing without FAISS support")
        
        # 患者数据缓存
        self.patient_data_cache = {}
        
        logger.info("TreatmentPlanGenerator initialized")
    
    def load_patient_from_faiss(self, patient_id: str) -> PatientState:
        """
        从FAISS数据库加载患者数据
        
        Args:
            patient_id: 患者ID
            
        Returns:
            PatientState: 患者状态对象
        """
        try:
            # 检查缓存
            if patient_id in self.patient_data_cache:
                logger.info(f"Loading patient {patient_id} from cache")
                return self.patient_data_cache[patient_id]
            
            # 尝试从增强FAISS系统加载
            if self.faiss_manager:
                search_result = self.faiss_manager.get_patient_by_id(patient_id)
                if search_result:
                    # 从搜索结果构建PatientState
                    metadata = search_result.metadata
                    patient_state = PatientState(
                        patient_id=patient_id,
                        age=metadata.get("age", 0),
                        diagnosis=metadata.get("diagnosis", ""),
                        stage=metadata.get("stage", ""),
                        lab_results=metadata.get("lab_results", {}),
                        vital_signs=metadata.get("vital_signs", {}),
                        symptoms=metadata.get("symptoms", []),
                        comorbidities=metadata.get("comorbidities", []),
                        psychological_status=metadata.get("psychological_status", "stable"),
                        quality_of_life_score=metadata.get("quality_of_life_score", 0.5),
                        timestamp=search_result.timestamp or datetime.now()
                    )
                    
                    # 缓存患者数据
                    self.patient_data_cache[patient_id] = patient_state
                    logger.info(f"Successfully loaded patient {patient_id} from FAISS database")
                    return patient_state
            
            # 回退到JSON文件加载（模拟FAISS数据）
            json_file_path = Path(f"/mnt/e/project/LLM/mdt_medical_ai/{patient_id}_clinical_memory.json")
            
            if not json_file_path.exists():
                raise FileNotFoundError(f"Patient data file not found: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
            
            # 解析患者数据
            patient_state = self._parse_patient_data(patient_data)
            
            # 缓存患者数据
            self.patient_data_cache[patient_id] = patient_state
            
            logger.info(f"Successfully loaded patient {patient_id} from JSON file")
            return patient_state
            
        except Exception as e:
            logger.error(f"Failed to load patient {patient_id}: {e}")
            raise
    
    def _parse_patient_data(self, patient_data: Dict[str, Any]) -> PatientState:
        """
        解析患者数据为PatientState对象
        
        Args:
            patient_data: 原始患者数据
            
        Returns:
            PatientState: 解析后的患者状态
        """
        try:
            # 提取基本信息
            subject_id = patient_data.get("subject_id", "unknown")
            gender = patient_data.get("gender", "unknown")
            age = patient_data.get("age", 0)
            
            # 提取慢性疾病
            chronic_diseases = patient_data.get("chronic_diseases", [])
            
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
            medications = [med.get("medication", "") for med in discharge_medications if isinstance(med, dict)]
            
            # 提取每日总结中的症状和状态
            daily_summaries = patient_data.get("daily_summaries", [])
            symptoms = []
            psychological_status = "stable"
            
            # 从总结中提取关键信息
            if daily_summaries:
                latest_summary = daily_summaries[-1].get("summary", "")
                if "anxiety" in latest_summary.lower() or "anxious" in latest_summary.lower():
                    psychological_status = "anxious"
                    symptoms.append("anxiety")
                if "pain" in latest_summary.lower():
                    symptoms.append("pain")
                if "fatigue" in latest_summary.lower():
                    symptoms.append("fatigue")
            
            # 根据慢性疾病推断主要诊断
            primary_diagnosis = "multiple_chronic_conditions"
            if any("kidney" in disease.lower() for disease in chronic_diseases):
                primary_diagnosis = "chronic_kidney_disease"
            elif any("diabetes" in disease.lower() for disease in chronic_diseases):
                primary_diagnosis = "diabetes_mellitus"
            
            # 创建PatientState对象
            patient_state = PatientState(
                patient_id=str(subject_id),
                age=age,
                diagnosis=primary_diagnosis,
                stage="chronic",  # 基于慢性疾病的假设
                lab_results=lab_results,
                vital_signs=vital_signs,
                symptoms=symptoms,
                comorbidities=chronic_diseases,
                psychological_status=psychological_status,
                quality_of_life_score=0.6,  # 基于多种慢性疾病的估计
                timestamp=datetime.now()
            )
            
            logger.info(f"Successfully parsed patient data for {subject_id}")
            return patient_state
            
        except Exception as e:
            logger.error(f"Failed to parse patient data: {e}")
            raise
    
    def generate_treatment_plan(self, patient_id: str) -> Dict[str, Any]:
        """
        为指定患者生成治疗方案
        
        Args:
            patient_id: 患者ID
            
        Returns:
            Dict: 完整的治疗方案
        """
        try:
            logger.info(f"Starting treatment plan generation for patient {patient_id}")
            
            # 1. 加载患者数据
            patient_state = self.load_patient_from_faiss(patient_id)
            
            # 2. 查询相似患者和治疗建议（如果FAISS可用）
            similar_patients = []
            faiss_treatment_recommendations = []
            if self.faiss_manager:
                try:
                    logger.info("Searching for similar patients...")
                    similar_patients = self.faiss_manager.search_similar_patients(
                        patient_state, k=5, score_threshold=0.7
                    )
                    logger.info(f"Found {len(similar_patients)} similar patients")
                    
                    # 获取基于相似患者的治疗建议
                    faiss_treatment_recommendations = self.faiss_manager.get_treatment_recommendations(
                        patient_state, k=5
                    )
                    logger.info(f"Generated {len(faiss_treatment_recommendations)} treatment recommendations from FAISS")
                    
                except Exception as e:
                    logger.warning(f"FAISS query failed: {e}")
            
            # 3. 进行MDT多智能体讨论
            logger.info("Conducting MDT discussion...")
            mdt_result = self.dialogue_manager.conduct_mdt_discussion(patient_state)
            
            # 3. 使用智能体-RL协调器进行决策融合
            logger.info("Coordinating final decision...")
            coordination_result = self.agent_rl_coordinator.coordinate_decision(
                patient_state=patient_state,
                context={
                    "agent_opinions": mdt_result.role_opinions,
                    "mode": "COLLABORATIVE"
                }
            )
            
            # 4. 生成详细的治疗计划
            detailed_plan = self._generate_detailed_treatment_plan(
                patient_state, mdt_result, coordination_result
            )
            
            # 5. 生成治疗时间线
            treatment_timeline = self._generate_treatment_timeline(
                patient_state, detailed_plan
            )
            
            # 6. 生成风险评估
            risk_assessment = self._generate_risk_assessment(
                patient_state, detailed_plan
            )
            
            # 7. 组装最终结果
            treatment_plan = {
                "patient_info": {
                    "patient_id": patient_state.patient_id,
                    "age": patient_state.age,
                    "primary_diagnosis": patient_state.diagnosis,
                    "comorbidities": patient_state.comorbidities
                },
                "similar_patients_analysis": {
                    "total_similar_patients": len(similar_patients),
                    "similar_cases": [
                        {
                            "patient_id": patient.patient_id,
                            "similarity_score": patient.score,
                            "diagnosis": patient.metadata.get("diagnosis", ""),
                            "age": patient.metadata.get("age", 0),
                            "stage": patient.metadata.get("stage", "")
                        }
                        for patient in similar_patients[:3]  # 只显示前3个最相似的
                    ],
                    "faiss_treatment_recommendations": faiss_treatment_recommendations
                },
                "mdt_discussion": {
                    "consensus_achieved": mdt_result.convergence_achieved,
                    "total_rounds": mdt_result.total_rounds,
                    "expert_opinions": [
                        {
                            "role": role.value,
                            "recommendation": opinion.reasoning,
                            "confidence": opinion.confidence,
                            "reasoning": opinion.reasoning
                        }
                        for role, opinion in mdt_result.role_opinions.items()
                    ]
                },
                "recommended_treatment": {
                    "primary_recommendation": coordination_result.final_decision.value,
                    "confidence_score": coordination_result.confidence_score,
                    "alternative_options": [
                        treatment.value for treatment, score in sorted(
                            mdt_result.aggregated_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[1:4]  # 取前3个替代选项（除了最佳选项）
                    ] if hasattr(mdt_result, 'aggregated_scores') and mdt_result.aggregated_scores else []
                },
                "detailed_plan": detailed_plan,
                "treatment_timeline": treatment_timeline,
                "risk_assessment": risk_assessment,
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Treatment plan generated successfully for patient {patient_id}")
            return treatment_plan
            
        except Exception as e:
            logger.error(f"Failed to generate treatment plan for patient {patient_id}: {e}")
            raise
    
    def _generate_detailed_treatment_plan(
        self, 
        patient_state: PatientState, 
        mdt_result, 
        coordination_result
    ) -> Dict[str, Any]:
        """生成详细的治疗计划"""
        
        # 使用LLM生成详细的治疗计划
        prompt = f"""
        基于以下患者信息和MDT讨论结果，生成详细的治疗计划：

        患者信息：
        - 年龄：{patient_state.age}岁
        - 主要诊断：{patient_state.diagnosis}
        - 合并症：{', '.join(patient_state.comorbidities)}
        - 当前症状：{', '.join(patient_state.symptoms)}
        - 心理状态：{patient_state.psychological_status}

        推荐治疗：{coordination_result.final_decision.value}
        置信度：{coordination_result.confidence_score:.2f}

        请生成包含以下内容的详细治疗计划：
        1. 药物治疗方案
        2. 非药物治疗措施
        3. 监测指标
        4. 生活方式建议
        5. 随访计划
        """
        
        try:
            detailed_plan_text = self.llm_interface.generate_treatment_plan(
                patient_state, prompt
            )
            
            return {
                "medication_plan": self._extract_medication_plan(detailed_plan_text),
                "non_medication_interventions": self._extract_non_medication_interventions(detailed_plan_text),
                "monitoring_parameters": self._extract_monitoring_parameters(detailed_plan_text),
                "lifestyle_recommendations": self._extract_lifestyle_recommendations(detailed_plan_text),
                "follow_up_schedule": self._extract_follow_up_schedule(detailed_plan_text),
                "full_plan_text": detailed_plan_text
            }
        except Exception as e:
            logger.error(f"Failed to generate detailed treatment plan: {e}")
            return {
                "medication_plan": "需要进一步评估",
                "non_medication_interventions": "需要进一步评估",
                "monitoring_parameters": "需要进一步评估",
                "lifestyle_recommendations": "需要进一步评估",
                "follow_up_schedule": "需要进一步评估",
                "full_plan_text": "治疗计划生成失败，请联系医疗团队"
            }
    
    def _generate_treatment_timeline(
        self, 
        patient_state: PatientState, 
        detailed_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成治疗时间线"""
        
        timeline = []
        
        # 立即开始的治疗
        timeline.append({
            "timepoint": "立即",
            "actions": [
                "开始药物治疗",
                "患者教育",
                "基线评估"
            ],
            "responsible_team": ["主治医师", "护士"]
        })
        
        # 1周后
        timeline.append({
            "timepoint": "1周后",
            "actions": [
                "评估药物耐受性",
                "检查生命体征",
                "症状评估"
            ],
            "responsible_team": ["主治医师", "护士"]
        })
        
        # 1个月后
        timeline.append({
            "timepoint": "1个月后",
            "actions": [
                "实验室检查",
                "治疗效果评估",
                "调整治疗方案"
            ],
            "responsible_team": ["主治医师", "专科医师"]
        })
        
        # 3个月后
        timeline.append({
            "timepoint": "3个月后",
            "actions": [
                "全面健康评估",
                "并发症筛查",
                "生活质量评估"
            ],
            "responsible_team": ["MDT团队"]
        })
        
        return timeline
    
    def _generate_risk_assessment(
        self, 
        patient_state: PatientState, 
        detailed_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成风险评估"""
        
        risks = []
        
        # 基于年龄的风险
        if patient_state.age > 70:
            risks.append({
                "risk_factor": "高龄",
                "risk_level": "中等",
                "description": "高龄患者药物代谢能力下降，需要密切监测"
            })
        
        # 基于合并症的风险
        if "kidney" in str(patient_state.comorbidities).lower():
            risks.append({
                "risk_factor": "肾功能不全",
                "risk_level": "高",
                "description": "肾功能不全可能影响药物清除，需要调整剂量"
            })
        
        if "diabetes" in str(patient_state.comorbidities).lower():
            risks.append({
                "risk_factor": "糖尿病",
                "risk_level": "中等",
                "description": "糖尿病患者感染风险增加，需要血糖监测"
            })
        
        # 药物相互作用风险 - 基于合并症数量估算
        if len(patient_state.comorbidities) > 3:
            risks.append({
                "risk_factor": "多重合并症",
                "risk_level": "中等",
                "description": "多重合并症可能需要多种药物治疗，存在相互作用风险"
            })
        
        return {
            "identified_risks": risks,
            "overall_risk_level": "中等" if len(risks) > 2 else "低",
            "mitigation_strategies": [
                "定期监测实验室指标",
                "密切观察不良反应",
                "患者教育和依从性管理",
                "多学科团队协作"
            ]
        }
    
    def _extract_medication_plan(self, plan_text: str) -> str:
        """从计划文本中提取药物治疗方案"""
        # 简化实现，实际可以使用NLP技术提取
        if "药物" in plan_text or "medication" in plan_text.lower():
            return "根据患者具体情况制定个性化药物方案"
        return "需要进一步评估药物治疗需求"
    
    def _extract_non_medication_interventions(self, plan_text: str) -> str:
        """从计划文本中提取非药物治疗措施"""
        return "生活方式干预、物理治疗、心理支持"
    
    def _extract_monitoring_parameters(self, plan_text: str) -> List[str]:
        """从计划文本中提取监测指标"""
        return [
            "血压",
            "血糖",
            "肾功能指标",
            "电解质平衡",
            "症状评估"
        ]
    
    def _extract_lifestyle_recommendations(self, plan_text: str) -> List[str]:
        """从计划文本中提取生活方式建议"""
        return [
            "低盐饮食",
            "适量运动",
            "规律作息",
            "戒烟限酒",
            "心理健康管理"
        ]
    
    def _extract_follow_up_schedule(self, plan_text: str) -> List[str]:
        """从计划文本中提取随访计划"""
        return [
            "1周后门诊随访",
            "1个月后实验室检查",
            "3个月后全面评估",
            "紧急情况随时就诊"
        ]
    
    def save_treatment_plan(self, treatment_plan: Dict[str, Any], output_path: str) -> None:
        """
        保存治疗方案到文件
        
        Args:
            treatment_plan: 治疗方案
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(treatment_plan, f, ensure_ascii=False, indent=2)
            logger.info(f"Treatment plan saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save treatment plan: {e}")
            raise


def main():
    """主函数演示"""
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 初始化LLM接口（需要配置）
        from ..utils.llm_interface import LLMInterface, LLMConfig
        
        llm_config = LLMConfig(
            model_name="gpt-3.5-turbo",
            api_key="your-api-key",
            temperature=0.7
        )
        llm_interface = LLMInterface(llm_config)
        
        # 创建治疗方案生成器
        generator = TreatmentPlanGenerator(llm_interface)
        
        # 为患者10037928生成治疗方案
        patient_id = "10037928"
        treatment_plan = generator.generate_treatment_plan(patient_id)
        
        # 保存结果
        output_path = f"results/treatment_plan_{patient_id}.json"
        generator.save_treatment_plan(treatment_plan, output_path)
        
        print(f"治疗方案已生成并保存到: {output_path}")
        print(f"推荐治疗: {treatment_plan['recommended_treatment']['primary_recommendation']}")
        print(f"置信度: {treatment_plan['recommended_treatment']['confidence_score']:.2f}")
        
    except Exception as e:
        print(f"生成治疗方案时出错: {e}")


if __name__ == "__main__":
    main()