"""
智能数据生成器
文件路径: src/utils/data_generator.py
作者: 姚刚
功能: 整合Memory System和LLM，生成高质量的医疗数据和治疗方案
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# 导入系统组件
from ..core.data_models import PatientState, TreatmentOption, RoleType, MemoryState
from ..knowledge.rag_system import MedicalKnowledgeRAG
from ..consensus.role_agents import RoleAgent
from ..consensus.dialogue_manager import MultiAgentDialogueManager
from .llm_interface import LLMInterface, LLMConfig

# 导入Memory Controller
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../total_code'))
from memory_state import MemoryControllerInterface

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """数据生成配置"""
    # LLM配置
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    
    # 数据生成配置
    simulation_days: int = 30
    patients_per_batch: int = 10
    include_dialogue: bool = True
    include_timeline: bool = True
    
    # 质量控制
    min_confidence_threshold: float = 0.6
    max_retry_attempts: int = 3
    
    # 输出配置
    output_format: str = "json"  # json, csv, excel
    include_metadata: bool = True


class IntelligentDataGenerator:
    """智能数据生成器"""
    
    def __init__(self, config: DataGenerationConfig = None):
        self.config = config or DataGenerationConfig()
        
        # 初始化组件
        self.llm_interface = LLMInterface(LLMConfig(
            model_name=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        ))
        
        self.memory_controller = MemoryControllerInterface()
        self.rag_system = MedicalKnowledgeRAG()
        self.dialogue_manager = MultiAgentDialogueManager(self.rag_system)
        
        # 数据存储
        self.generated_data = []
        self.generation_metadata = {
            "start_time": None,
            "end_time": None,
            "total_patients": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "config": asdict(self.config)
        }
        
        logger.info("IntelligentDataGenerator initialized")
    
    def generate_patient_treatment_dataset(
        self, 
        num_patients: int = None,
        patient_profiles: List[Dict] = None
    ) -> Dict[str, Any]:
        """生成患者治疗数据集"""
        
        num_patients = num_patients or self.config.patients_per_batch
        self.generation_metadata["start_time"] = datetime.now().isoformat()
        self.generation_metadata["total_patients"] = num_patients
        
        logger.info(f"Starting generation of {num_patients} patient treatment records")
        
        # 生成或使用提供的患者档案
        if patient_profiles is None:
            patient_profiles = self._generate_patient_profiles(num_patients)
        
        generated_records = []
        
        for i, profile in enumerate(patient_profiles):
            try:
                logger.info(f"Generating data for patient {i+1}/{num_patients}")
                
                # 创建患者状态
                patient_state = self._create_patient_state_from_profile(profile)
                
                # 生成完整的患者数据记录
                patient_record = self._generate_complete_patient_record(patient_state)
                
                if patient_record:
                    generated_records.append(patient_record)
                    self.generation_metadata["successful_generations"] += 1
                else:
                    self.generation_metadata["failed_generations"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to generate data for patient {i+1}: {e}")
                self.generation_metadata["failed_generations"] += 1
        
        self.generation_metadata["end_time"] = datetime.now().isoformat()
        self.generated_data = generated_records
        
        # 构建最终数据集
        dataset = {
            "patients": generated_records,
            "metadata": self.generation_metadata,
            "summary": self._generate_dataset_summary(generated_records)
        }
        
        logger.info(f"Dataset generation completed. Success: {self.generation_metadata['successful_generations']}, Failed: {self.generation_metadata['failed_generations']}")
        
        return dataset
    
    def generate_enhanced_treatment_plan(
        self, 
        patient_id: str,
        use_memory_context: bool = True,
        use_llm_enhancement: bool = True
    ) -> Dict[str, Any]:
        """为特定患者生成增强的治疗方案"""
        
        try:
            # 获取患者记忆状态
            if use_memory_context:
                memory_state = self.memory_controller.retrieve(patient_id)
                patient_summary = self.memory_controller.generate_patient_state_summary(patient_id)
            else:
                memory_state = None
                patient_summary = {}
            
            # 创建患者状态对象
            patient_state = self._create_patient_state_from_memory(memory_state, patient_summary)
            
            # 获取相关医学知识
            knowledge_context = self.rag_system.retrieve_relevant_knowledge(
                patient_state, 
                query_type="treatment_planning"
            )
            
            # 生成基础治疗方案（多角色协商）
            base_consensus = self.dialogue_manager.conduct_mdt_discussion(patient_state)
            
            # LLM增强治疗方案
            if use_llm_enhancement:
                enhanced_plan = self.llm_interface.generate_treatment_plan(
                    patient_state,
                    patient_summary,
                    knowledge_context
                )
            else:
                enhanced_plan = {}
            
            # 整合结果
            treatment_plan = {
                "patient_id": patient_id,
                "patient_state": asdict(patient_state),
                "mdt_consensus": asdict(base_consensus),
                "llm_enhanced_plan": enhanced_plan,
                "knowledge_context": knowledge_context,
                "generation_timestamp": datetime.now().isoformat(),
                "generation_config": {
                    "use_memory_context": use_memory_context,
                    "use_llm_enhancement": use_llm_enhancement,
                    "include_dialogue": self.config.include_dialogue
                }
            }
            
            return treatment_plan
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced treatment plan for {patient_id}: {e}")
            return {"error": str(e), "patient_id": patient_id}
    
    def generate_patient_timeline_simulation(
        self, 
        patient_id: str,
        simulation_days: int = None
    ) -> Dict[str, Any]:
        """生成患者时间线模拟"""
        
        simulation_days = simulation_days or self.config.simulation_days
        
        try:
            # 获取患者当前状态
            memory_state = self.memory_controller.retrieve(patient_id)
            patient_summary = self.memory_controller.generate_patient_state_summary(patient_id)
            patient_state = self._create_patient_state_from_memory(memory_state, patient_summary)
            
            # 生成时间线事件
            timeline_events = self.llm_interface.generate_patient_timeline_events(
                patient_state,
                patient_summary,
                simulation_days
            )
            
            # 模拟每日状态变化
            daily_states = []
            current_memory = memory_state
            
            for day in range(1, simulation_days + 1):
                # 获取当天事件
                day_events = [e for e in timeline_events if e.get("day") == day]
                
                # 更新记忆状态
                if day_events:
                    self.memory_controller.update_daily(patient_id, day_events)
                    current_memory = self.memory_controller.retrieve(patient_id)
                
                # 记录当天状态
                daily_state = {
                    "day": day,
                    "date": (datetime.now() + timedelta(days=day-1)).isoformat(),
                    "events": day_events,
                    "patient_summary": self.memory_controller.generate_patient_state_summary(patient_id),
                    "memory_state": asdict(current_memory) if current_memory else None
                }
                
                daily_states.append(daily_state)
            
            timeline_simulation = {
                "patient_id": patient_id,
                "simulation_period": {
                    "start_date": datetime.now().isoformat(),
                    "end_date": (datetime.now() + timedelta(days=simulation_days)).isoformat(),
                    "total_days": simulation_days
                },
                "initial_state": asdict(patient_state),
                "timeline_events": timeline_events,
                "daily_states": daily_states,
                "generation_timestamp": datetime.now().isoformat()
            }
            
            return timeline_simulation
            
        except Exception as e:
            logger.error(f"Failed to generate timeline simulation for {patient_id}: {e}")
            return {"error": str(e), "patient_id": patient_id}
    
    def _generate_patient_profiles(self, num_patients: int) -> List[Dict]:
        """生成患者档案"""
        
        profiles = []
        
        # 预定义的患者模板
        templates = [
            {
                "diagnosis": "Breast Cancer",
                "stage": "II",
                "age_range": (45, 65),
                "gender": "Female",
                "performance_status": "Good"
            },
            {
                "diagnosis": "Lung Cancer", 
                "stage": "IIIA",
                "age_range": (55, 75),
                "gender": "Male",
                "performance_status": "Fair"
            },
            {
                "diagnosis": "Colorectal Cancer",
                "stage": "III",
                "age_range": (50, 70),
                "gender": "Mixed",
                "performance_status": "Good"
            }
        ]
        
        for i in range(num_patients):
            template = templates[i % len(templates)]
            
            profile = {
                "patient_id": f"PATIENT_{i+1:04d}",
                "diagnosis": template["diagnosis"],
                "stage": template["stage"],
                "age": np.random.randint(template["age_range"][0], template["age_range"][1]),
                "gender": template["gender"] if template["gender"] != "Mixed" else np.random.choice(["Male", "Female"]),
                "performance_status": template["performance_status"],
                "comorbidities": self._generate_random_comorbidities(),
                "lab_values": self._generate_random_lab_values(),
                "vital_signs": self._generate_random_vital_signs()
            }
            
            profiles.append(profile)
        
        return profiles
    
    def _generate_random_comorbidities(self) -> List[str]:
        """生成随机合并症"""
        possible_comorbidities = [
            "Hypertension", "Diabetes", "Heart Disease", 
            "COPD", "Kidney Disease", "Liver Disease"
        ]
        
        num_comorbidities = np.random.poisson(1.5)
        return np.random.choice(
            possible_comorbidities, 
            size=min(num_comorbidities, len(possible_comorbidities)), 
            replace=False
        ).tolist()
    
    def _generate_random_lab_values(self) -> Dict[str, float]:
        """生成随机实验室值"""
        return {
            "hemoglobin": np.random.normal(12.0, 2.0),
            "white_blood_cell": np.random.normal(7000, 2000),
            "platelet": np.random.normal(250000, 50000),
            "creatinine": np.random.normal(1.0, 0.3),
            "bilirubin": np.random.normal(1.0, 0.5)
        }
    
    def _generate_random_vital_signs(self) -> Dict[str, float]:
        """生成随机生命体征"""
        return {
            "blood_pressure_systolic": np.random.normal(130, 20),
            "blood_pressure_diastolic": np.random.normal(80, 10),
            "heart_rate": np.random.normal(75, 15),
            "temperature": np.random.normal(36.5, 0.5),
            "respiratory_rate": np.random.normal(16, 3)
        }
    
    def _create_patient_state_from_profile(self, profile: Dict) -> PatientState:
        """从档案创建患者状态"""
        return PatientState(
            patient_id=profile["patient_id"],
            diagnosis=profile["diagnosis"],
            stage=profile["stage"],
            age=profile["age"],
            lab_results=profile.get("lab_values", {}),
            vital_signs=profile.get("vital_signs", {}),
            symptoms=profile.get("symptoms", []),
            comorbidities=profile.get("comorbidities", []),
            psychological_status="stable",
            quality_of_life_score=0.7,
            timestamp=datetime.now().isoformat()
        )
    
    def _create_patient_state_from_memory(
        self, 
        memory_state: MemoryState, 
        patient_summary: Dict
    ) -> PatientState:
        """从记忆状态创建患者状态"""
        
        if not memory_state:
            # 创建默认患者状态
            return PatientState(
                patient_id="UNKNOWN",
                diagnosis="Unknown",
                stage="Unknown",
                age=60,
                lab_results={},
                vital_signs={},
                symptoms=[],
                comorbidities=[],
                psychological_status="stable",
                quality_of_life_score=0.7,
                timestamp=datetime.now().isoformat()
            )
        
        # 从记忆状态提取信息
        individual_memory = memory_state.individual_memory
        
        return PatientState(
            patient_id=memory_state.patient_id,
            diagnosis=patient_summary.get("diagnosis", "Unknown"),
            stage=patient_summary.get("stage", "Unknown"),
            age=patient_summary.get("age", 60),
            lab_results=individual_memory.get("baseline_labs", {}),
            vital_signs=individual_memory.get("baseline_vitals", {}),
            symptoms=patient_summary.get("current_symptoms", []),
            comorbidities=individual_memory.get("comorbidities", []),
            psychological_status="stable",
            quality_of_life_score=0.7,
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_complete_patient_record(self, patient_state: PatientState) -> Dict[str, Any]:
        """生成完整的患者记录"""
        
        try:
            # 1. 生成增强治疗方案
            treatment_plan = self.generate_enhanced_treatment_plan(
                patient_state.patient_id,
                use_memory_context=True,
                use_llm_enhancement=True
            )
            
            # 2. 生成时间线模拟（如果配置启用）
            timeline_simulation = None
            if self.config.include_timeline:
                timeline_simulation = self.generate_patient_timeline_simulation(
                    patient_state.patient_id,
                    self.config.simulation_days
                )
            
            # 3. 整合完整记录
            complete_record = {
                "patient_id": patient_state.patient_id,
                "basic_info": asdict(patient_state),
                "treatment_plan": treatment_plan,
                "timeline_simulation": timeline_simulation,
                "generation_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "generator_version": "1.0",
                    "config": asdict(self.config)
                }
            }
            
            return complete_record
            
        except Exception as e:
            logger.error(f"Failed to generate complete record for {patient_state.patient_id}: {e}")
            return None
    
    def _generate_dataset_summary(self, records: List[Dict]) -> Dict[str, Any]:
        """生成数据集摘要"""
        
        if not records:
            return {"total_records": 0, "error": "No records generated"}
        
        # 统计信息
        total_records = len(records)
        successful_plans = sum(1 for r in records if r.get("treatment_plan") and "error" not in r["treatment_plan"])
        successful_timelines = sum(1 for r in records if r.get("timeline_simulation") and "error" not in r.get("timeline_simulation", {}))
        
        # 诊断分布
        diagnoses = [r["basic_info"]["diagnosis"] for r in records]
        diagnosis_counts = pd.Series(diagnoses).value_counts().to_dict()
        
        # 年龄分布
        ages = [r["basic_info"]["age"] for r in records]
        age_stats = {
            "mean": np.mean(ages),
            "median": np.median(ages),
            "min": np.min(ages),
            "max": np.max(ages)
        }
        
        summary = {
            "total_records": total_records,
            "successful_treatment_plans": successful_plans,
            "successful_timeline_simulations": successful_timelines,
            "success_rate": {
                "treatment_plans": successful_plans / total_records if total_records > 0 else 0,
                "timeline_simulations": successful_timelines / total_records if total_records > 0 else 0
            },
            "diagnosis_distribution": diagnosis_counts,
            "age_statistics": age_stats,
            "generation_time": {
                "start": self.generation_metadata["start_time"],
                "end": self.generation_metadata["end_time"],
                "duration_seconds": (
                    self.generation_metadata["end_time"] - self.generation_metadata["start_time"]
                ).total_seconds() if self.generation_metadata["end_time"] and self.generation_metadata["start_time"] else 0
            }
        }
        
        return summary
    
    def export_dataset(self, dataset: Dict[str, Any], output_path: str) -> str:
        """导出数据集"""
        
        try:
            if self.config.output_format.lower() == "json":
                output_file = f"{output_path}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2, default=str)
            
            elif self.config.output_format.lower() == "csv":
                # 展平数据为CSV格式
                flattened_data = []
                for record in dataset["patients"]:
                    flat_record = self._flatten_record(record)
                    flattened_data.append(flat_record)
                
                df = pd.DataFrame(flattened_data)
                output_file = f"{output_path}.csv"
                df.to_csv(output_file, index=False, encoding='utf-8')
            
            elif self.config.output_format.lower() == "excel":
                output_file = f"{output_path}.xlsx"
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    # 患者基本信息
                    basic_info_df = pd.DataFrame([
                        record["basic_info"] for record in dataset["patients"]
                    ])
                    basic_info_df.to_excel(writer, sheet_name='Patient_Info', index=False)
                    
                    # 摘要信息
                    summary_df = pd.DataFrame([dataset["summary"]])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            else:
                raise ValueError(f"Unsupported output format: {self.config.output_format}")
            
            logger.info(f"Dataset exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            raise
    
    def _flatten_record(self, record: Dict) -> Dict:
        """展平记录为平面结构"""
        
        flat = {}
        
        # 基本信息
        basic_info = record.get("basic_info", {})
        for key, value in basic_info.items():
            if isinstance(value, (dict, list)):
                flat[f"basic_{key}"] = json.dumps(value, ensure_ascii=False)
            else:
                flat[f"basic_{key}"] = value
        
        # 治疗方案摘要
        treatment_plan = record.get("treatment_plan", {})
        if "mdt_consensus" in treatment_plan:
            consensus = treatment_plan["mdt_consensus"]
            flat["consensus_recommended_treatment"] = consensus.get("recommended_treatment")
            flat["consensus_confidence"] = consensus.get("confidence_score")
        
        if "llm_enhanced_plan" in treatment_plan:
            llm_plan = treatment_plan["llm_enhanced_plan"]
            flat["llm_primary_treatment"] = llm_plan.get("primary_treatment")
            flat["llm_timeline"] = llm_plan.get("timeline")
        
        # 时间线摘要
        timeline = record.get("timeline_simulation", {})
        if timeline and "timeline_events" in timeline:
            events = timeline["timeline_events"]
            flat["timeline_total_events"] = len(events)
            flat["timeline_high_severity_events"] = len([e for e in events if e.get("severity", 0) >= 4])
        
        return flat
    
    def export_to_csv(self, dataset: Dict, output_path: str) -> str:
        """导出数据集为CSV格式"""
        try:
            flattened_data = []
            for record in dataset["patients"]:
                flat_record = self._flatten_record(record)
                flattened_data.append(flat_record)
            
            df = pd.DataFrame(flattened_data)
            output_file = f"{output_path}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            logger.info(f"Dataset exported to CSV: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise
    
    def export_to_excel(self, dataset: Dict, output_path: str) -> str:
        """导出数据集为Excel格式"""
        try:
            output_file = f"{output_path}.xlsx"
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 患者基本信息
                basic_info_df = pd.DataFrame([
                    record["basic_info"] for record in dataset["patients"]
                ])
                basic_info_df.to_excel(writer, sheet_name='Patient_Info', index=False)
                
                # 摘要信息
                summary_df = pd.DataFrame([dataset["summary"]])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            logger.info(f"Dataset exported to Excel: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export to Excel: {e}")
            raise
    
    def export_to_json(self, dataset: Dict, output_path: str) -> str:
        """导出数据集为JSON格式"""
        try:
            output_file = f"{output_path}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Dataset exported to JSON: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            raise