"""
智能体协作流程管理器
文件路径: src/workflow/intelligent_collaboration_manager.py
作者: 姚刚
功能: 管理完整的智能体协作流程，包括角色选择、FAISS查询、MDT讨论和决策融合
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import asyncio

from ..core.data_models import PatientState, TreatmentOption, RoleType
from ..consensus.enhanced_role_definitions import ExtendedRoleType, RoleCharacteristics
from ..consensus.role_factory import IntelligentRoleFactory
from ..consensus.enhanced_dialogue_manager import EnhancedMultiAgentDialogueManager
from ..knowledge.enhanced_faiss_integration import EnhancedFAISSManager, SearchResult
from ..treatment.treatment_generator import TreatmentPlanGenerator
from ..utils.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class CollaborationMetrics:
    """协作指标"""
    
    total_processing_time: float = 0.0
    faiss_query_time: float = 0.0
    role_selection_time: float = 0.0
    mdt_discussion_time: float = 0.0
    decision_fusion_time: float = 0.0
    similar_patients_found: int = 0
    roles_selected: List[str] = field(default_factory=list)
    consensus_achieved: bool = False
    total_discussion_rounds: int = 0
    final_confidence: float = 0.0


@dataclass
class CollaborationResult:
    """协作结果"""
    
    patient_id: str
    treatment_plan: Dict[str, Any]
    collaboration_metrics: CollaborationMetrics
    similar_patients: List[SearchResult]
    selected_roles: List[ExtendedRoleType]
    dialogue_summary: Dict[str, Any]
    recommendations_analysis: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class IntelligentCollaborationManager:
    """智能体协作流程管理器"""
    
    def __init__(
        self,
        llm_interface: LLMInterface,
        faiss_db_path: str = "clinical_memory_db",
        enable_faiss: bool = True,
        enable_enhanced_roles: bool = True
    ):
        """
        初始化智能体协作流程管理器
        
        Args:
            llm_interface: LLM接口实例
            faiss_db_path: FAISS数据库路径
            enable_faiss: 是否启用FAISS功能
            enable_enhanced_roles: 是否启用增强角色系统
        """
        self.llm_interface = llm_interface
        self.enable_faiss = enable_faiss
        self.enable_enhanced_roles = enable_enhanced_roles
        
        # 初始化核心组件
        self._initialize_components(faiss_db_path)
        
        # 协作历史
        self.collaboration_history: List[CollaborationResult] = []
        
        logger.info("Intelligent Collaboration Manager initialized")
    
    def _initialize_components(self, faiss_db_path: str):
        """初始化各个组件"""
        
        # 1. 初始化FAISS管理器
        self.faiss_manager = None
        if self.enable_faiss:
            try:
                self.faiss_manager = EnhancedFAISSManager(
                    db_path=faiss_db_path,
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("FAISS Manager initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS Manager: {e}")
                self.enable_faiss = False
        
        # 2. 初始化角色工厂
        self.role_factory = None
        if self.enable_enhanced_roles:
            try:
                self.role_factory = IntelligentRoleFactory()
                logger.info("Role Factory initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Role Factory: {e}")
                self.enable_enhanced_roles = False
        
        # 3. 初始化增强对话管理器
        self.enhanced_dialogue_manager = None
        if self.enable_enhanced_roles and self.role_factory:
            try:
                self.enhanced_dialogue_manager = EnhancedMultiAgentDialogueManager(
                    role_factory=self.role_factory
                )
                logger.info("Enhanced Dialogue Manager initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Dialogue Manager: {e}")
        
        # 4. 初始化治疗方案生成器
        self.treatment_generator = TreatmentPlanGenerator(
            llm_interface=self.llm_interface,
            enable_faiss=self.enable_faiss
        )
        
        # 如果有FAISS管理器，将其传递给治疗生成器
        if self.faiss_manager:
            self.treatment_generator.faiss_manager = self.faiss_manager
    
    async def process_patient_comprehensive(
        self, 
        patient_id: str,
        use_enhanced_workflow: bool = True
    ) -> CollaborationResult:
        """
        综合处理患者，使用完整的智能体协作流程
        
        Args:
            patient_id: 患者ID
            use_enhanced_workflow: 是否使用增强工作流程
            
        Returns:
            CollaborationResult: 协作结果
        """
        start_time = datetime.now()
        metrics = CollaborationMetrics()
        
        try:
            logger.info(f"Starting comprehensive patient processing for {patient_id}")
            
            # 1. 加载患者数据
            patient_state = self.treatment_generator.load_patient_from_faiss(patient_id)
            if not patient_state:
                raise ValueError(f"Patient {patient_id} not found")
            
            # 2. FAISS相似患者查询
            similar_patients = []
            faiss_start = datetime.now()
            if self.faiss_manager:
                similar_patients = await self._query_similar_patients(patient_state)
                metrics.similar_patients_found = len(similar_patients)
            metrics.faiss_query_time = (datetime.now() - faiss_start).total_seconds()
            
            # 3. 智能角色选择
            selected_roles = []
            role_start = datetime.now()
            if self.role_factory and use_enhanced_workflow:
                selected_roles = await self._select_optimal_roles(patient_state)
                metrics.roles_selected = [role.value for role in selected_roles]
            metrics.role_selection_time = (datetime.now() - role_start).total_seconds()
            
            # 4. 增强MDT讨论
            dialogue_result = None
            mdt_start = datetime.now()
            if self.enhanced_dialogue_manager and selected_roles and use_enhanced_workflow:
                dialogue_result = await self._conduct_enhanced_mdt(
                    patient_state, selected_roles, similar_patients
                )
                metrics.consensus_achieved = dialogue_result.get("consensus_achieved", False)
                metrics.total_discussion_rounds = dialogue_result.get("total_rounds", 0)
            metrics.mdt_discussion_time = (datetime.now() - mdt_start).total_seconds()
            
            # 5. 生成治疗方案
            fusion_start = datetime.now()
            if use_enhanced_workflow and dialogue_result:
                treatment_plan = await self._generate_enhanced_treatment_plan(
                    patient_state, dialogue_result, similar_patients
                )
            else:
                # 使用标准流程
                treatment_plan = self.treatment_generator.generate_treatment_plan(patient_id)
            
            metrics.final_confidence = treatment_plan.get("recommended_treatment", {}).get("confidence_score", 0.0)
            metrics.decision_fusion_time = (datetime.now() - fusion_start).total_seconds()
            
            # 6. 分析推荐结果
            recommendations_analysis = self._analyze_recommendations(
                treatment_plan, similar_patients, dialogue_result
            )
            
            # 7. 计算总处理时间
            metrics.total_processing_time = (datetime.now() - start_time).total_seconds()
            
            # 8. 创建协作结果
            collaboration_result = CollaborationResult(
                patient_id=patient_id,
                treatment_plan=treatment_plan,
                collaboration_metrics=metrics,
                similar_patients=similar_patients,
                selected_roles=selected_roles,
                dialogue_summary=dialogue_result or {},
                recommendations_analysis=recommendations_analysis
            )
            
            # 9. 保存到历史
            self.collaboration_history.append(collaboration_result)
            
            logger.info(f"Comprehensive processing completed for {patient_id}")
            logger.info(f"Total time: {metrics.total_processing_time:.2f}s, "
                       f"Similar patients: {metrics.similar_patients_found}, "
                       f"Roles: {len(selected_roles)}, "
                       f"Consensus: {metrics.consensus_achieved}")
            
            return collaboration_result
            
        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}: {e}")
            raise
    
    async def _query_similar_patients(self, patient_state: PatientState) -> List[SearchResult]:
        """查询相似患者"""
        
        try:
            similar_patients = await self.faiss_manager.async_search_similar_patients(
                patient_state, k=5
            )
            logger.info(f"Found {len(similar_patients)} similar patients")
            return similar_patients
        except Exception as e:
            logger.warning(f"Failed to query similar patients: {e}")
            return []
    
    async def _select_optimal_roles(self, patient_state: PatientState) -> List[ExtendedRoleType]:
        """选择最优角色组合"""
        
        try:
            # 分析患者条件
            condition_analysis = self.role_factory.analyze_patient_conditions(patient_state)
            
            # 选择角色团队
            selected_roles = self.role_factory.select_optimal_team(
                condition_analysis, team_size_preference="medium"
            )
            
            logger.info(f"Selected {len(selected_roles)} roles: {[role.value for role in selected_roles]}")
            return selected_roles
            
        except Exception as e:
            logger.warning(f"Failed to select optimal roles: {e}")
            return []
    
    async def _conduct_enhanced_mdt(
        self, 
        patient_state: PatientState, 
        selected_roles: List[ExtendedRoleType],
        similar_patients: List[SearchResult]
    ) -> Dict[str, Any]:
        """进行增强MDT讨论"""
        
        try:
            # 准备上下文信息
            context_info = {
                "similar_patients": [
                    {
                        "patient_id": p.patient_id,
                        "similarity_score": p.score,
                        "diagnosis": p.metadata.get("diagnosis", ""),
                        "outcome": "positive"  # 简化假设
                    }
                    for p in similar_patients[:3]
                ],
                "evidence_level": "high" if len(similar_patients) >= 3 else "medium"
            }
            
            # 进行增强对话
            dialogue_result = await self.enhanced_dialogue_manager.conduct_enhanced_dialogue(
                patient_state=patient_state,
                selected_roles=selected_roles,
                context_info=context_info,
                max_rounds=5
            )
            
            logger.info("Enhanced MDT discussion completed")
            return dialogue_result
            
        except Exception as e:
            logger.warning(f"Enhanced MDT discussion failed: {e}")
            return {}
    
    async def _generate_enhanced_treatment_plan(
        self,
        patient_state: PatientState,
        dialogue_result: Dict[str, Any],
        similar_patients: List[SearchResult]
    ) -> Dict[str, Any]:
        """生成增强治疗方案"""
        
        try:
            # 基础治疗方案
            base_plan = self.treatment_generator.generate_treatment_plan(patient_state.patient_id)
            
            # 增强信息
            enhancement = {
                "evidence_based_insights": {
                    "similar_cases_analysis": len(similar_patients),
                    "expert_consensus_level": dialogue_result.get("consensus_level", "medium"),
                    "recommendation_strength": dialogue_result.get("recommendation_strength", "moderate")
                },
                "collaborative_decision_process": {
                    "participating_roles": dialogue_result.get("participating_roles", []),
                    "discussion_rounds": dialogue_result.get("total_rounds", 0),
                    "key_considerations": dialogue_result.get("key_considerations", [])
                },
                "personalization_factors": {
                    "patient_specific_adjustments": dialogue_result.get("personalization_notes", []),
                    "risk_benefit_analysis": dialogue_result.get("risk_analysis", {}),
                    "quality_of_life_considerations": dialogue_result.get("qol_factors", [])
                }
            }
            
            # 合并增强信息
            base_plan["enhanced_analysis"] = enhancement
            
            logger.info("Enhanced treatment plan generated")
            return base_plan
            
        except Exception as e:
            logger.warning(f"Failed to generate enhanced treatment plan: {e}")
            # 返回基础方案
            return self.treatment_generator.generate_treatment_plan(patient_state.patient_id)
    
    def _analyze_recommendations(
        self,
        treatment_plan: Dict[str, Any],
        similar_patients: List[SearchResult],
        dialogue_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析推荐结果"""
        
        analysis = {
            "recommendation_confidence": treatment_plan.get("recommended_treatment", {}).get("confidence_score", 0.0),
            "evidence_strength": "high" if len(similar_patients) >= 3 else "medium",
            "expert_agreement": dialogue_result.get("consensus_achieved", False) if dialogue_result else False,
            "personalization_level": "high" if dialogue_result else "standard",
            "decision_factors": {
                "similar_cases": len(similar_patients),
                "expert_roles": len(dialogue_result.get("participating_roles", [])) if dialogue_result else 0,
                "discussion_depth": dialogue_result.get("total_rounds", 0) if dialogue_result else 0
            }
        }
        
        # 计算综合推荐强度
        factors = [
            analysis["recommendation_confidence"],
            1.0 if analysis["evidence_strength"] == "high" else 0.7,
            1.0 if analysis["expert_agreement"] else 0.5,
            1.0 if analysis["personalization_level"] == "high" else 0.8
        ]
        analysis["overall_recommendation_strength"] = sum(factors) / len(factors)
        
        return analysis
    
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """获取协作统计信息"""
        
        if not self.collaboration_history:
            return {"message": "No collaboration history available"}
        
        total_cases = len(self.collaboration_history)
        
        # 计算平均指标
        avg_processing_time = sum(c.collaboration_metrics.total_processing_time for c in self.collaboration_history) / total_cases
        avg_similar_patients = sum(c.collaboration_metrics.similar_patients_found for c in self.collaboration_history) / total_cases
        consensus_rate = sum(1 for c in self.collaboration_history if c.collaboration_metrics.consensus_achieved) / total_cases
        avg_confidence = sum(c.collaboration_metrics.final_confidence for c in self.collaboration_history) / total_cases
        
        # 角色使用统计
        role_usage = {}
        for collaboration in self.collaboration_history:
            for role in collaboration.collaboration_metrics.roles_selected:
                role_usage[role] = role_usage.get(role, 0) + 1
        
        return {
            "total_cases_processed": total_cases,
            "average_processing_time": avg_processing_time,
            "average_similar_patients_found": avg_similar_patients,
            "consensus_achievement_rate": consensus_rate,
            "average_confidence_score": avg_confidence,
            "role_usage_statistics": role_usage,
            "faiss_enabled": self.enable_faiss,
            "enhanced_roles_enabled": self.enable_enhanced_roles
        }
    
    def export_collaboration_report(self, output_path: str, patient_id: Optional[str] = None):
        """导出协作报告"""
        
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 过滤数据
            if patient_id:
                collaborations = [c for c in self.collaboration_history if c.patient_id == patient_id]
                filename = f"collaboration_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                collaborations = self.collaboration_history
                filename = f"collaboration_report_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # 准备报告数据
            report_data = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_collaborations": len(collaborations),
                    "patient_filter": patient_id,
                    "system_configuration": {
                        "faiss_enabled": self.enable_faiss,
                        "enhanced_roles_enabled": self.enable_enhanced_roles
                    }
                },
                "statistics": self.get_collaboration_statistics(),
                "collaborations": [
                    {
                        "patient_id": c.patient_id,
                        "timestamp": c.timestamp.isoformat(),
                        "metrics": {
                            "total_processing_time": c.collaboration_metrics.total_processing_time,
                            "similar_patients_found": c.collaboration_metrics.similar_patients_found,
                            "roles_selected": c.collaboration_metrics.roles_selected,
                            "consensus_achieved": c.collaboration_metrics.consensus_achieved,
                            "final_confidence": c.collaboration_metrics.final_confidence
                        },
                        "treatment_summary": {
                            "primary_recommendation": c.treatment_plan.get("recommended_treatment", {}).get("primary_recommendation"),
                            "confidence_score": c.treatment_plan.get("recommended_treatment", {}).get("confidence_score"),
                            "alternative_options": c.treatment_plan.get("recommended_treatment", {}).get("alternative_options", [])
                        },
                        "recommendations_analysis": c.recommendations_analysis
                    }
                    for c in collaborations
                ]
            }
            
            # 保存报告
            report_file = output_path / filename
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Collaboration report exported to {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Failed to export collaboration report: {e}")
            raise
    
    def cleanup(self):
        """清理资源"""
        
        if self.faiss_manager:
            self.faiss_manager.cleanup()
        
        logger.info("Intelligent Collaboration Manager cleaned up")


# 使用示例
async def demonstrate_intelligent_collaboration():
    """演示智能协作功能"""
    
    logger.info("=== 智能协作流程演示 ===")
    
    # 模拟LLM接口
    class MockLLMInterface:
        def generate_response(self, prompt: str, **kwargs) -> str:
            return "Mock response for demonstration"
    
    # 创建协作管理器
    collaboration_manager = IntelligentCollaborationManager(
        llm_interface=MockLLMInterface(),
        enable_faiss=True,
        enable_enhanced_roles=True
    )
    
    # 处理患者
    try:
        result = await collaboration_manager.process_patient_comprehensive(
            patient_id="10037928",
            use_enhanced_workflow=True
        )
        
        logger.info(f"Processing completed for {result.patient_id}")
        logger.info(f"Similar patients found: {result.collaboration_metrics.similar_patients_found}")
        logger.info(f"Roles selected: {result.collaboration_metrics.roles_selected}")
        logger.info(f"Final confidence: {result.collaboration_metrics.final_confidence}")
        
        # 导出报告
        report_path = collaboration_manager.export_collaboration_report("reports")
        logger.info(f"Report exported to: {report_path}")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
    
    finally:
        collaboration_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(demonstrate_intelligent_collaboration())