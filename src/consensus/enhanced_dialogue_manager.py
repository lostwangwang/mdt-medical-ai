"""
增强的多智能体对话管理器
文件路径: src/consensus/enhanced_dialogue_manager.py
作者: 姚刚
功能: 集成智能角色选择和增强角色定义的对话管理
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import asyncio
from datetime import datetime
import json

from .enhanced_role_definitions import ExtendedRoleType, EnhancedRoleDefinitions
from .role_factory import IntelligentRoleFactory, RoleSelectionCriteria
from .role_agents import RoleAgent
from .dialogue_manager import MultiAgentDialogueManager
from ..core.data_models import PatientState, RoleType, ConsensusResult, RoleOpinion, TreatmentOption
from ..rl.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)


@dataclass
class EnhancedDialogueState:
    """增强的对话状态"""
    
    round_number: int = 0
    selected_roles: Set[ExtendedRoleType] = field(default_factory=set)
    role_characteristics: Dict[ExtendedRoleType, Any] = field(default_factory=dict)
    interaction_history: List[Dict] = field(default_factory=list)
    consensus_progress: Dict[str, float] = field(default_factory=dict)
    role_confidence_evolution: Dict[ExtendedRoleType, List[float]] = field(default_factory=dict)
    collaboration_metrics: Dict[str, float] = field(default_factory=dict)
    patient_context: Optional[PatientState] = None


@dataclass
class DialogueMetrics:
    """对话质量指标"""
    
    consensus_speed: float = 0.0  # 达成共识的速度
    opinion_diversity: float = 0.0  # 意见多样性
    collaboration_quality: float = 0.0  # 协作质量
    patient_centeredness: float = 0.0  # 以患者为中心程度
    evidence_utilization: float = 0.0  # 证据利用率
    role_participation_balance: float = 0.0  # 角色参与平衡度


class EnhancedMultiAgentDialogueManager:
    """增强的多智能体对话管理器"""
    
    def __init__(
        self,
        rl_agent: Optional[PPOAgent] = None,
        max_rounds: int = 10,
        consensus_threshold: float = 0.8,
        enable_adaptive_roles: bool = True
    ):
        """初始化增强对话管理器"""
        
        self.rl_agent = rl_agent
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.enable_adaptive_roles = enable_adaptive_roles
        
        # 初始化组件
        self.role_factory = IntelligentRoleFactory()
        self.role_definitions = EnhancedRoleDefinitions()
        self.legacy_dialogue_manager = MultiAgentDialogueManager(rl_agent, max_rounds, consensus_threshold)
        
        # 对话状态
        self.dialogue_state = EnhancedDialogueState()
        self.dialogue_metrics = DialogueMetrics()
        
        # 角色智能体缓存
        self.role_agents_cache: Dict[ExtendedRoleType, RoleAgent] = {}
        
    def initialize_dialogue(
        self, 
        patient_state: PatientState,
        treatment_options: List[TreatmentOption],
        custom_role_selection: Optional[Set[ExtendedRoleType]] = None
    ) -> EnhancedDialogueState:
        """初始化对话"""
        
        logger.info("=== 初始化增强MDT对话 ===")
        
        # 重置对话状态
        self.dialogue_state = EnhancedDialogueState()
        self.dialogue_state.patient_context = patient_state
        
        # 智能角色选择
        if custom_role_selection:
            selected_roles = custom_role_selection
            logger.info(f"使用自定义角色选择: {[role.value for role in selected_roles]}")
        else:
            selected_roles, condition_scores = self.role_factory.select_optimal_team(patient_state)
            logger.info(f"智能选择角色: {[role.value for role in selected_roles]}")
            logger.info(f"病情分析结果: {condition_scores}")
        
        self.dialogue_state.selected_roles = selected_roles
        
        # 获取角色特征
        for role in selected_roles:
            characteristics = self.role_definitions.get_role_characteristics(role)
            self.dialogue_state.role_characteristics[role] = characteristics
            
            # 初始化置信度演化
            self.dialogue_state.role_confidence_evolution[role] = []
        
        # 获取交互建议
        interaction_recommendations = self.role_factory.get_role_interaction_recommendations(selected_roles)
        logger.info(f"角色交互建议: {interaction_recommendations}")
        
        # 创建角色智能体
        self._create_role_agents(selected_roles, patient_state, treatment_options)
        
        logger.info(f"对话初始化完成，参与角色: {len(selected_roles)}个")
        return self.dialogue_state
    
    def _create_role_agents(
        self,
        selected_roles: Set[ExtendedRoleType],
        patient_state: PatientState,
        treatment_options: List[TreatmentOption]
    ):
        """创建角色智能体"""
        
        # 转换为原有角色类型以兼容现有系统
        legacy_roles = self.role_factory.convert_to_legacy_roles(selected_roles)
        
        # 为每个原有角色创建智能体
        for legacy_role in legacy_roles:
            if legacy_role not in self.legacy_dialogue_manager.role_agents:
                agent = RoleAgent(legacy_role, patient_state, treatment_options)
                self.legacy_dialogue_manager.role_agents[legacy_role] = agent
                logger.info(f"创建角色智能体: {legacy_role.value}")
    
    async def conduct_enhanced_dialogue(
        self,
        patient_state: PatientState,
        treatment_options: List[TreatmentOption]
    ) -> Tuple[ConsensusResult, DialogueMetrics]:
        """进行增强的MDT对话"""
        
        logger.info("=== 开始增强MDT对话 ===")
        
        # 初始化对话
        self.initialize_dialogue(patient_state, treatment_options)
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 使用原有对话管理器进行对话
        consensus_result = await self.legacy_dialogue_manager.conduct_dialogue(
            patient_state, treatment_options
        )
        
        # 计算对话指标
        end_time = datetime.now()
        dialogue_duration = (end_time - start_time).total_seconds()
        
        self.dialogue_metrics = self._calculate_dialogue_metrics(
            consensus_result, dialogue_duration
        )
        
        # 增强结果分析
        enhanced_result = self._enhance_consensus_result(consensus_result)
        
        logger.info(f"对话完成，轮次: {consensus_result.rounds}, 达成共识: {consensus_result.consensus_reached}")
        logger.info(f"对话质量指标: {self.dialogue_metrics}")
        
        return enhanced_result, self.dialogue_metrics
    
    def _calculate_dialogue_metrics(
        self,
        consensus_result: ConsensusResult,
        dialogue_duration: float
    ) -> DialogueMetrics:
        """计算对话质量指标"""
        
        metrics = DialogueMetrics()
        
        # 共识速度 (轮次越少越好)
        metrics.consensus_speed = max(0, 1.0 - (consensus_result.rounds / self.max_rounds))
        
        # 意见多样性 (基于不同意见的数量)
        unique_opinions = set()
        for opinion in consensus_result.role_opinions.values():
            unique_opinions.add(opinion.recommended_treatment)
        metrics.opinion_diversity = min(len(unique_opinions) / len(consensus_result.role_opinions), 1.0)
        
        # 协作质量 (基于置信度分布)
        confidences = [opinion.confidence for opinion in consensus_result.role_opinions.values()]
        if confidences:
            confidence_std = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) ** 0.5
            metrics.collaboration_quality = max(0, 1.0 - confidence_std)
        
        # 以患者为中心程度 (检查是否考虑了患者偏好)
        patient_mentions = 0
        for opinion in consensus_result.role_opinions.values():
            if any(keyword in opinion.reasoning.lower() for keyword in 
                   ["patient", "quality of life", "preference", "comfort", "family"]):
                patient_mentions += 1
        metrics.patient_centeredness = patient_mentions / len(consensus_result.role_opinions)
        
        # 证据利用率 (检查是否引用了证据)
        evidence_mentions = 0
        for opinion in consensus_result.role_opinions.values():
            if any(keyword in opinion.reasoning.lower() for keyword in 
                   ["study", "research", "evidence", "guideline", "trial", "data"]):
                evidence_mentions += 1
        metrics.evidence_utilization = evidence_mentions / len(consensus_result.role_opinions)
        
        # 角色参与平衡度 (所有角色都应该参与)
        metrics.role_participation_balance = 1.0 if len(consensus_result.role_opinions) > 0 else 0.0
        
        return metrics
    
    def _enhance_consensus_result(self, consensus_result: ConsensusResult) -> ConsensusResult:
        """增强共识结果"""
        
        # 添加角色特征信息到推理中
        enhanced_opinions = {}
        
        for role_type, opinion in consensus_result.role_opinions.items():
            # 查找对应的扩展角色类型
            extended_role = self._find_extended_role(role_type)
            
            if extended_role and extended_role in self.dialogue_state.role_characteristics:
                characteristics = self.dialogue_state.role_characteristics[extended_role]
                
                # 增强推理信息
                enhanced_reasoning = f"{opinion.reasoning}\n\n[角色特征] "
                enhanced_reasoning += f"主要关注: {', '.join(characteristics.primary_concerns[:3])}; "
                enhanced_reasoning += f"专业领域: {', '.join(characteristics.expertise_areas[:2])}; "
                enhanced_reasoning += f"沟通风格: {characteristics.communication_style}"
                
                # 创建增强的意见
                enhanced_opinion = RoleOpinion(
                    recommended_treatment=opinion.recommended_treatment,
                    confidence=opinion.confidence,
                    reasoning=enhanced_reasoning
                )
                enhanced_opinions[role_type] = enhanced_opinion
            else:
                enhanced_opinions[role_type] = opinion
        
        # 创建增强的共识结果
        enhanced_result = ConsensusResult(
            consensus_reached=consensus_result.consensus_reached,
            recommended_treatment=consensus_result.recommended_treatment,
            confidence_score=consensus_result.confidence_score,
            role_opinions=enhanced_opinions,
            rounds=consensus_result.rounds
        )
        
        return enhanced_result
    
    def _find_extended_role(self, legacy_role: RoleType) -> Optional[ExtendedRoleType]:
        """查找对应的扩展角色类型"""
        
        role_mapping = {
            RoleType.ONCOLOGIST: ExtendedRoleType.ONCOLOGIST,
            RoleType.RADIOLOGIST: ExtendedRoleType.RADIOLOGIST,
            RoleType.NURSE: ExtendedRoleType.NURSE,
            RoleType.PSYCHOLOGIST: ExtendedRoleType.PSYCHOLOGIST,
            RoleType.PATIENT_ADVOCATE: ExtendedRoleType.PATIENT_ADVOCATE
        }
        
        return role_mapping.get(legacy_role)
    
    def get_role_interaction_analysis(self) -> Dict[str, Any]:
        """获取角色交互分析"""
        
        analysis = {
            "selected_roles": [role.value for role in self.dialogue_state.selected_roles],
            "role_characteristics_summary": {},
            "interaction_recommendations": self.role_factory.get_role_interaction_recommendations(
                self.dialogue_state.selected_roles
            ),
            "collaboration_metrics": self.dialogue_metrics.__dict__
        }
        
        # 角色特征摘要
        for role, characteristics in self.dialogue_state.role_characteristics.items():
            analysis["role_characteristics_summary"][role.value] = {
                "primary_concerns": characteristics.primary_concerns[:3],
                "expertise_areas": characteristics.expertise_areas[:3],
                "communication_style": characteristics.communication_style,
                "risk_tolerance": characteristics.risk_tolerance
            }
        
        return analysis
    
    def export_dialogue_report(self, output_path: str):
        """导出对话报告"""
        
        report = {
            "dialogue_summary": {
                "timestamp": datetime.now().isoformat(),
                "patient_id": self.dialogue_state.patient_context.patient_id if self.dialogue_state.patient_context else None,
                "selected_roles": [role.value for role in self.dialogue_state.selected_roles],
                "rounds_conducted": self.dialogue_state.round_number,
                "consensus_reached": getattr(self, 'final_consensus_result', {}).get('consensus_reached', False)
            },
            "role_analysis": self.get_role_interaction_analysis(),
            "quality_metrics": self.dialogue_metrics.__dict__,
            "recommendations": {
                "role_optimization": self._generate_role_optimization_recommendations(),
                "process_improvement": self._generate_process_improvement_recommendations()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"对话报告已导出到: {output_path}")
    
    def _generate_role_optimization_recommendations(self) -> List[str]:
        """生成角色优化建议"""
        
        recommendations = []
        
        # 基于对话质量指标生成建议
        if self.dialogue_metrics.consensus_speed < 0.5:
            recommendations.append("建议增加决策支持工具以提高共识达成速度")
        
        if self.dialogue_metrics.opinion_diversity < 0.3:
            recommendations.append("建议引入更多不同专业背景的角色以增加意见多样性")
        
        if self.dialogue_metrics.patient_centeredness < 0.6:
            recommendations.append("建议加强患者代表的参与和发言权重")
        
        if self.dialogue_metrics.evidence_utilization < 0.4:
            recommendations.append("建议集成更多循证医学数据库和指南")
        
        return recommendations
    
    def _generate_process_improvement_recommendations(self) -> List[str]:
        """生成流程改进建议"""
        
        recommendations = []
        
        # 基于角色配置生成建议
        if len(self.dialogue_state.selected_roles) > 6:
            recommendations.append("当前团队规模较大，建议优化角色选择以提高效率")
        
        if len(self.dialogue_state.selected_roles) < 3:
            recommendations.append("当前团队规模较小，建议增加相关专科角色")
        
        # 基于协作质量生成建议
        if self.dialogue_metrics.collaboration_quality < 0.6:
            recommendations.append("建议改进角色间的协作机制和沟通协议")
        
        return recommendations


# 使用示例
async def demonstrate_enhanced_dialogue():
    """演示增强对话管理器的使用"""
    
    logger.info("=== 增强对话管理器演示 ===")
    
    # 创建增强对话管理器
    enhanced_manager = EnhancedMultiAgentDialogueManager(
        max_rounds=8,
        consensus_threshold=0.75,
        enable_adaptive_roles=True
    )
    
    # 模拟患者状态和治疗选项
    from ..core.data_models import PatientState, TreatmentOption
    from datetime import datetime
    
    patient_state = PatientState(
        patient_id="demo_patient_enhanced",
        age=68,
        diagnosis="renal_cell_carcinoma",
        stage="stage_3",
        lab_results={"creatinine": 2.1, "hemoglobin": 9.5},
        vital_signs={"bp": "140/85", "hr": 75},
        symptoms=["fatigue", "weight_loss", "abdominal_pain"],
        comorbidities=["chronic_kidney_disease", "diabetes"],
        psychological_status="concerned",
        quality_of_life_score=0.6,
        timestamp=datetime.now()
    )
    
    treatment_options = [
        TreatmentOption(name="surgery", description="肾部分切除术"),
        TreatmentOption(name="radiotherapy", description="立体定向放疗"),
        TreatmentOption(name="targeted_therapy", description="靶向治疗"),
        TreatmentOption(name="active_surveillance", description="主动监测")
    ]
    
    # 进行增强对话
    consensus_result, dialogue_metrics = await enhanced_manager.conduct_enhanced_dialogue(
        patient_state, treatment_options
    )
    
    # 获取交互分析
    interaction_analysis = enhanced_manager.get_role_interaction_analysis()
    logger.info(f"角色交互分析: {interaction_analysis}")
    
    # 导出报告
    enhanced_manager.export_dialogue_report("enhanced_dialogue_report.json")
    
    return consensus_result, dialogue_metrics


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_dialogue())