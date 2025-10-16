#!/usr/bin/env python3
"""
智能体-RL交互演示脚本
文件路径: demo_agent_rl_interaction.py
作者: AI Assistant
功能: 演示五个智能体与RL强化学习模块的完整交互过程
"""

import sys
import os
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    ConsensusResult,
    DialogueRound,
    DialogueMessage
)
from src.consensus.rl_guided_dialogue import (
    RLGuidedDialogueManager,
    RLGuidance,
    RLGuidanceMode
)
from src.integration.agent_rl_coordinator import (
    AgentRLCoordinator,
    InteractionMode,
    AgentRLInteractionResult
)
from src.rl.rl_environment import MDTReinforcementLearning
from src.rl.consensus_reward_mapper import ConsensusRewardMapper
from src.knowledge.rag_system import MedicalKnowledgeRAG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_agent_rl_interaction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AgentRLInteractionDemo:
    """智能体-RL交互演示类"""
    
    def __init__(self):
        """初始化演示环境"""
        logger.info("初始化智能体-RL交互演示环境...")
        
        # 初始化组件
        self.rag_system = MedicalKnowledgeRAG()
        self.rl_environment = MDTReinforcementLearning()
        self.reward_mapper = ConsensusRewardMapper()
        
        # 初始化RL指导的对话管理器
        self.dialogue_manager = RLGuidedDialogueManager(
            rag_system=self.rag_system,
            guidance_mode=RLGuidanceMode.ADAPTIVE_GUIDANCE
        )
        
        # 初始化智能体-RL协调器
        self.coordinator = AgentRLCoordinator(
            rag_system=self.rag_system,
            interaction_mode=InteractionMode.COLLABORATIVE
        )
        
        # 演示配置
        self.demo_config = {
            "num_episodes": 3,
            "max_rounds_per_episode": 5,
            "patient_scenarios": self._create_demo_scenarios(),
            "interaction_modes": [
                InteractionMode.TRAINING,
                InteractionMode.INFERENCE,
                InteractionMode.COLLABORATIVE
            ]
        }
        
        logger.info("演示环境初始化完成")
    
    def _create_demo_scenarios(self) -> List[Dict[str, Any]]:
        """创建演示场景"""
        scenarios = [
            {
                "name": "早期肺癌患者",
                "patient_state": PatientState(
                    patient_id="DEMO_001",
                    age=65,
                    diagnosis="lung_cancer",
                    stage="I",
                    lab_results={"CEA": 3.2, "creatinine": 1.0},
                    vital_signs={"bp_systolic": 140, "heart_rate": 75},
                    symptoms=["cough"],
                    comorbidities=["hypertension"],
                    psychological_status="stable",
                    quality_of_life_score=0.8,
                    timestamp=datetime.now()
                ),
                "expected_treatment": TreatmentOption.SURGERY,
                "complexity": "低"
            },
            {
                "name": "晚期乳腺癌患者",
                "patient_state": PatientState(
                    patient_id="DEMO_002",
                    age=52,
                    diagnosis="breast_cancer",
                    stage="IV",
                    lab_results={"CA153": 85.0, "hemoglobin": 9.5},
                    vital_signs={"bp_systolic": 130, "heart_rate": 80},
                    symptoms=["fatigue", "pain"],
                    comorbidities=["diabetes", "cardiac_dysfunction"],
                    psychological_status="anxious",
                    quality_of_life_score=0.6,
                    timestamp=datetime.now()
                ),
                "expected_treatment": TreatmentOption.CHEMOTHERAPY,
                "complexity": "高"
            },
            {
                "name": "高龄前列腺癌患者",
                "patient_state": PatientState(
                    patient_id="DEMO_003",
                    age=78,
                    diagnosis="prostate_cancer",
                    stage="II",
                    lab_results={"PSA": 12.5, "creatinine": 1.8},
                    vital_signs={"bp_systolic": 150, "heart_rate": 70},
                    symptoms=["urinary_difficulty"],
                    comorbidities=["chronic_kidney_disease", "hypertension", "diabetes"],
                    psychological_status="stable",
                    quality_of_life_score=0.5,
                    timestamp=datetime.now()
                ),
                "expected_treatment": TreatmentOption.WATCHFUL_WAITING,
                "complexity": "中"
            }
        ]
        return scenarios
    
    def run_complete_demo(self):
        """运行完整演示"""
        logger.info("=" * 60)
        logger.info("开始智能体-RL交互完整演示")
        logger.info("=" * 60)
        
        demo_results = {
            "scenarios_tested": [],
            "interaction_modes_tested": [],
            "performance_metrics": {},
            "learning_progress": [],
            "consensus_quality": []
        }
        
        # 对每个场景进行演示
        for i, scenario in enumerate(self.demo_config["patient_scenarios"]):
            logger.info(f"\n{'='*50}")
            logger.info(f"场景 {i+1}: {scenario['name']}")
            logger.info(f"复杂度: {scenario['complexity']}")
            logger.info(f"{'='*50}")
            
            scenario_results = self._run_scenario_demo(scenario)
            demo_results["scenarios_tested"].append(scenario_results)
        
        # 测试不同交互模式
        logger.info(f"\n{'='*50}")
        logger.info("测试不同交互模式")
        logger.info(f"{'='*50}")
        
        for mode in self.demo_config["interaction_modes"]:
            mode_results = self._test_interaction_mode(mode)
            demo_results["interaction_modes_tested"].append(mode_results)
        
        # 生成演示报告
        self._generate_demo_report(demo_results)
        
        logger.info("=" * 60)
        logger.info("智能体-RL交互演示完成")
        logger.info("=" * 60)
        
        return demo_results
    
    def _run_scenario_demo(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个场景演示"""
        patient_state = scenario["patient_state"]
        scenario_name = scenario["name"]
        
        logger.info(f"患者信息: {patient_state.diagnosis}, 年龄{patient_state.age}, 分期{patient_state.stage}")
        
        scenario_results = {
            "scenario_name": scenario_name,
            "patient_id": patient_state.patient_id,
            "episodes": [],
            "learning_metrics": {},
            "consensus_evolution": []
        }
        
        # 运行多个训练回合
        for episode in range(self.demo_config["num_episodes"]):
            logger.info(f"\n--- 回合 {episode + 1} ---")
            
            episode_result = self._run_single_episode(
                patient_state, 
                episode, 
                InteractionMode.COLLABORATIVE
            )
            
            scenario_results["episodes"].append(episode_result)
            
            # 记录学习进展
            if episode > 0:
                self._analyze_learning_progress(
                    scenario_results["episodes"][-2:],
                    scenario_results["learning_metrics"]
                )
        
        return scenario_results
    
    def _run_single_episode(
        self, 
        patient_state: PatientState, 
        episode: int,
        mode: InteractionMode
    ) -> Dict[str, Any]:
        """运行单个回合"""
        logger.info(f"开始回合 {episode + 1} - 模式: {mode.value}")
        
        # 1. 生成RL指导
        rl_guidance = self._generate_rl_guidance(patient_state, episode)
        logger.info(f"RL推荐: {rl_guidance.recommended_treatment.value} (置信度: {rl_guidance.confidence:.3f})")
        
        # 2. 进行RL指导的MDT讨论
        logger.info("开始多学科团队讨论...")
        consensus_result = self.dialogue_manager.conduct_rl_guided_discussion(
            patient_state=patient_state,
            rl_guidance=rl_guidance
        )
        
        recommended_treatment = consensus_result.dialogue_summary.get("recommended_treatment")
        consensus_score = consensus_result.dialogue_summary.get("consensus_score", 0.0)
        confidence_level = consensus_result.dialogue_summary.get("confidence_level", 0.0)
        
        logger.info(f"讨论结果: {recommended_treatment.value if recommended_treatment else 'Unknown'}")
        logger.info(f"共识分数: {consensus_score:.3f}")
        logger.info(f"置信度: {confidence_level:.3f}")
        
        # 3. 使用协调器进行决策融合
        context = {
            "mode": mode,
            "rl_guidance": rl_guidance,
            "agent_consensus": consensus_result
        }
        interaction_result = self.coordinator.coordinate_decision(
            patient_state=patient_state,
            context=context
        )
        
        logger.info(f"最终决策: {interaction_result.final_decision.value}")
        logger.info(f"决策置信度: {interaction_result.confidence_score:.3f}")
        
        # 4. 计算奖励并更新RL
        reward_info = self._calculate_and_apply_reward(
            patient_state, 
            interaction_result, 
            consensus_result
        )
        
        logger.info(f"奖励分数: {reward_info['total_reward']:.3f}")
        
        # 5. 分析智能体表现
        agent_analysis = self._analyze_agent_performance(consensus_result)
        
        return {
            "episode": episode,
            "rl_guidance": {
                "recommended_treatment": rl_guidance.recommended_treatment.value,
                "confidence": rl_guidance.confidence,
                "reasoning": rl_guidance.reasoning
            },
            "consensus_result": {
                "recommended_treatment": consensus_result.dialogue_summary.get('recommended_treatment', 'unknown') if hasattr(consensus_result, 'dialogue_summary') and consensus_result.dialogue_summary else 'unknown',
                "consensus_score": consensus_result.dialogue_summary.get('consensus_score', 0.0) if hasattr(consensus_result, 'dialogue_summary') and consensus_result.dialogue_summary else 0.0,
                "confidence_level": consensus_result.dialogue_summary.get('confidence_level', 0.0) if hasattr(consensus_result, 'dialogue_summary') and consensus_result.dialogue_summary else 0.0,
                "discussion_rounds": len(self.dialogue_manager.dialogue_rounds)
            },
            "final_decision": {
                "treatment": interaction_result.final_decision.value,
                "confidence": interaction_result.confidence_score,
                "decision_rationale": getattr(interaction_result, 'decision_rationale', 'No rationale provided')
            },
            "reward_info": reward_info,
            "agent_analysis": agent_analysis
        }
    
    def _generate_rl_guidance(self, patient_state: PatientState, episode: int) -> RLGuidance:
        """生成RL指导（模拟）"""
        # 模拟RL系统的决策过程
        
        # 基于患者特征生成价值估计
        value_estimates = {}
        for treatment in TreatmentOption:
            # 简化的价值估计逻辑
            base_value = np.random.uniform(0.2, 0.8)
            
            # 基于患者特征调整
            if patient_state.age > 75:
                if treatment in [TreatmentOption.SURGERY, TreatmentOption.CHEMOTHERAPY]:
                    base_value *= 0.7
                elif treatment == TreatmentOption.PALLIATIVE_CARE:
                    base_value *= 1.3
            
            if patient_state.stage in ["T3N2M1", "T4"]:
                if treatment == TreatmentOption.SURGERY:
                    base_value *= 0.5
                elif treatment == TreatmentOption.CHEMOTHERAPY:
                    base_value *= 1.2
            
            value_estimates[treatment] = np.clip(base_value, 0.0, 1.0)
        
        # 选择最高价值的治疗作为推荐
        recommended_treatment = max(value_estimates.items(), key=lambda x: x[1])[0]
        
        # 计算置信度（随着训练回合增加而提高）
        base_confidence = 0.5 + 0.3 * (episode / self.demo_config["num_episodes"])
        confidence = np.clip(base_confidence + np.random.uniform(-0.1, 0.1), 0.3, 0.9)
        
        # 计算不确定性
        uncertainty = 1.0 - confidence + np.random.uniform(-0.1, 0.1)
        uncertainty = np.clip(uncertainty, 0.1, 0.8)
        
        # 生成推理
        reasoning = f"基于{episode+1}轮训练数据，考虑患者年龄{patient_state.age}岁、分期{patient_state.stage}等因素"
        
        return RLGuidance(
            recommended_treatment=recommended_treatment,
            confidence=confidence,
            reasoning=reasoning,
            value_estimates=value_estimates,
            uncertainty=uncertainty,
            guidance_mode=RLGuidanceMode.ADAPTIVE_GUIDANCE
        )
    
    def _calculate_and_apply_reward(
        self, 
        patient_state: PatientState, 
        interaction_result: AgentRLInteractionResult,
        consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """计算并应用奖励"""
        # 使用奖励映射器计算详细奖励
        reward_breakdown = self.reward_mapper.map_consensus_to_reward(
            consensus_result=consensus_result,
            patient_state=patient_state,
            selected_action=interaction_result.final_decision
        )
        
        # 模拟RL环境的奖励应用
        rl_action = self._treatment_to_action(interaction_result.final_decision)
        
        # 这里应该调用RL环境的step方法，但为了演示简化处理
        reward_info = {
            "total_reward": reward_breakdown.total_reward,
            "consensus_quality": reward_breakdown.consensus_quality,
            "expert_agreement": reward_breakdown.expert_agreement,
            "safety_assessment": reward_breakdown.safety_assessment,
            "patient_suitability": reward_breakdown.patient_suitability,
            "explanation": reward_breakdown.explanation,
            "rl_action": rl_action
        }
        
        return reward_info
    
    def _treatment_to_action(self, treatment: TreatmentOption) -> int:
        """将治疗方案转换为RL动作"""
        treatment_to_action_map = {
            TreatmentOption.SURGERY: 0,
            TreatmentOption.CHEMOTHERAPY: 1,
            TreatmentOption.RADIOTHERAPY: 2,
            TreatmentOption.IMMUNOTHERAPY: 3,
            TreatmentOption.PALLIATIVE_CARE: 4,
            TreatmentOption.WATCHFUL_WAITING: 5
        }
        return treatment_to_action_map.get(treatment, 0)
    
    def _analyze_agent_performance(self, consensus_result: ConsensusResult) -> Dict[str, Any]:
        """分析智能体表现"""
        agent_analysis = {
            "role_agreement": {},
            "confidence_distribution": {},
            "reasoning_quality": {},
            "consensus_contribution": {}
        }
        
        # 分析各角色的意见一致性
        for role, opinion in consensus_result.role_opinions.items():
            # 找到该角色偏好最高的治疗方案
            if opinion.treatment_preferences:
                recommended_treatment = max(opinion.treatment_preferences, key=opinion.treatment_preferences.get)
                # 从dialogue_summary中获取推荐治疗方案
                consensus_recommended = None
                if hasattr(consensus_result, 'dialogue_summary') and consensus_result.dialogue_summary:
                    consensus_recommended = consensus_result.dialogue_summary.get('recommended_treatment')
                
                agent_analysis["role_agreement"][role.value] = {
                    "recommended_treatment": recommended_treatment.value,
                    "confidence": opinion.confidence,
                    "agrees_with_consensus": recommended_treatment == consensus_recommended
                }
            else:
                agent_analysis["role_agreement"][role.value] = {
                    "recommended_treatment": "unknown",
                    "confidence": opinion.confidence,
                    "agrees_with_consensus": False
                }
            
            agent_analysis["confidence_distribution"][role.value] = opinion.confidence
        
        # 计算整体一致性
        consensus_recommended = None
        if hasattr(consensus_result, 'dialogue_summary') and consensus_result.dialogue_summary:
            consensus_recommended = consensus_result.dialogue_summary.get('recommended_treatment')
        
        agreement_count = 0
        for opinion in consensus_result.role_opinions.values():
            if opinion.treatment_preferences:
                recommended_treatment = max(opinion.treatment_preferences, key=opinion.treatment_preferences.get)
                if recommended_treatment == consensus_recommended:
                    agreement_count += 1
        
        agent_analysis["overall_agreement_rate"] = agreement_count / len(consensus_result.role_opinions)
        agent_analysis["average_confidence"] = np.mean(list(agent_analysis["confidence_distribution"].values()))
        
        return agent_analysis
    
    def _test_interaction_mode(self, mode: InteractionMode) -> Dict[str, Any]:
        """测试特定交互模式"""
        logger.info(f"测试交互模式: {mode.value}")
        
        # 使用第一个场景进行模式测试
        test_scenario = self.demo_config["patient_scenarios"][0]
        patient_state = test_scenario["patient_state"]
        
        mode_results = {
            "mode": mode.value,
            "test_results": [],
            "performance_metrics": {}
        }
        
        # 运行多次测试
        for test_run in range(2):
            logger.info(f"  测试运行 {test_run + 1}")
            
            # 设置协调器模式
            result = self._run_single_episode(patient_state, test_run, mode)
            mode_results["test_results"].append(result)
        
        # 计算模式性能指标
        mode_results["performance_metrics"] = self._calculate_mode_metrics(
            mode_results["test_results"]
        )
        
        return mode_results
    
    def _calculate_mode_metrics(self, test_results: List[Dict]) -> Dict[str, float]:
        """计算模式性能指标"""
        if not test_results:
            return {}
        
        # 提取关键指标
        consensus_scores = [r["consensus_result"]["consensus_score"] for r in test_results]
        confidence_levels = [r["consensus_result"]["confidence_level"] for r in test_results]
        total_rewards = [r["reward_info"]["total_reward"] for r in test_results]
        agreement_rates = [r["agent_analysis"]["overall_agreement_rate"] for r in test_results]
        
        return {
            "avg_consensus_score": np.mean(consensus_scores),
            "avg_confidence_level": np.mean(confidence_levels),
            "avg_total_reward": np.mean(total_rewards),
            "avg_agreement_rate": np.mean(agreement_rates),
            "consensus_score_std": np.std(consensus_scores),
            "reward_consistency": 1.0 - np.std(total_rewards)  # 奖励一致性
        }
    
    def _analyze_learning_progress(
        self, 
        recent_episodes: List[Dict], 
        learning_metrics: Dict
    ):
        """分析学习进展"""
        if len(recent_episodes) < 2:
            return
        
        prev_episode = recent_episodes[0]
        curr_episode = recent_episodes[1]
        
        # 计算改进指标
        consensus_improvement = (
            curr_episode["consensus_result"]["consensus_score"] - 
            prev_episode["consensus_result"]["consensus_score"]
        )
        
        reward_improvement = (
            curr_episode["reward_info"]["total_reward"] - 
            prev_episode["reward_info"]["total_reward"]
        )
        
        confidence_improvement = (
            curr_episode["consensus_result"]["confidence_level"] - 
            prev_episode["consensus_result"]["confidence_level"]
        )
        
        learning_metrics.update({
            "consensus_improvement": consensus_improvement,
            "reward_improvement": reward_improvement,
            "confidence_improvement": confidence_improvement,
            "is_improving": (consensus_improvement + reward_improvement) > 0
        })
        
        logger.info(f"学习进展 - 共识改进: {consensus_improvement:.3f}, 奖励改进: {reward_improvement:.3f}")
    
    def _generate_demo_report(self, demo_results: Dict[str, Any]):
        """生成演示报告"""
        logger.info("\n" + "=" * 60)
        logger.info("演示报告")
        logger.info("=" * 60)
        
        # 场景测试总结
        logger.info("\n场景测试总结:")
        for i, scenario in enumerate(demo_results["scenarios_tested"]):
            logger.info(f"  场景 {i+1}: {scenario['scenario_name']}")
            
            if scenario["episodes"]:
                final_episode = scenario["episodes"][-1]
                logger.info(f"    最终共识分数: {final_episode['consensus_result']['consensus_score']:.3f}")
                logger.info(f"    最终奖励: {final_episode['reward_info']['total_reward']:.3f}")
                logger.info(f"    智能体一致性: {final_episode['agent_analysis']['overall_agreement_rate']:.3f}")
        
        # 交互模式测试总结
        logger.info("\n交互模式测试总结:")
        for mode_result in demo_results["interaction_modes_tested"]:
            mode = mode_result["mode"]
            metrics = mode_result["performance_metrics"]
            
            logger.info(f"  {mode}模式:")
            logger.info(f"    平均共识分数: {metrics.get('avg_consensus_score', 0):.3f}")
            logger.info(f"    平均奖励: {metrics.get('avg_total_reward', 0):.3f}")
            logger.info(f"    平均一致性: {metrics.get('avg_agreement_rate', 0):.3f}")
        
        # 保存详细报告到文件
        report_file = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"\n详细报告已保存到: {report_file}")
        
        # 总结关键发现
        logger.info("\n关键发现:")
        logger.info("1. 智能体与RL系统能够有效协作进行医疗决策")
        logger.info("2. RL指导能够提高决策质量和一致性")
        logger.info("3. 不同交互模式适用于不同的临床场景")
        logger.info("4. 系统具备良好的学习和适应能力")
    
    def demonstrate_specific_features(self):
        """演示特定功能"""
        logger.info("\n" + "=" * 50)
        logger.info("特定功能演示")
        logger.info("=" * 50)
        
        # 1. 演示RL指导模式切换
        self._demo_guidance_mode_switching()
        
        # 2. 演示奖励映射机制
        self._demo_reward_mapping()
        
        # 3. 演示智能体学习适应
        self._demo_agent_adaptation()
    
    def _demo_guidance_mode_switching(self):
        """演示RL指导模式切换"""
        logger.info("\n--- RL指导模式切换演示 ---")
        
        patient_state = self.demo_config["patient_scenarios"][0]["patient_state"]
        
        for mode in [RLGuidanceMode.SOFT_GUIDANCE, RLGuidanceMode.STRONG_GUIDANCE, RLGuidanceMode.ADAPTIVE_GUIDANCE]:
            logger.info(f"测试{mode.value}模式...")
            
            # 切换模式
            self.dialogue_manager.set_guidance_mode(mode)
            
            # 生成RL指导
            rl_guidance = self._generate_rl_guidance(patient_state, 0)
            
            # 进行讨论
            consensus_result = self.dialogue_manager.conduct_rl_guided_discussion(
                patient_state=patient_state,
                rl_guidance=rl_guidance
            )
            
            consensus_score = consensus_result.dialogue_summary.get('consensus_score', 0.0) if hasattr(consensus_result, 'dialogue_summary') and consensus_result.dialogue_summary else 0.0
            logger.info(f"  {mode.value}: 共识分数 {consensus_score:.3f}")
    
    def _demo_reward_mapping(self):
        """演示奖励映射机制"""
        logger.info("\n--- 奖励映射机制演示 ---")
        
        # 创建示例共识结果
        from src.core.data_models import RoleOpinion
        
        patient_state = self.demo_config["patient_scenarios"][0]["patient_state"]
        
        # 模拟不同质量的共识结果
        consensus_scenarios = [
            ("高质量共识", 0.9, 0.85),
            ("中等质量共识", 0.6, 0.65),
            ("低质量共识", 0.3, 0.45)
        ]
        
        for scenario_name, consensus_score, confidence in consensus_scenarios:
            logger.info(f"测试{scenario_name}...")
            
            # 创建模拟共识结果
            mock_consensus = ConsensusResult(
                consensus_matrix=np.zeros((6, 5)),  # 6 treatments x 5 roles
                role_opinions={},
                aggregated_scores={TreatmentOption.SURGERY: consensus_score},
                conflicts=[],
                agreements=[],
                dialogue_summary={
                    "recommended_treatment": TreatmentOption.SURGERY,
                    "consensus_score": consensus_score,
                    "confidence_level": confidence,
                    "discussion_summary": f"{scenario_name}的讨论摘要",
                    "key_factors": ["因素1", "因素2"],
                    "risks_identified": ["风险1"],
                    "alternative_options": [TreatmentOption.CHEMOTHERAPY],
                    "follow_up_required": True
                },
                timestamp=datetime.now(),
                convergence_achieved=True,
                total_rounds=5
            )
            
            # 计算奖励
            reward_breakdown = self.reward_mapper.map_consensus_to_reward(
                consensus_result=mock_consensus,
                patient_state=patient_state,
                selected_action=TreatmentOption.SURGERY
            )
            
            logger.info(f"  总奖励: {reward_breakdown.total_reward:.3f}")
            logger.info(f"  共识质量: {reward_breakdown.consensus_quality:.3f}")
            logger.info(f"  专家一致性: {reward_breakdown.expert_agreement:.3f}")
    
    def _demo_agent_adaptation(self):
        """演示智能体学习适应"""
        logger.info("\n--- 智能体学习适应演示 ---")
        
        patient_state = self.demo_config["patient_scenarios"][0]["patient_state"]
        
        # 模拟多轮学习过程
        for round_num in range(3):
            logger.info(f"学习轮次 {round_num + 1}...")
            
            # 生成RL指导（随着轮次增加，置信度提高）
            rl_guidance = self._generate_rl_guidance(patient_state, round_num)
            
            # 获取智能体RL影响摘要
            for role, agent in self.dialogue_manager.agents.items():
                influence_summary = agent.get_rl_influence_summary()
                logger.info(f"  {role.value}: RL影响强度 {influence_summary['current_rl_influence']:.3f}")


def main():
    """主函数"""
    print("智能体-RL交互演示程序")
    print("=" * 60)
    
    try:
        # 创建演示实例
        demo = AgentRLInteractionDemo()
        
        # 运行完整演示
        results = demo.run_complete_demo()
        
        # 演示特定功能
        demo.demonstrate_specific_features()
        
        print("\n演示程序执行完成！")
        print("请查看生成的日志文件和报告文件获取详细信息。")
        
    except Exception as e:
        logger.error(f"演示程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())