"""
集成工作流程管理器
文件路径: src/integration/workflow_manager.py
作者: Tianyu (系统集成) / 姚刚 (RL集成)
功能: 管理MDT系统各组件间的协调工作流程
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..core.data_models import PatientState, TreatmentOption, ConsensusResult
from ..consensus.consensus_matrix import ConsensusMatrix
from ..consensus.dialogue_manager import MultiAgentDialogueManager
from ..knowledge.rag_system import MedicalKnowledgeRAG
from ..rl.rl_environment import MDTReinforcementLearning

logger = logging.getLogger(__name__)


@dataclass
class WorkflowEvent:
    """工作流程事件"""

    timestamp: datetime
    event_type: (
        str  # "patient_update", "mdt_discussion", "rl_training", "decision_made"
    )
    patient_id: str
    data: Dict[str, Any]
    outcome: Optional[Dict[str, Any]] = None


class MemoryControllerStub:
    """Memory Controller存根 (用于与杜军模块集成前的测试)"""

    def __init__(self):
        self.patient_memories = {}
        self.evolution_rules = {
            "lab_evolution_rate": 0.02,
            "vital_fluctuation": 0.05,
            "symptom_progression": 0.1,
        }

    def retrieve_patient_state(self, patient_id: str, day: int = 0) -> Dict[str, Any]:
        """检索患者状态"""
        if patient_id not in self.patient_memories:
            # 创建初始患者状态
            self.patient_memories[patient_id] = self._create_initial_memory(patient_id)

        # 根据天数模拟演化
        memory = self.patient_memories[patient_id].copy()
        return self._simulate_temporal_evolution(memory, day)

    def update_patient_memory(self, patient_id: str, new_data: Dict[str, Any]) -> None:
        """更新患者记忆"""
        if patient_id not in self.patient_memories:
            self.patient_memories[patient_id] = self._create_initial_memory(patient_id)

        # 更新记忆数据
        self.patient_memories[patient_id]["last_updated"] = datetime.now()
        self.patient_memories[patient_id]["events"].append(new_data)

    def _create_initial_memory(self, patient_id: str) -> Dict[str, Any]:
        """创建初始记忆"""
        return {
            "patient_id": patient_id,
            "initial_state": {
                "age": np.random.randint(40, 85),
                "diagnosis": "breast_cancer",
                "stage": np.random.choice(["I", "II", "III"]),
                "lab_results": {
                    "creatinine": np.random.uniform(0.8, 1.5),
                    "hemoglobin": np.random.uniform(10.0, 14.0),
                },
                "vital_signs": {
                    "bp_systolic": np.random.randint(110, 160),
                    "heart_rate": np.random.randint(65, 95),
                },
                "comorbidities": np.random.choice(
                    [[], ["diabetes"], ["hypertension"], ["diabetes", "hypertension"]]
                ),
                "quality_of_life_score": np.random.uniform(0.4, 0.9),
            },
            "events": [],
            "last_updated": datetime.now(),
        }

    def _simulate_temporal_evolution(
        self, memory: Dict[str, Any], day: int
    ) -> Dict[str, Any]:
        """模拟时间演化"""
        evolved_state = memory["initial_state"].copy()

        if day > 0:
            # 实验室值演化
            creatinine_change = (
                day
                * self.evolution_rules["lab_evolution_rate"]
                * np.random.normal(1, 0.1)
            )
            evolved_state["lab_results"]["creatinine"] += creatinine_change
            evolved_state["lab_results"]["creatinine"] = max(
                0.5, evolved_state["lab_results"]["creatinine"]
            )

            # 生命体征波动
            bp_fluctuation = np.random.normal(0, 10)
            evolved_state["vital_signs"]["bp_systolic"] += bp_fluctuation
            evolved_state["vital_signs"]["bp_systolic"] = max(
                90, min(200, evolved_state["vital_signs"]["bp_systolic"])
            )

            # 生活质量变化
            qol_change = -day * 0.005 + np.random.normal(0, 0.02)  # 略微下降趋势
            evolved_state["quality_of_life_score"] += qol_change
            evolved_state["quality_of_life_score"] = max(
                0.1, min(1.0, evolved_state["quality_of_life_score"])
            )

        return evolved_state


class IntegratedWorkflowManager:
    """集成工作流程管理器"""

    def __init__(self):
        # 初始化各组件
        self.rag_system = MedicalKnowledgeRAG()
        self.consensus_system = ConsensusMatrix()
        self.dialogue_manager = MultiAgentDialogueManager(self.rag_system)
        self.rl_environment = MDTReinforcementLearning(self.consensus_system)

        # 临时使用存根，正式版本中会替换为杜军的Memory Controller
        self.memory_controller = MemoryControllerStub()

        # 工作流程状态
        self.workflow_events = []
        self.active_patients = {}

        logger.info("Integrated Workflow Manager initialized")

    def register_patient(self, patient_id: str, initial_data: Dict[str, Any]) -> None:
        """注册新患者"""
        self.active_patients[patient_id] = {
            "registration_time": datetime.now(),
            "current_status": "active",
            "last_mdt_discussion": None,
            "treatment_decisions": [],
            "rl_training_data": [],
        }

        # 更新记忆系统
        self.memory_controller.update_patient_memory(
            patient_id,
            {
                "event_type": "registration",
                "data": initial_data,
                "timestamp": datetime.now(),
            },
        )

        self._log_workflow_event(
            "patient_registered", patient_id, {"initial_data": initial_data}
        )

        logger.info(f"Patient {patient_id} registered in workflow")

    def run_temporal_simulation(self, patient_id: str, days: int) -> Dict[str, Any]:
        """运行时序模拟"""
        logger.info(f"Starting temporal simulation for {patient_id}, {days} days")

        if patient_id not in self.active_patients:
            # 自动注册患者
            self.register_patient(patient_id, {"auto_generated": True})

        simulation_results = {
            "patient_id": patient_id,
            "simulation_days": days,
            "daily_events": [],
            "mdt_discussions": [],
            "rl_training_episodes": [],
            "performance_metrics": {},
        }

        # 每日模拟循环
        for day in range(days):
            daily_result = self._simulate_single_day(patient_id, day)
            simulation_results["daily_events"].append(daily_result)

            # 检查是否需要MDT讨论
            if self._should_trigger_mdt_discussion(patient_id, day):
                mdt_result = self._trigger_mdt_discussion(patient_id, day)
                simulation_results["mdt_discussions"].append(mdt_result)

                # 基于MDT结果进行RL训练
                rl_episode = self._conduct_rl_training_episode(patient_id, mdt_result)
                simulation_results["rl_training_episodes"].append(rl_episode)

            # 每10天输出进展
            if day % 10 == 0:
                logger.info(f"Simulation day {day} completed for {patient_id}")

        # 计算最终性能指标
        simulation_results["performance_metrics"] = self._calculate_simulation_metrics(
            simulation_results
        )

        logger.info(f"Temporal simulation completed for {patient_id}")
        return simulation_results

    def _simulate_single_day(self, patient_id: str, day: int) -> Dict[str, Any]:
        """模拟单日事件"""
        # 从记忆系统获取当前患者状态
        memory_state = self.memory_controller.retrieve_patient_state(patient_id, day)

        # 生成每日医疗事件
        daily_events = self._generate_daily_medical_events(
            patient_id, day, memory_state
        )

        # 更新记忆系统
        for event in daily_events:
            self.memory_controller.update_patient_memory(patient_id, event)

        # 记录工作流程事件
        self._log_workflow_event(
            "daily_simulation",
            patient_id,
            {"day": day, "events": daily_events, "state": memory_state},
        )

        return {
            "day": day,
            "patient_state": memory_state,
            "medical_events": daily_events,
            "timestamp": datetime.now(),
        }

    def _generate_daily_medical_events(
        self, patient_id: str, day: int, state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成每日医疗事件"""
        events = []

        # 实验室检查 (每3天一次)
        if day % 3 == 0:
            lab_event = {
                "event_type": "lab_test",
                "timestamp": datetime.now() + timedelta(days=day),
                "tests": {
                    "creatinine": state["lab_results"]["creatinine"],
                    "hemoglobin": state["lab_results"]["hemoglobin"],
                },
            }
            events.append(lab_event)

        # 生命体征监测 (每天)
        vital_event = {
            "event_type": "vital_signs",
            "timestamp": datetime.now() + timedelta(days=day),
            "measurements": state["vital_signs"],
        }
        events.append(vital_event)

        # 症状评估 (每5天)
        if day % 5 == 0:
            # 模拟症状变化
            symptoms = ["fatigue"] if state["quality_of_life_score"] < 0.6 else []
            if state["lab_results"]["creatinine"] > 1.5:
                symptoms.append("kidney_dysfunction_symptoms")

            symptom_event = {
                "event_type": "symptom_assessment",
                "timestamp": datetime.now() + timedelta(days=day),
                "symptoms": symptoms,
                "severity_scores": {
                    symptom: np.random.uniform(1, 5) for symptom in symptoms
                },
            }
            events.append(symptom_event)

        # 随机医疗干预 (20%概率)
        if np.random.random() < 0.2:
            intervention_event = {
                "event_type": "medical_intervention",
                "timestamp": datetime.now() + timedelta(days=day),
                "intervention_type": np.random.choice(
                    ["medication_adjustment", "supportive_care", "monitoring"]
                ),
                "details": f"Day {day} routine intervention",
            }
            events.append(intervention_event)

        return events

    def _should_trigger_mdt_discussion(self, patient_id: str, day: int) -> bool:
        """判断是否应该触发MDT讨论"""
        # 定期讨论 (每7天)
        if day % 7 == 0 and day > 0:
            return True

        # 获取当前状态
        memory_state = self.memory_controller.retrieve_patient_state(patient_id, day)

        # 危急情况触发
        if memory_state["lab_results"]["creatinine"] > 2.0:
            logger.warning(f"Critical creatinine level for {patient_id} on day {day}")
            return True

        if memory_state["quality_of_life_score"] < 0.3:
            logger.warning(f"Poor quality of life for {patient_id} on day {day}")
            return True

        return False

    def _trigger_mdt_discussion(self, patient_id: str, day: int) -> Dict[str, Any]:
        """触发MDT讨论"""
        logger.info(f"Triggering MDT discussion for {patient_id} on day {day}")

        # 获取当前患者状态
        memory_state = self.memory_controller.retrieve_patient_state(patient_id, day)

        # 创建PatientState对象
        patient_state = PatientState(
            patient_id=patient_id,
            age=memory_state["age"],
            diagnosis=memory_state["diagnosis"],
            stage=memory_state["stage"],
            lab_results=memory_state["lab_results"],
            vital_signs=memory_state["vital_signs"],
            symptoms=memory_state.get("symptoms", ["fatigue"]),
            comorbidities=memory_state["comorbidities"],
            psychological_status=memory_state.get("psychological_status", "stable"),
            quality_of_life_score=memory_state["quality_of_life_score"],
            timestamp=datetime.now(),
        )

        # 运行MDT讨论
        consensus_result = self.dialogue_manager.conduct_mdt_discussion(patient_state)

        # 更新患者记录
        self.active_patients[patient_id]["last_mdt_discussion"] = datetime.now()

        # 记录决策
        decision = {
            "day": day,
            "recommended_treatment": max(
                consensus_result.aggregated_scores.items(), key=lambda x: x[1]
            )[0],
            "consensus_score": max(consensus_result.aggregated_scores.values()),
            "discussion_rounds": consensus_result.total_rounds,
            "timestamp": datetime.now(),
        }

        self.active_patients[patient_id]["treatment_decisions"].append(decision)

        # 记录工作流程事件
        self._log_workflow_event(
            "mdt_discussion",
            patient_id,
            {"day": day, "consensus_result": consensus_result, "decision": decision},
        )

        return {
            "day": day,
            "patient_id": patient_id,
            "consensus_result": consensus_result,
            "decision": decision,
            "trigger_reason": "scheduled" if day % 7 == 0 else "critical_condition",
        }

    def _conduct_rl_training_episode(
        self, patient_id: str, mdt_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """进行RL训练episode"""
        # 从MDT结果创建训练数据
        patient_state_dict = {
            "patient_id": patient_id,
            "age": 65,  # 简化
            "quality_of_life_score": np.random.uniform(0.4, 0.8),
            "comorbidities": [],
        }

        # 重置RL环境
        state_vector = self.rl_environment.reset()

        # 从MDT推荐中选择动作
        recommended_treatment = mdt_result["decision"]["recommended_treatment"]
        action_map = {treatment: i for i, treatment in enumerate(TreatmentOption)}
        action = action_map.get(recommended_treatment, 0)

        # 执行动作并获取奖励
        next_state, reward, done, info = self.rl_environment.step(action)

        # 记录训练数据
        training_episode = {
            "day": mdt_result["day"],
            "patient_id": patient_id,
            "state_vector": state_vector.tolist(),
            "action": action,
            "recommended_treatment": recommended_treatment.value,
            "reward": reward,
            "info": info,
        }

        self.active_patients[patient_id]["rl_training_data"].append(training_episode)

        # 记录工作流程事件
        self._log_workflow_event("rl_training", patient_id, training_episode)

        return training_episode

    def _calculate_simulation_metrics(
        self, simulation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算模拟性能指标"""
        mdt_discussions = simulation_results["mdt_discussions"]
        rl_episodes = simulation_results["rl_training_episodes"]

        metrics = {
            "total_mdt_discussions": len(mdt_discussions),
            "total_rl_episodes": len(rl_episodes),
            "avg_consensus_score": 0.0,
            "avg_discussion_rounds": 0.0,
            "avg_rl_reward": 0.0,
            "convergence_rate": 0.0,
            "decision_consistency": 0.0,
        }

        if mdt_discussions:
            consensus_scores = [
                d["consensus_result"].aggregated_scores for d in mdt_discussions
            ]
            avg_scores = []
            for scores in consensus_scores:
                avg_scores.append(max(scores.values()))

            metrics["avg_consensus_score"] = np.mean(avg_scores)
            metrics["avg_discussion_rounds"] = np.mean(
                [d["consensus_result"].total_rounds for d in mdt_discussions]
            )
            metrics["convergence_rate"] = sum(
                [d["consensus_result"].convergence_achieved for d in mdt_discussions]
            ) / len(mdt_discussions)

            # 决策一致性 (相邻决策的相似度)
            if len(mdt_discussions) > 1:
                decisions = [
                    d["decision"]["recommended_treatment"] for d in mdt_discussions
                ]
                consistency_count = sum(
                    [
                        1
                        for i in range(1, len(decisions))
                        if decisions[i] == decisions[i - 1]
                    ]
                )
                metrics["decision_consistency"] = consistency_count / (
                    len(decisions) - 1
                )

        if rl_episodes:
            metrics["avg_rl_reward"] = np.mean([ep["reward"] for ep in rl_episodes])

        return metrics

    def _log_workflow_event(
        self, event_type: str, patient_id: str, data: Dict[str, Any]
    ) -> None:
        """记录工作流程事件"""
        event = WorkflowEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            patient_id=patient_id,
            data=data,
        )

        self.workflow_events.append(event)

        # 保持事件历史在合理范围内
        if len(self.workflow_events) > 10000:
            self.workflow_events = self.workflow_events[-8000:]  # 保留最近8000个事件

    def get_patient_workflow_summary(self, patient_id: str) -> Dict[str, Any]:
        """获取患者工作流程摘要"""
        if patient_id not in self.active_patients:
            return {}

        patient_info = self.active_patients[patient_id]
        patient_events = [
            event for event in self.workflow_events if event.patient_id == patient_id
        ]

        summary = {
            "patient_id": patient_id,
            "registration_time": patient_info["registration_time"],
            "current_status": patient_info["current_status"],
            "total_events": len(patient_events),
            "last_mdt_discussion": patient_info["last_mdt_discussion"],
            "total_decisions": len(patient_info["treatment_decisions"]),
            "total_rl_episodes": len(patient_info["rl_training_data"]),
            "event_timeline": [
                {
                    "timestamp": event.timestamp,
                    "type": event.event_type,
                    "summary": self._summarize_event_data(event.data),
                }
                for event in patient_events[-10:]  # 最近10个事件
            ],
        }

        return summary

    def _summarize_event_data(self, data: Dict[str, Any]) -> str:
        """总结事件数据"""
        if "day" in data:
            return f"Day {data['day']}"
        elif "decision" in data:
            decision = data["decision"]
            return f"Decision: {decision['recommended_treatment'].value} (score: {decision['consensus_score']:.2f})"
        elif "reward" in data:
            return f"RL reward: {data['reward']:.3f}"
        else:
            return "General event"

    def get_system_performance_overview(self) -> Dict[str, Any]:
        """获取系统整体性能概览"""
        total_patients = len(self.active_patients)
        total_events = len(self.workflow_events)

        # 按事件类型统计
        event_types = {}
        for event in self.workflow_events:
            event_type = event.event_type
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1

        # 计算平均性能指标
        all_decisions = []
        all_rl_rewards = []

        for patient_info in self.active_patients.values():
            all_decisions.extend(patient_info["treatment_decisions"])
            all_rl_rewards.extend(
                [ep["reward"] for ep in patient_info["rl_training_data"]]
            )

        overview = {
            "system_uptime": datetime.now(),
            "total_patients": total_patients,
            "total_workflow_events": total_events,
            "event_type_distribution": event_types,
            "total_treatment_decisions": len(all_decisions),
            "total_rl_episodes": len(all_rl_rewards),
            "avg_consensus_score": (
                np.mean([d["consensus_score"] for d in all_decisions])
                if all_decisions
                else 0
            ),
            "avg_rl_reward": np.mean(all_rl_rewards) if all_rl_rewards else 0,
            "active_patients": list(self.active_patients.keys()),
        }

        return overview

    def export_workflow_data(self, filepath: str) -> None:
        """导出工作流程数据"""
        import json

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_overview": self.get_system_performance_overview(),
            "active_patients": {
                pid: self.get_patient_workflow_summary(pid)
                for pid in self.active_patients.keys()
            },
            "workflow_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "patient_id": event.patient_id,
                    "data_summary": self._summarize_event_data(event.data),
                }
                for event in self.workflow_events[-1000:]  # 导出最近1000个事件
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Workflow data exported to {filepath}")
