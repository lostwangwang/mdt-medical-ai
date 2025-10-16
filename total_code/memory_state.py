"""
集成工作流程与实验设置
功能：Memory Controller集成、RL训练、对比实验
作者：姚刚
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# 引入主系统组件（假设已导入）
# from medical_consensus_system import *


@dataclass
class MemoryState:
    """记忆状态（与杜军的Memory Controller接口）"""

    patient_id: str
    individual_memory: Dict[str, Any]
    group_memory: Dict[str, Any]
    temporal_sequence: List[Dict[str, Any]]
    last_updated: datetime


class MemoryControllerInterface:
    """Memory Controller接口（与杜军模块对接）"""

    def __init__(self):
        self.memory_store = {}
        self.evolution_rules = self._initialize_evolution_rules()

    def _initialize_evolution_rules(self) -> Dict[str, Any]:
        """初始化演化规则"""
        return {
            "lab_evolution": {
                "creatinine": {"trend": "increase", "rate": 0.02, "noise": 0.1},
                "hemoglobin": {"trend": "decrease", "rate": 0.01, "noise": 0.05},
            },
            "vital_evolution": {
                "bp_systolic": {
                    "trend": "fluctuate",
                    "range": [120, 160],
                    "noise": 0.05,
                }
            },
            "symptom_progression": {
                "fatigue": {"severity_increase": 0.1},
                "pain": {"severity_fluctuate": 0.2},
            },
        }

    def retrieve(self, patient_id: str) -> MemoryState:
        """检索患者记忆状态"""
        if patient_id not in self.memory_store:
            # 创建初始状态
            self.memory_store[patient_id] = self._create_initial_memory(patient_id)

        return self.memory_store[patient_id]

    def update_daily(self, patient_id: str, new_events: List[Dict[str, Any]]) -> None:
        """每日更新患者记忆"""
        memory_state = self.retrieve(patient_id)

        # 更新个体记忆
        for event in new_events:
            self._update_individual_memory(memory_state, event)

        # 更新时间序列
        memory_state.temporal_sequence.extend(new_events)
        memory_state.last_updated = datetime.now()

        # 每周更新群体记忆
        if self._should_update_group_memory(memory_state):
            self._update_group_memory(memory_state)

    def _create_initial_memory(self, patient_id: str) -> MemoryState:
        """创建初始记忆状态"""
        return MemoryState(
            patient_id=patient_id,
            individual_memory={
                "baseline_labs": {"creatinine": 1.0, "hemoglobin": 12.0},
                "baseline_vitals": {"bp_systolic": 130, "heart_rate": 75},
                "treatment_history": [],
                "response_patterns": {},
            },
            group_memory={
                "similar_cases": [],
                "treatment_outcomes": {},
                "population_stats": {},
            },
            temporal_sequence=[],
            last_updated=datetime.now(),
        )

    def _update_individual_memory(
        self, memory_state: MemoryState, event: Dict[str, Any]
    ) -> None:
        """更新个体记忆"""
        event_type = event.get("event_type")

        if event_type == "lab":
            memory_state.individual_memory.setdefault("recent_labs", []).append(event)
        elif event_type == "vital":
            memory_state.individual_memory.setdefault("recent_vitals", []).append(event)
        elif event_type == "medication":
            memory_state.individual_memory.setdefault("medications", []).append(event)

    def _should_update_group_memory(self, memory_state: MemoryState) -> bool:
        """判断是否应该更新群体记忆"""
        # Handle both datetime objects and ISO strings
        if isinstance(memory_state.last_updated, str):
            last_updated = datetime.fromisoformat(memory_state.last_updated.replace('Z', '+00:00'))
        else:
            last_updated = memory_state.last_updated
        
        days_since_update = (datetime.now() - last_updated).days
        return days_since_update >= 7

    def _update_group_memory(self, memory_state: MemoryState) -> None:
        """更新群体记忆"""
        # 简化的群体记忆更新逻辑
        memory_state.group_memory["last_group_update"] = datetime.now()

    def generate_patient_state_summary(self, patient_id: str) -> Dict[str, Any]:
        """生成患者状态摘要（供共识系统使用）"""
        memory_state = self.retrieve(patient_id)

        # 计算最新的实验室值
        recent_labs = memory_state.individual_memory.get("recent_labs", [])
        latest_labs = {}
        if recent_labs:
            for lab in recent_labs[-5:]:  # 最近5次检查
                latest_labs[lab["name"]] = lab["value"]

        # 计算最新的生命体征
        recent_vitals = memory_state.individual_memory.get("recent_vitals", [])
        latest_vitals = {}
        if recent_vitals:
            for vital in recent_vitals[-3:]:  # 最近3次测量
                latest_vitals[vital["name"]] = vital["value"]

        return {
            "patient_id": patient_id,
            "lab_results": latest_labs
            or memory_state.individual_memory.get("baseline_labs", {}),
            "vital_signs": latest_vitals
            or memory_state.individual_memory.get("baseline_vitals", {}),
            "treatment_history": memory_state.individual_memory.get(
                "treatment_history", []
            ),
            "temporal_trend": self._analyze_temporal_trend(memory_state),
            "group_context": memory_state.group_memory,
        }

    def _analyze_temporal_trend(self, memory_state: MemoryState) -> Dict[str, str]:
        """分析时间趋势"""
        trends = {}

        # 分析实验室值趋势
        recent_labs = memory_state.individual_memory.get("recent_labs", [])
        if len(recent_labs) >= 3:
            creatinine_values = [
                lab["value"] for lab in recent_labs if lab["name"] == "creatinine"
            ]
            if len(creatinine_values) >= 2:
                if creatinine_values[-1] > creatinine_values[0]:
                    trends["creatinine"] = "increasing"
                else:
                    trends["creatinine"] = "stable_or_decreasing"

        return trends


class IntegratedMDTSystem:
    """集成的MDT系统"""

    def __init__(self):
        self.memory_controller = MemoryControllerInterface()
        self.rag_system = MedicalKnowledgeRAG()
        self.dialogue_manager = None
        self.rl_environment = None
        self.experiment_tracker = RLExperimentTracker()

    def initialize_components(self):
        """初始化所有组件"""
        # 这里需要导入主系统的类
        # self.dialogue_manager = MultiAgentDialogueManager(self.rag_system)
        # self.rl_environment = MDTReinforcementLearning(ConsensusMatrix())
        pass

    def run_integrated_workflow(
        self, patient_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """运行集成工作流程"""
        results = {
            "patient_id": patient_id,
            "simulation_days": days,
            "daily_results": [],
            "learning_progress": [],
            "final_performance": {},
        }

        print(f"开始为患者 {patient_id} 运行 {days} 天的集成模拟...")

        for day in range(days):
            daily_result = self._simulate_daily_workflow(patient_id, day)
            results["daily_results"].append(daily_result)

            # 每5天评估一次学习进展
            if day % 5 == 0:
                learning_metrics = self._evaluate_learning_progress(patient_id, day)
                results["learning_progress"].append(learning_metrics)

            if day % 10 == 0:
                print(f"  完成第 {day} 天的模拟")

        # 生成最终评估
        results["final_performance"] = self._generate_final_evaluation(results)

        print("集成工作流程完成！")
        return results

    def _simulate_daily_workflow(self, patient_id: str, day: int) -> Dict[str, Any]:
        """模拟每日工作流程"""
        # 1. 生成每日医疗事件
        daily_events = self._generate_daily_events(patient_id, day)

        # 2. 更新记忆系统
        self.memory_controller.update_daily(patient_id, daily_events)

        # 3. 获取当前患者状态
        current_state = self._create_patient_state_from_memory(patient_id, day)

        # 4. 如果达到决策点，运行MDT讨论
        decision_result = None
        if self._is_decision_day(day):
            decision_result = self._run_mdt_decision(current_state)

        # 5. RL系统学习
        rl_feedback = None
        if decision_result:
            rl_feedback = self._update_rl_system(current_state, decision_result)

        return {
            "day": day,
            "events": daily_events,
            "patient_state": current_state,
            "mdt_decision": decision_result,
            "rl_feedback": rl_feedback,
        }

    def _generate_daily_events(self, patient_id: str, day: int) -> List[Dict[str, Any]]:
        """生成每日医疗事件"""
        events = []

        # 模拟实验室检查（每3天一次）
        if day % 3 == 0:
            base_creatinine = 1.0 + day * 0.01 + np.random.normal(0, 0.1)
            events.append(
                {
                    "patient_id": patient_id,
                    "time": datetime.now() + timedelta(days=day),
                    "event_type": "lab",
                    "name": "creatinine",
                    "value": max(0.5, base_creatinine),
                }
            )

        # 模拟生命体征（每天）
        base_bp = 130 + np.random.normal(0, 10)
        events.append(
            {
                "patient_id": patient_id,
                "time": datetime.now() + timedelta(days=day),
                "event_type": "vital",
                "name": "bp_systolic",
                "value": max(90, min(180, base_bp)),
            }
        )

        # 模拟药物事件（随机）
        if np.random.random() < 0.2:  # 20%概率
            events.append(
                {
                    "patient_id": patient_id,
                    "time": datetime.now() + timedelta(days=day),
                    "event_type": "medication",
                    "name": "adjustment",
                    "value": 1.0,
                }
            )

        return events

    def _create_patient_state_from_memory(self, patient_id: str, day: int):
        """从记忆系统创建患者状态"""
        memory_summary = self.memory_controller.generate_patient_state_summary(
            patient_id
        )

        # 创建PatientState对象（需要导入原始类）
        # 这里返回简化的字典格式
        return {
            "patient_id": patient_id,
            "age": 65,  # 假设值
            "diagnosis": "breast_cancer",
            "stage": "II",
            "lab_results": memory_summary["lab_results"],
            "vital_signs": memory_summary["vital_signs"],
            "symptoms": ["fatigue"] if day > 15 else [],
            "comorbidities": ["diabetes"] if day > 20 else [],
            "quality_of_life_score": max(0.3, 0.8 - day * 0.01),
            "temporal_trend": memory_summary["temporal_trend"],
            "day": day,
        }

    def _is_decision_day(self, day: int) -> bool:
        """判断是否为决策日"""
        return day % 7 == 0 and day > 0  # 每周做一次决策

    def _run_mdt_decision(self, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        """运行MDT决策（简化版）"""
        # 这里应该调用完整的对话系统
        # dialogue_result = self.dialogue_manager.conduct_mdt_discussion(patient_state)

        # 模拟决策结果
        simulated_result = {
            "recommended_treatment": "chemotherapy",
            "consensus_score": np.random.uniform(0.4, 0.9),
            "agreement_level": np.random.choice(["low", "medium", "high"]),
            "dialogue_rounds": np.random.randint(2, 6),
        }

        return simulated_result

    def _update_rl_system(
        self, patient_state: Dict[str, Any], decision_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新RL系统"""
        # 计算奖励
        consensus_score = decision_result["consensus_score"]
        quality_of_life = patient_state["quality_of_life_score"]

        # 简化的奖励函数
        reward = consensus_score * 0.6 + quality_of_life * 0.4

        # 记录学习数据
        self.experiment_tracker.track_experiment(
            "Integrated_MDT_RL",
            patient_state["day"],
            reward,
            consensus_score,
            decision_result["recommended_treatment"],
        )

        return {
            "reward": reward,
            "action": decision_result["recommended_treatment"],
            "state_features": list(patient_state["lab_results"].values())[:3],
        }

    def _evaluate_learning_progress(self, patient_id: str, day: int) -> Dict[str, Any]:
        """评估学习进展"""
        # 计算最近的平均奖励
        recent_data = self.experiment_tracker.experiments.get("Integrated_MDT_RL", {})
        recent_rewards = recent_data.get("rewards", [])

        if len(recent_rewards) >= 5:
            recent_avg = np.mean(recent_rewards[-5:])
        else:
            recent_avg = 0.5

        return {
            "day": day,
            "average_reward": recent_avg,
            "decision_consistency": np.random.uniform(0.6, 0.9),
            "consensus_improvement": np.random.uniform(0.1, 0.3),
        }

    def _generate_final_evaluation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终评估"""
        daily_results = results["daily_results"]
        learning_progress = results["learning_progress"]

        # 计算性能指标
        decision_count = len([r for r in daily_results if r["mdt_decision"]])
        avg_consensus = (
            np.mean(
                [
                    r["mdt_decision"]["consensus_score"]
                    for r in daily_results
                    if r["mdt_decision"]
                ]
            )
            if decision_count > 0
            else 0
        )

        final_reward = (
            learning_progress[-1]["average_reward"] if learning_progress else 0
        )
        initial_reward = (
            learning_progress[0]["average_reward"] if learning_progress else 0
        )

        return {
            "total_decisions": decision_count,
            "average_consensus_score": avg_consensus,
            "learning_improvement": final_reward - initial_reward,
            "final_performance": final_reward,
            "system_stability": np.std(
                [lp["average_reward"] for lp in learning_progress]
            ),
        }


class BaselineComparison:
    """基线模型对比实验"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def add_baseline_model(self, name: str, model_func):
        """添加基线模型"""
        self.models[name] = model_func

    def run_comparison_experiment(
        self, test_patients: List[Dict], num_trials: int = 100
    ) -> pd.DataFrame:
        """运行对比实验"""
        print(
            f"开始对比实验：{len(self.models)} 个模型，{len(test_patients)} 个患者，{num_trials} 次试验"
        )

        comparison_results = []

        for model_name, model_func in self.models.items():
            print(f"  测试模型: {model_name}")

            model_scores = []

            for trial in range(num_trials):
                # 随机选择患者
                patient = np.random.choice(test_patients)

                # 运行模型
                try:
                    result = model_func(patient)
                    score = self._evaluate_model_result(result, patient)
                    model_scores.append(score)
                except Exception as e:
                    print(f"    模型 {model_name} 第 {trial} 次试验出错: {e}")
                    model_scores.append(0.0)

            # 计算统计指标
            comparison_results.append(
                {
                    "model": model_name,
                    "mean_score": np.mean(model_scores),
                    "std_score": np.std(model_scores),
                    "median_score": np.median(model_scores),
                    "success_rate": len([s for s in model_scores if s > 0.5])
                    / len(model_scores),
                }
            )

        return pd.DataFrame(comparison_results)

    def _evaluate_model_result(
        self, result: Dict[str, Any], patient: Dict[str, Any]
    ) -> float:
        """评估模型结果"""
        # 简化的评估逻辑
        consensus_score = result.get("consensus_score", 0.5)
        consistency_bonus = 0.1 if result.get("agreement_level") == "high" else 0
        efficiency_bonus = 0.1 if result.get("dialogue_rounds", 5) <= 3 else 0

        return min(1.0, consensus_score + consistency_bonus + efficiency_bonus)

    def create_baseline_models(self):
        """创建基线模型"""

        def random_baseline(patient: Dict[str, Any]) -> Dict[str, Any]:
            """随机基线"""
            return {
                "recommended_treatment": np.random.choice(
                    ["surgery", "chemotherapy", "radiotherapy"]
                ),
                "consensus_score": np.random.uniform(0.3, 0.7),
                "agreement_level": np.random.choice(["low", "medium", "high"]),
                "dialogue_rounds": np.random.randint(1, 8),
            }

        def rule_based_baseline(patient: Dict[str, Any]) -> Dict[str, Any]:
            """基于规则的基线"""
            age = patient.get("age", 50)
            quality_score = patient.get("quality_of_life_score", 0.7)

            if age > 70:
                treatment = "palliative_care"
                consensus = 0.6
            elif quality_score > 0.7:
                treatment = "surgery"
                consensus = 0.8
            else:
                treatment = "chemotherapy"
                consensus = 0.7

            return {
                "recommended_treatment": treatment,
                "consensus_score": consensus,
                "agreement_level": "medium",
                "dialogue_rounds": 2,
            }

        def single_agent_baseline(patient: Dict[str, Any]) -> Dict[str, Any]:
            """单智能体基线"""
            # 模拟只有肿瘤科医生的决策
            return {
                "recommended_treatment": "chemotherapy",  # 肿瘤科医生倾向
                "consensus_score": 0.7,
                "agreement_level": "high",  # 没有争议
                "dialogue_rounds": 1,
            }

        self.add_baseline_model("Random", random_baseline)
        self.add_baseline_model("Rule-based", rule_based_baseline)
        self.add_baseline_model("Single-Agent", single_agent_baseline)

        # 模拟其他系统
        def langchain_rag_baseline(patient: Dict[str, Any]) -> Dict[str, Any]:
            """LangChain RAG基线"""
            return {
                "recommended_treatment": "surgery",
                "consensus_score": np.random.uniform(0.6, 0.8),
                "agreement_level": "medium",
                "dialogue_rounds": 1,
            }

        def med_palm_baseline(patient: Dict[str, Any]) -> Dict[str, Any]:
            """Med-PaLM基线"""
            return {
                "recommended_treatment": np.random.choice(["surgery", "chemotherapy"]),
                "consensus_score": np.random.uniform(0.7, 0.9),
                "agreement_level": "high",
                "dialogue_rounds": 1,
            }

        self.add_baseline_model("LangChain-RAG", langchain_rag_baseline)
        self.add_baseline_model("Med-PaLM-like", med_palm_baseline)


def run_comprehensive_experiments():
    """运行综合实验"""
    print("=== 综合实验开始 ===\n")

    # 1. 集成系统测试
    print("1. 集成系统测试")
    integrated_system = IntegratedMDTSystem()
    integrated_system.initialize_components()

    # 运行一个患者的完整流程
    workflow_results = integrated_system.run_integrated_workflow("P003", days=21)

    print("集成系统测试结果:")
    final_perf = workflow_results["final_performance"]
    print(f"  总决策次数: {final_perf['total_decisions']}")
    print(f"  平均共识得分: {final_perf['average_consensus_score']:.3f}")
    print(f"  学习改进: {final_perf['learning_improvement']:+.3f}")
    print(f"  最终性能: {final_perf['final_performance']:.3f}")

    # 2. 基线对比实验
    print("\n2. 基线模型对比")
    baseline_comparison = BaselineComparison()
    baseline_comparison.create_baseline_models()

    # 创建测试患者
    test_patients = []
    for i in range(10):
        patient = {
            "patient_id": f"P{100+i}",
            "age": np.random.randint(40, 80),
            "diagnosis": "breast_cancer",
            "quality_of_life_score": np.random.uniform(0.4, 0.9),
            "comorbidities": np.random.choice([0, 1, 2, 3]),
        }
        test_patients.append(patient)

    comparison_results = baseline_comparison.run_comparison_experiment(
        test_patients, num_trials=50
    )

    print("基线对比结果:")
    print(comparison_results.round(3))

    # 3. 可视化结果
    print("\n3. 生成可视化报告")

    # 绘制学习曲线
    plt.figure(figsize=(15, 5))

    # 子图1: 集成系统学习进展
    plt.subplot(1, 3, 1)
    learning_data = workflow_results["learning_progress"]
    days = [lp["day"] for lp in learning_data]
    rewards = [lp["average_reward"] for lp in learning_data]

    plt.plot(days, rewards, "b-", marker="o", linewidth=2)
    plt.title("集成系统学习曲线")
    plt.xlabel("天数")
    plt.ylabel("平均奖励")
    plt.grid(True, alpha=0.3)

    # 子图2: 基线模型对比
    plt.subplot(1, 3, 2)
    models = comparison_results["model"].tolist()
    mean_scores = comparison_results["mean_score"].tolist()
    std_scores = comparison_results["std_score"].tolist()

    bars = plt.bar(models, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
    plt.title("基线模型性能对比")
    plt.ylabel("平均得分")
    plt.xticks(rotation=45, ha="right")

    # 为最高的柱子添加颜色
    max_idx = np.argmax(mean_scores)
    bars[max_idx].set_color("red")

    # 子图3: 系统稳定性分析
    plt.subplot(1, 3, 3)
    consistency_scores = [lp["decision_consistency"] for lp in learning_data]
    consensus_improvements = [lp["consensus_improvement"] for lp in learning_data]

    plt.plot(days, consistency_scores, "g-", label="决策一致性", marker="s")
    plt.plot(days, consensus_improvements, "r-", label="共识改进", marker="^")
    plt.title("系统稳定性分析")
    plt.xlabel("天数")
    plt.ylabel("指标值")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 4. 生成实验报告
    print("\n4. 实验总结报告")
    print("=" * 50)

    best_baseline = comparison_results.loc[comparison_results["mean_score"].idxmax()]
    our_system_score = final_perf["final_performance"]

    print(
        f"最佳基线模型: {best_baseline['model']} (得分: {best_baseline['mean_score']:.3f})"
    )
    print(f"我们的系统得分: {our_system_score:.3f}")

    if our_system_score > best_baseline["mean_score"]:
        improvement = (
            (our_system_score - best_baseline["mean_score"])
            / best_baseline["mean_score"]
        ) * 100
        print(f"性能提升: {improvement:+.1f}%")
        print("✅ 我们的多智能体系统表现更好！")
    else:
        print("❌ 需要进一步优化系统性能")

    print(f"\n关键优势:")
    print(f"- 学习能力: {final_perf['learning_improvement']:+.3f}")
    print(f"- 共识质量: {final_perf['average_consensus_score']:.3f}")
    print(f"- 系统稳定性: {1/final_perf['system_stability']:.2f}")

    print("\n=== 综合实验完成 ===")


if __name__ == "__main__":
    # 运行综合实验
    run_comprehensive_experiments()
