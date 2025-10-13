"""
医疗决策强化学习环境
文件路径: src/rl/rl_environment.py
作者: 姚刚
功能: 实现MDT医疗决策的强化学习环境和奖励机制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import gymnasium as gym
from gymnasium import spaces

from ..core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    ConsensusResult,
    RLState,
    RLAction,
    RLReward,
)

logger = logging.getLogger(__name__)


class MDTReinforcementLearning(gym.Env):
    """医疗决策强化学习环境"""

    def __init__(self, consensus_system=None):
        super(MDTReinforcementLearning, self).__init__()

        self.consensus_system = consensus_system
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.episode_count = 0

        # 定义动作空间 (治疗方案选择)
        self.action_space = spaces.Discrete(len(TreatmentOption))

        # 定义状态空间 (患者特征 + 共识特征)
        # 假设: 3个患者特征 + 4个共识特征 + 6个治疗偏好
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # 奖励权重配置
        self.reward_weights = {
            "consensus_score": 0.4,
            "consistency_bonus": 0.2,
            "conflict_penalty": 0.2,
            "patient_suitability": 0.2,
        }

        self.current_patient_state = None
        self.current_consensus_result = None

    def reset(self, patient_state: PatientState = None) -> np.ndarray:
        """重置环境"""
        if patient_state:
            self.current_patient_state = patient_state
        else:
            # 生成随机患者状态用于训练
            self.current_patient_state = self._generate_random_patient()

        # 生成当前状态的共识结果
        if self.consensus_system:
            self.current_consensus_result = self.consensus_system.generate_consensus(
                self.current_patient_state
            )
        else:
            self.current_consensus_result = self._generate_mock_consensus()

        # 创建状态向量
        state_vector = self.create_state_vector(
            self.current_patient_state, self.current_consensus_result
        )

        logger.debug(
            f"Environment reset for patient {self.current_patient_state.patient_id}"
        )

        return state_vector

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一个步骤"""
        # 将动作索引转换为治疗方案
        treatment_options = list(TreatmentOption)
        selected_treatment = treatment_options[action]

        # 计算奖励
        reward_info = self.calculate_reward(
            selected_treatment,
            self.current_consensus_result,
            self.current_patient_state,
        )

        # 记录历史
        self.action_history.append(selected_treatment)
        self.reward_history.append(reward_info.total_reward)

        # 生成下一个状态 (在实际应用中可能是时序演化的状态)
        next_state_vector = self.create_state_vector(
            self.current_patient_state, self.current_consensus_result
        )

        # 判断是否结束 (单步决策，每步都结束)
        done = True

        # 额外信息
        info = {
            "selected_treatment": selected_treatment.value,
            "reward_breakdown": {
                "consensus_score": reward_info.consensus_score,
                "consistency_bonus": reward_info.consistency_bonus,
                "conflict_penalty": reward_info.conflict_penalty,
                "patient_suitability": reward_info.patient_suitability,
            },
            "patient_id": self.current_patient_state.patient_id,
        }

        logger.debug(
            f"Action: {selected_treatment.value}, Reward: {reward_info.total_reward:.3f}"
        )

        return next_state_vector, reward_info.total_reward, done, info

    def create_state_vector(
        self, patient_state: PatientState, consensus_result: ConsensusResult
    ) -> np.ndarray:
        """创建状态向量"""
        # 患者特征 (3维)
        patient_features = [
            patient_state.age / 100.0,  # 归一化年龄
            len(patient_state.comorbidities) / 5.0,  # 归一化并发症数量
            patient_state.quality_of_life_score,  # 生活质量评分
        ]

        # 共识特征 (4维)
        aggregated_scores = list(consensus_result.aggregated_scores.values())
        consensus_features = [
            np.mean(aggregated_scores),  # 平均共识得分
            np.std(aggregated_scores),  # 共识变异度
            len(consensus_result.conflicts) / len(TreatmentOption),  # 冲突比例
            len(consensus_result.agreements) / len(TreatmentOption),  # 一致比例
        ]

        # 治疗偏好特征 (6维)
        treatment_preferences = []
        for treatment in TreatmentOption:
            score = consensus_result.aggregated_scores.get(treatment, 0.0)
            treatment_preferences.append(score)

        # 组合所有特征
        state_vector = np.array(
            patient_features + consensus_features + treatment_preferences,
            dtype=np.float32,
        )

        # 确保向量在合理范围内
        state_vector = np.clip(state_vector, -1.0, 1.0)

        return state_vector

    def calculate_reward(
        self,
        action: TreatmentOption,
        consensus_result: ConsensusResult,
        patient_state: PatientState,
    ) -> RLReward:
        """计算奖励函数"""

        # 1. 基础奖励：共识得分
        consensus_score = consensus_result.aggregated_scores.get(action, 0.0)

        # 2. 一致性奖励
        consensus_bonus = 0.0
        for agreement in consensus_result.agreements:
            if agreement["treatment"] == action:
                consensus_bonus = (
                    agreement["consensus_score"] * agreement["agreement_strength"]
                )
                break

        # 3. 冲突惩罚
        conflict_penalty = 0.0
        for conflict in consensus_result.conflicts:
            if conflict["treatment"] == action:
                conflict_penalty = conflict["variance"]
                break

        # 4. 患者适应性奖励
        patient_suitability = self._calculate_patient_suitability(action, patient_state)

        # 5. 综合奖励计算
        total_reward = (
            consensus_score * self.reward_weights["consensus_score"]
            + consensus_bonus * self.reward_weights["consistency_bonus"]
            - conflict_penalty * self.reward_weights["conflict_penalty"]
            + patient_suitability * self.reward_weights["patient_suitability"]
        )

        return RLReward(
            consensus_score=consensus_score,
            consistency_bonus=consensus_bonus,
            conflict_penalty=conflict_penalty,
            patient_suitability=patient_suitability,
            total_reward=total_reward,
        )

    def _calculate_patient_suitability(
        self, action: TreatmentOption, patient_state: PatientState
    ) -> float:
        """计算治疗方案对患者的适应性"""
        suitability_scores = {
            TreatmentOption.SURGERY: self._surgery_suitability(patient_state),
            TreatmentOption.CHEMOTHERAPY: self._chemotherapy_suitability(patient_state),
            TreatmentOption.RADIOTHERAPY: self._radiotherapy_suitability(patient_state),
            TreatmentOption.IMMUNOTHERAPY: self._immunotherapy_suitability(
                patient_state
            ),
            TreatmentOption.PALLIATIVE_CARE: self._palliative_suitability(
                patient_state
            ),
            TreatmentOption.WATCHFUL_WAITING: self._watchful_waiting_suitability(
                patient_state
            ),
        }

        return suitability_scores.get(action, 0.5)

    def _surgery_suitability(self, patient_state: PatientState) -> float:
        """手术适应性评估"""
        base_score = 0.7

        # 年龄因素
        if patient_state.age > 75:
            base_score *= 0.7
        elif patient_state.age < 50:
            base_score *= 1.1

        # 并发症因素
        if len(patient_state.comorbidities) > 2:
            base_score *= 0.6

        # 生活质量因素
        if patient_state.quality_of_life_score < 0.4:
            base_score *= 0.8

        return np.clip(base_score, 0.0, 1.0)

    def _chemotherapy_suitability(self, patient_state: PatientState) -> float:
        """化疗适应性评估"""
        base_score = 0.6

        # 年龄因素
        if patient_state.age > 70:
            base_score *= 0.8

        # 生活质量因素 (化疗对生活质量影响较大)
        base_score *= patient_state.quality_of_life_score

        # 并发症因素
        if "cardiac_dysfunction" in patient_state.comorbidities:
            base_score *= 0.5

        return np.clip(base_score, 0.0, 1.0)

    def _radiotherapy_suitability(self, patient_state: PatientState) -> float:
        """放疗适应性评估"""
        base_score = 0.65

        # 年龄对放疗影响相对较小
        if patient_state.age > 80:
            base_score *= 0.9

        # 生活质量因素
        if patient_state.quality_of_life_score > 0.6:
            base_score *= 1.1

        return np.clip(base_score, 0.0, 1.0)

    def _immunotherapy_suitability(self, patient_state: PatientState) -> float:
        """免疫疗法适应性评估"""
        base_score = 0.5  # 相对较新的治疗方法

        # 免疫疗法对年轻患者可能更有效
        if patient_state.age < 65:
            base_score *= 1.2

        # 生活质量因素
        base_score *= 0.5 + patient_state.quality_of_life_score * 0.5

        return np.clip(base_score, 0.0, 1.0)

    def _palliative_suitability(self, patient_state: PatientState) -> float:
        """姑息治疗适应性评估"""
        base_score = 0.4

        # 对于生活质量很差或高龄患者，姑息治疗适应性提高
        if patient_state.quality_of_life_score < 0.4:
            base_score += 0.4

        if patient_state.age > 80:
            base_score += 0.3

        if len(patient_state.comorbidities) > 3:
            base_score += 0.2

        return np.clip(base_score, 0.0, 1.0)

    def _watchful_waiting_suitability(self, patient_state: PatientState) -> float:
        """观察等待适应性评估"""
        base_score = 0.3

        # 对于早期疾病和高龄患者可能更合适
        if patient_state.stage in ["I", "0"]:
            base_score += 0.3

        if patient_state.age > 75:
            base_score += 0.2

        # 生活质量好的患者可能更适合观察等待
        if patient_state.quality_of_life_score > 0.7:
            base_score += 0.2

        return np.clip(base_score, 0.0, 1.0)

    def _generate_random_patient(self) -> PatientState:
        """生成随机患者状态用于训练"""
        return PatientState(
            patient_id=f"TRAIN_{self.episode_count}",
            age=np.random.randint(40, 85),
            diagnosis="breast_cancer",
            stage=np.random.choice(["I", "II", "III", "IV"]),
            lab_results={
                "creatinine": np.random.uniform(0.8, 2.0),
                "hemoglobin": np.random.uniform(9.0, 15.0),
            },
            vital_signs={
                "bp_systolic": np.random.randint(110, 170),
                "heart_rate": np.random.randint(60, 100),
            },
            symptoms=np.random.choice(
                [["fatigue"], ["pain"], ["fatigue", "pain"], []], 1
            )[0],
            comorbidities=np.random.choice(
                [[], ["diabetes"], ["hypertension"], ["diabetes", "hypertension"]]
            ),
            psychological_status=np.random.choice(["stable", "anxious", "depressed"]),
            quality_of_life_score=np.random.uniform(0.3, 0.9),
            timestamp=datetime.now(),
        )

    def _generate_mock_consensus(self) -> ConsensusResult:
        """生成模拟共识结果"""
        # 为训练目的生成模拟的共识结果
        aggregated_scores = {}
        for treatment in TreatmentOption:
            aggregated_scores[treatment] = np.random.uniform(-0.5, 0.8)

        # 简化的冲突和一致意见生成
        conflicts = []
        agreements = []

        # 随机生成一些冲突
        if np.random.random() < 0.3:  # 30%概率有冲突
            conflicted_treatment = np.random.choice(list(TreatmentOption))
            conflicts.append(
                {
                    "treatment": conflicted_treatment,
                    "variance": np.random.uniform(0.5, 1.0),
                    "conflicting_roles": ["oncologist", "nurse"],
                }
            )

        # 随机生成一些一致意见
        if np.random.random() < 0.6:  # 60%概率有一致意见
            agreed_treatment = np.random.choice(list(TreatmentOption))
            agreements.append(
                {
                    "treatment": agreed_treatment,
                    "consensus_score": np.random.uniform(0.4, 0.9),
                    "agreement_strength": np.random.uniform(0.7, 1.0),
                }
            )

        return ConsensusResult(
            consensus_matrix=pd.DataFrame(),  # 简化
            role_opinions={},  # 简化
            aggregated_scores=aggregated_scores,
            conflicts=conflicts,
            agreements=agreements,
            dialogue_summary=None,
            timestamp=datetime.now(),
        )

    def get_training_metrics(self) -> Dict[str, Any]:
        """获取训练指标"""
        if not self.reward_history:
            return {}

        recent_rewards = self.reward_history[-100:]  # 最近100个episode

        metrics = {
            "total_episodes": len(self.reward_history),
            "average_reward": np.mean(self.reward_history),
            "recent_average_reward": np.mean(recent_rewards),
            "reward_std": np.std(self.reward_history),
            "max_reward": np.max(self.reward_history),
            "min_reward": np.min(self.reward_history),
            "improvement": self._calculate_improvement(),
            "convergence_indicator": self._assess_convergence(),
        }

        return metrics

    def _calculate_improvement(self) -> float:
        """计算学习改进"""
        if len(self.reward_history) < 20:
            return 0.0

        early_rewards = self.reward_history[:10]
        recent_rewards = self.reward_history[-10:]

        return np.mean(recent_rewards) - np.mean(early_rewards)

    def _assess_convergence(self) -> float:
        """评估收敛程度"""
        if len(self.reward_history) < 50:
            return 0.0

        recent_rewards = self.reward_history[-50:]
        variance = np.var(recent_rewards)

        # 方差越小，收敛程度越高
        convergence = max(0.0, 1.0 - variance * 2)

        return convergence

    def reset_training_history(self):
        """重置训练历史"""
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.episode_count = 0
        logger.info("Training history reset")

    def render(self, mode="human"):
        """环境渲染"""
        if mode == "human":
            print(f"Episode: {self.episode_count}")
            print(f"Current Patient: {self.current_patient_state.patient_id}")
            print(f"Recent Rewards: {self.reward_history[-5:]}")


class RLTrainer:
    """强化学习训练器"""

    def __init__(self, environment: MDTReinforcementLearning):
        self.env = environment
        self.training_history = []

    def train_dqn(
        self, episodes: int = 1000, learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """使用DQN算法训练"""
        # 这里应该实现真正的DQN训练
        # 为了演示，我们使用简化的随机策略

        logger.info(f"Starting DQN training for {episodes} episodes")

        episode_rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0

            # 简化的随机策略 (在真实实现中会使用神经网络)
            action = np.random.randint(0, len(TreatmentOption))

            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

            episode_rewards.append(total_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode}, Average Reward: {avg_reward:.3f}")

        training_results = {
            "episodes": episodes,
            "final_average_reward": np.mean(episode_rewards[-100:]),
            "best_reward": np.max(episode_rewards),
            "learning_curve": episode_rewards,
            "final_metrics": self.env.get_training_metrics(),
        }

        logger.info("DQN training completed")

        return training_results

    def evaluate_policy(self, num_episodes: int = 100) -> Dict[str, Any]:
        """评估策略性能"""
        evaluation_rewards = []
        treatment_selections = []

        for _ in range(num_episodes):
            state = self.env.reset()

            # 使用贪婪策略选择最佳动作
            action = self._select_best_action(state)

            next_state, reward, done, info = self.env.step(action)

            evaluation_rewards.append(reward)
            treatment_selections.append(info["selected_treatment"])

        evaluation_results = {
            "average_reward": np.mean(evaluation_rewards),
            "reward_std": np.std(evaluation_rewards),
            "treatment_distribution": self._analyze_treatment_distribution(
                treatment_selections
            ),
            "performance_consistency": 1.0 - np.std(evaluation_rewards),
        }

        return evaluation_results

    def _select_best_action(self, state: np.ndarray) -> int:
        """选择最佳动作 (简化版)"""
        # 在真实实现中，这里会使用训练好的神经网络
        # 现在使用简化的启发式规则

        # 基于状态特征选择动作
        patient_age_norm = state[0]
        quality_of_life = state[2]
        avg_consensus = state[3]

        if patient_age_norm > 0.8 and quality_of_life < 0.4:
            return TreatmentOption.PALLIATIVE_CARE.value
        elif avg_consensus > 0.5:
            return TreatmentOption.SURGERY.value
        else:
            return TreatmentOption.CHEMOTHERAPY.value

    def _analyze_treatment_distribution(
        self, treatments: List[str]
    ) -> Dict[str, float]:
        """分析治疗方案选择分布"""
        from collections import Counter

        distribution = Counter(treatments)
        total = len(treatments)

        return {treatment: count / total for treatment, count in distribution.items()}
