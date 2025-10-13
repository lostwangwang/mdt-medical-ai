"""
可视化与实验对比模块
功能：共识矩阵可视化、RL学习曲线、模型对比实验
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ConsensusVisualizer:
    """共识矩阵可视化工具"""

    @staticmethod
    def plot_consensus_heatmap(
        consensus_matrix: pd.DataFrame, title: str = "医疗团队治疗方案共识矩阵"
    ) -> None:
        """绘制共识矩阵热力图"""
        plt.figure(figsize=(12, 8))

        # 创建热力图
        sns.heatmap(
            consensus_matrix,
            annot=True,
            cmap="RdYlGn",
            center=0,
            fmt=".2f",
            cbar_kws={"label": "支持度 (-1: 强烈反对, +1: 强烈支持)"},
        )

        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("医疗团队角色", fontsize=12)
        plt.ylabel("治疗方案", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_consensus_radar(
        aggregated_scores: Dict[str, float], title: str = "治疗方案综合评分"
    ) -> None:
        """绘制治疗方案雷达图"""
        treatments = list(aggregated_scores.keys())
        scores = list(aggregated_scores.values())

        # 准备雷达图数据
        angles = np.linspace(0, 2 * np.pi, len(treatments), endpoint=False).tolist()
        scores += scores[:1]  # 闭合雷达图
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # 绘制雷达图
        ax.plot(angles, scores, "o-", linewidth=2, label="共识评分")
        ax.fill(angles, scores, alpha=0.25)

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(treatments, fontsize=10)
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"])
        ax.grid(True)

        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_role_disagreement(conflicts: List[Dict[str, Any]]) -> None:
        """绘制角色间分歧分析"""
        if not conflicts:
            print("没有发现显著的角色间分歧")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 分歧治疗方案统计
        treatments = [conflict["treatment"] for conflict in conflicts]
        variances = [conflict["variance"] for conflict in conflicts]

        ax1.bar(treatments, variances, color="coral")
        ax1.set_title("治疗方案分歧程度", fontsize=14, fontweight="bold")
        ax1.set_xlabel("治疗方案")
        ax1.set_ylabel("分歧程度（方差）")
        ax1.tick_params(axis="x", rotation=45)

        # 分歧范围分析
        min_scores = [conflict["min_score"] for conflict in conflicts]
        max_scores = [conflict["max_score"] for conflict in conflicts]

        x = np.arange(len(treatments))
        width = 0.35

        ax2.bar(x - width / 2, min_scores, width, label="最低评分", color="lightcoral")
        ax2.bar(x + width / 2, max_scores, width, label="最高评分", color="lightgreen")

        ax2.set_title("分歧评分范围", fontsize=14, fontweight="bold")
        ax2.set_xlabel("治疗方案")
        ax2.set_ylabel("评分范围")
        ax2.set_xticks(x)
        ax2.set_xticklabels(treatments, rotation=45)
        ax2.legend()

        plt.tight_layout()
        plt.show()


class RLExperimentTracker:
    """强化学习实验追踪器"""

    def __init__(self):
        self.experiments = {}
        self.learning_curves = {}

    def track_experiment(
        self,
        experiment_name: str,
        episode: int,
        reward: float,
        consensus_score: float,
        action_taken: str,
    ) -> None:
        """追踪实验数据"""
        if experiment_name not in self.experiments:
            self.experiments[experiment_name] = {
                "episodes": [],
                "rewards": [],
                "consensus_scores": [],
                "actions": [],
            }

        self.experiments[experiment_name]["episodes"].append(episode)
        self.experiments[experiment_name]["rewards"].append(reward)
        self.experiments[experiment_name]["consensus_scores"].append(consensus_score)
        self.experiments[experiment_name]["actions"].append(action_taken)

    def plot_learning_curves(self, experiment_names: List[str] = None) -> None:
        """绘制学习曲线"""
        if experiment_names is None:
            experiment_names = list(self.experiments.keys())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        colors = plt.cm.Set1(np.linspace(0, 1, len(experiment_names)))

        for i, exp_name in enumerate(experiment_names):
            if exp_name not in self.experiments:
                continue

            data = self.experiments[exp_name]
            episodes = data["episodes"]
            rewards = data["rewards"]
            consensus_scores = data["consensus_scores"]

            # 计算移动平均
            window_size = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
            if window_size > 1:
                rewards_ma = pd.Series(rewards).rolling(window=window_size).mean()
                consensus_ma = (
                    pd.Series(consensus_scores).rolling(window=window_size).mean()
                )
            else:
                rewards_ma = rewards
                consensus_ma = consensus_scores

            # 绘制奖励曲线
            ax1.plot(episodes, rewards, alpha=0.3, color=colors[i])
            ax1.plot(
                episodes, rewards_ma, label=f"{exp_name}", color=colors[i], linewidth=2
            )

            # 绘制共识得分曲线
            ax2.plot(episodes, consensus_scores, alpha=0.3, color=colors[i])
            ax2.plot(
                episodes,
                consensus_ma,
                label=f"{exp_name}",
                color=colors[i],
                linewidth=2,
            )

        ax1.set_title("强化学习奖励曲线", fontsize=14, fontweight="bold")
        ax1.set_xlabel("训练轮数")
        ax1.set_ylabel("累积奖励")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_title("共识得分变化", fontsize=14, fontweight="bold")
        ax2.set_xlabel("训练轮数")
        ax2.set_ylabel("共识得分")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_experiment_report(self, experiment_name: str) -> Dict[str, Any]:
        """生成实验报告"""
        if experiment_name not in self.experiments:
            return {}

        data = self.experiments[experiment_name]
        rewards = data["rewards"]
        consensus_scores = data["consensus_scores"]

        report = {
            "experiment_name": experiment_name,
            "total_episodes": len(data["episodes"]),
            "final_reward": rewards[-1] if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "avg_reward": np.mean(rewards) if rewards else 0,
            "reward_std": np.std(rewards) if rewards else 0,
            "final_consensus": consensus_scores[-1] if consensus_scores else 0,
            "avg_consensus": np.mean(consensus_scores) if consensus_scores else 0,
            "improvement": rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
        }

        return report


class ModelComparison:
    """模型对比实验"""

    def __init__(self):
        self.baselines = {}
        self.results = {}

    def add_baseline(self, name: str, model_fn) -> None:
        """添加基线模型"""
        self.baselines[name] = model_fn

    def run_comparison(
        self, test_cases: List[Dict], metrics: List[str] = None
    ) -> pd.DataFrame:
        """运行对比实验"""
        if metrics is None:
            metrics = ["accuracy", "consensus_alignment", "response_time"]

        results = []

        for model_name, model_fn in self.baselines.items():
            model_results = {"model": model_name}

            for metric in metrics:
                scores = []
                for test_case in test_cases:
                    score = self._evaluate_model(model_fn, test_case, metric)
                    scores.append(score)

                model_results[f"{metric}_mean"] = np.mean(scores)
                model_results[f"{metric}_std"] = np.std(scores)

            results.append(model_results)

        return pd.DataFrame(results)

    def _evaluate_model(self, model_fn, test_case: Dict, metric: str) -> float:
        """评估单个模型"""
        # 这里应该实现具体的评估逻辑
        # 返回模拟分数
        if metric == "accuracy":
            return np.random.uniform(0.6, 0.95)
        elif metric == "consensus_alignment":
            return np.random.uniform(0.5, 0.9)
        elif metric == "response_time":
            return np.random.uniform(0.1, 2.0)
        else:
            return np.random.uniform(0, 1)

    def plot_comparison(self, results_df: pd.DataFrame) -> None:
        """绘制对比结果"""
        models = results_df["model"].tolist()
        metrics = [
            col.replace("_mean", "")
            for col in results_df.columns
            if col.endswith("_mean")
        ]

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]

        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        for i, metric in enumerate(metrics):
            means = results_df[f"{metric}_mean"].tolist()
            stds = results_df[f"{metric}_std"].tolist()

            bars = axes[i].bar(
                models, means, yerr=stds, capsize=5, color=colors, alpha=0.7
            )

            axes[i].set_title(
                f'{metric.replace("_", " ").title()}', fontsize=12, fontweight="bold"
            )
            axes[i].set_ylabel("Score")
            axes[i].tick_params(axis="x", rotation=45)

            # 添加数值标签
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.show()


def demo_visualization():
    """演示可视化功能"""
    # 创建示例数据
    treatments = ["surgery", "chemotherapy", "radiotherapy", "immunotherapy"]
    roles = ["oncologist", "radiologist", "nurse", "psychologist", "patient_advocate"]

    # 生成示例共识矩阵
    np.random.seed(42)
    matrix_data = np.random.uniform(-1, 1, (len(treatments), len(roles)))
    consensus_matrix = pd.DataFrame(matrix_data, index=treatments, columns=roles)

    # 可视化
    visualizer = ConsensusVisualizer()
    visualizer.plot_consensus_heatmap(consensus_matrix)

    # 生成聚合评分
    aggregated_scores = {
        "surgery": 0.7,
        "chemotherapy": 0.5,
        "radiotherapy": 0.3,
        "immunotherapy": 0.8,
        "palliative_care": -0.2,
        "watchful_waiting": -0.1,
    }

    visualizer.plot_consensus_radar(aggregated_scores)

    # 生成分歧数据
    conflicts = [
        {
            "treatment": "surgery",
            "variance": 0.8,
            "min_score": -0.5,
            "max_score": 0.9,
            "conflicting_roles": ["nurse", "patient_advocate"],
        },
        {
            "treatment": "chemotherapy",
            "variance": 0.6,
            "min_score": -0.3,
            "max_score": 0.8,
            "conflicting_roles": ["psychologist"],
        },
    ]

    visualizer.plot_role_disagreement(conflicts)


def demo_rl_tracking():
    """演示RL实验追踪"""
    tracker = RLExperimentTracker()

    # 模拟实验数据
    experiments = ["MDT_RL", "Baseline_RL", "No_Memory_RL"]

    for exp in experiments:
        for episode in range(1000):
            # 模拟学习过程
            base_reward = np.random.normal(0.5, 0.2)
            if exp == "MDT_RL":
                reward = base_reward + 0.3 * (episode / 1000)  # 改进的学习
            elif exp == "Baseline_RL":
                reward = base_reward + 0.1 * (episode / 1000)  # 较慢的学习
            else:
                reward = base_reward  # 无改进

            consensus_score = max(0, min(1, reward + np.random.normal(0, 0.1)))
            action = np.random.choice(["surgery", "chemotherapy", "radiotherapy"])

            tracker.track_experiment(exp, episode, reward, consensus_score, action)

    # 绘制学习曲线
    tracker.plot_learning_curves()

    # 生成报告
    for exp in experiments:
        report = tracker.generate_experiment_report(exp)
        print(f"\n{exp} 实验报告:")
        for key, value in report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    print("运行可视化演示...")
    demo_visualization()

    print("\n运行RL追踪演示...")
    demo_rl_tracking()
