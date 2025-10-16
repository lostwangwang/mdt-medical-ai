"""
Visualization and Experimental Comparison Module
Features: Consensus matrix visualization, RL learning curves, model comparison experiments
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.font_manager as fm
import platform

# Configure font support
def setup_font():
    """Setup font support for charts"""
    system = platform.system()
    
    if system == "Windows":
        # Common fonts for Windows
        fonts = ['Arial', 'Calibri', 'Times New Roman']
    elif system == "Darwin":  # macOS
        # macOS fonts
        fonts = ['Arial', 'Helvetica', 'Times New Roman']
    else:  # Linux
        # Linux fonts
        fonts = ['DejaVu Sans', 'Liberation Sans', 'Arial']
    
    # Try to set available fonts
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
            print(f"✅ Successfully set font: {font}")
            return
        except:
            continue
    
    # If none available, use default settings with warning
    print("⚠️  No suitable font found, may display as boxes")
    plt.rcParams['axes.unicode_minus'] = False

# Initialize font settings
setup_font()


class ConsensusVisualizer:
    """Consensus matrix visualization tool"""

    @staticmethod
    def plot_consensus_heatmap(
        consensus_matrix: pd.DataFrame, title: str = "Medical Team Treatment Consensus Matrix"
    ) -> None:
        """Plot consensus matrix heatmap"""
        plt.figure(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(
            consensus_matrix,
            annot=True,
            cmap="RdYlGn",
            center=0,
            fmt=".2f",
            cbar_kws={"label": "Support Level (-1: Strongly Oppose, +1: Strongly Support)"},
        )

        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Medical Team Roles", fontsize=12)
        plt.ylabel("Treatment Options", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_consensus_radar(
        aggregated_scores: Dict[str, float], title: str = "Treatment Options Comprehensive Score"
    ) -> None:
        """Plot treatment options radar chart"""
        treatments = list(aggregated_scores.keys())
        scores = list(aggregated_scores.values())

        # Prepare radar chart data
        angles = np.linspace(0, 2 * np.pi, len(treatments), endpoint=False).tolist()
        scores += scores[:1]  # Close radar chart
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # Draw radar chart
        ax.plot(angles, scores, "o-", linewidth=2, label="Consensus Score")
        ax.fill(angles, scores, alpha=0.25)

        # Set labels
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
        """Plot role disagreement analysis"""
        if not conflicts:
            print("No significant role disagreements found")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Treatment disagreement statistics
        treatments = [conflict["treatment"] for conflict in conflicts]
        variances = [conflict["variance"] for conflict in conflicts]

        ax1.bar(treatments, variances, color="coral")
        ax1.set_title("Treatment Option Disagreement Level", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Treatment Options")
        ax1.set_ylabel("Disagreement Level (Variance)")
        ax1.tick_params(axis="x", rotation=45)

        # Disagreement range analysis
        min_scores = [conflict["min_score"] for conflict in conflicts]
        max_scores = [conflict["max_score"] for conflict in conflicts]

        x = np.arange(len(treatments))
        width = 0.35

        ax2.bar(x - width / 2, min_scores, width, label="Lowest Score", color="lightcoral")
        ax2.bar(x + width / 2, max_scores, width, label="Highest Score", color="lightgreen")

        ax2.set_title("Disagreement Score Range", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Treatment Options")
        ax2.set_ylabel("Score Range")
        ax2.set_xticks(x)
        ax2.set_xticklabels(treatments, rotation=45)
        ax2.legend()

        plt.tight_layout()
        plt.show()


class RLExperimentTracker:
    """Reinforcement Learning experiment tracker"""

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
        """Track experiment data"""
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
        """Plot learning curves"""
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

            # Calculate moving average
            window_size = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
            if window_size > 1:
                rewards_ma = pd.Series(rewards).rolling(window=window_size).mean()
                consensus_ma = (
                    pd.Series(consensus_scores).rolling(window=window_size).mean()
                )
            else:
                rewards_ma = rewards
                consensus_ma = consensus_scores

            # Plot reward curves
            ax1.plot(episodes, rewards, alpha=0.3, color=colors[i])
            ax1.plot(
                episodes, rewards_ma, label=f"{exp_name}", color=colors[i], linewidth=2
            )

            # Plot consensus score curves
            ax2.plot(episodes, consensus_scores, alpha=0.3, color=colors[i])
            ax2.plot(
                episodes,
                consensus_ma,
                label=f"{exp_name}",
                color=colors[i],
                linewidth=2,
            )

        ax1.set_title("Reinforcement Learning Reward Curves", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Training Episodes")
        ax1.set_ylabel("Cumulative Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_title("Consensus Score Changes", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Training Episodes")
        ax2.set_ylabel("Consensus Score")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_experiment_report(self, experiment_name: str) -> Dict[str, Any]:
        """Generate experiment report"""
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
    """Model comparison experiments"""

    def __init__(self):
        self.baselines = {}
        self.results = {}

    def add_baseline(self, name: str, model_fn) -> None:
        """Add baseline model"""
        self.baselines[name] = model_fn

    def run_comparison(
        self, test_cases: List[Dict], metrics: List[str] = None
    ) -> pd.DataFrame:
        """Run comparison experiments"""
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
        """Evaluate single model"""
        # Specific evaluation logic should be implemented here
        # Return simulated scores
        if metric == "accuracy":
            return np.random.uniform(0.6, 0.95)
        elif metric == "consensus_alignment":
            return np.random.uniform(0.5, 0.9)
        elif metric == "response_time":
            return np.random.uniform(0.1, 2.0)
        else:
            return np.random.uniform(0, 1)

    def plot_comparison(self, results_df: pd.DataFrame) -> None:
        """Plot comparison results"""
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

            # Add value labels
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
    """Demonstrate visualization features"""
    # Create sample data
    treatments = ["surgery", "chemotherapy", "radiotherapy", "immunotherapy"]
    roles = ["oncologist", "radiologist", "nurse", "psychologist", "patient_advocate"]

    # Generate sample consensus matrix
    np.random.seed(42)
    matrix_data = np.random.uniform(-1, 1, (len(treatments), len(roles)))
    consensus_matrix = pd.DataFrame(matrix_data, index=treatments, columns=roles)

    # Visualization
    visualizer = ConsensusVisualizer()
    visualizer.plot_consensus_heatmap(consensus_matrix)

    # Generate aggregated scores
    aggregated_scores = {
        "surgery": 0.7,
        "chemotherapy": 0.5,
        "radiotherapy": 0.3,
        "immunotherapy": 0.8,
        "palliative_care": -0.2,
        "watchful_waiting": -0.1,
    }

    visualizer.plot_consensus_radar(aggregated_scores)

    # Generate disagreement data
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
    """Demonstrate RL experiment tracking"""
    tracker = RLExperimentTracker()

    # Simulate experiment data
    experiments = ["MDT_RL", "Baseline_RL", "No_Memory_RL"]

    for exp in experiments:
        for episode in range(1000):
            # Simulate learning process
            base_reward = np.random.normal(0.5, 0.2)
            if exp == "MDT_RL":
                reward = base_reward + 0.3 * (episode / 1000)  # Improved learning
            elif exp == "Baseline_RL":
                reward = base_reward + 0.1 * (episode / 1000)  # Slower learning
            else:
                reward = base_reward  # No improvement

            consensus_score = max(0, min(1, reward + np.random.normal(0, 0.1)))
            action = np.random.choice(["surgery", "chemotherapy", "radiotherapy"])

            tracker.track_experiment(exp, episode, reward, consensus_score, action)

    # Plot learning curves
    tracker.plot_learning_curves()

    # Generate reports
    for exp in experiments:
        report = tracker.generate_experiment_report(exp)
        print(f"\n{exp} Experiment Report:")
        for key, value in report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    print("Running visualization demo...")
    demo_visualization()

    print("\nRunning RL tracking demo...")
    demo_rl_tracking()
