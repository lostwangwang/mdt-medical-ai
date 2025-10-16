"""
System Visualization Tools
File path: src/utils/visualization.py
Author: Team maintenance (Yao Gang main contributor for visualization logic)
Function: Provide various visualization functions for MDT system
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import networkx as nx
from wordcloud import WordCloud
import logging

from ..core.data_models import PatientState, ConsensusResult, TreatmentOption, RoleType

logger = logging.getLogger(__name__)

# Set matplotlib parameters for better display
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# Color themes
COLOR_THEMES = {
    "medical": {
        "primary": "#2E86C1",  # Medical blue
        "secondary": "#E74C3C",  # Emergency red
        "success": "#27AE60",  # Health green
        "warning": "#F39C12",  # Warning orange
        "info": "#8E44AD",  # Info purple
        "light": "#ECF0F1",  # Light gray
        "dark": "#2C3E50",  # Dark blue-gray
    },
    "roles": {
        "oncologist": "#E74C3C",  # Red
        "radiologist": "#3498DB",  # Blue
        "nurse": "#2ECC71",  # Green
        "psychologist": "#9B59B6",  # Purple
        "patient_advocate": "#F39C12",  # Orange
    },
    "treatments": {
        "surgery": "#E74C3C",  # Red
        "chemotherapy": "#8E44AD",  # Purple
        "radiotherapy": "#3498DB",  # Blue
        "immunotherapy": "#2ECC71",  # Green
        "palliative_care": "#F39C12",  # Orange
        "watchful_waiting": "#95A5A6",  # Gray
    },
}


class SystemVisualizer:
    """System Visualizer"""

    def __init__(self, theme: str = "medical"):
        self.theme = COLOR_THEMES.get(theme, COLOR_THEMES["medical"])
        self.figure_size = (12, 8)
        self.dpi = 300

        # Set default style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        logger.info("System visualizer initialized")

    def create_patient_analysis_dashboard(
        self, patient_state: PatientState, consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """Create patient analysis dashboard"""
        logger.info(
            f"Creating patient analysis dashboard for {patient_state.patient_id}"
        )

        dashboard = {}

        # 1. Patient basic information card
        dashboard["patient_info"] = self._create_patient_info_card(patient_state)

        # 2. Consensus matrix heatmap
        dashboard["consensus_heatmap"] = self._create_consensus_heatmap(
            consensus_result
        )

        # 3. Treatment plan radar chart
        dashboard["treatment_radar"] = self._create_treatment_radar_chart(
            consensus_result
        )

        # 4. Role opinion comparison chart
        dashboard["role_comparison"] = self._create_role_comparison_chart(
            consensus_result
        )

        # 5. Conflict and consistency analysis
        dashboard["conflict_analysis"] = self._create_conflict_analysis_chart(
            consensus_result
        )

        # 6. Dialogue flow chart
        if (
            hasattr(consensus_result, "dialogue_summary")
            and consensus_result.dialogue_summary
        ):
            dashboard["dialogue_flow"] = self._create_dialogue_flow_chart(
                consensus_result
            )

        logger.info("Patient analysis dashboard created successfully")
        return dashboard

    def create_training_dashboard(
        self, training_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create training results dashboard"""
        logger.info("Creating training dashboard")

        dashboard = {}

        # 1. Learning curve
        dashboard["learning_curve"] = self._create_learning_curve(training_results)

        # 2. Reward distribution
        dashboard["reward_distribution"] = self._create_reward_distribution(
            training_results
        )

        # 3. Performance metrics trends
        dashboard["performance_trends"] = self._create_performance_trends(
            training_results
        )

        # 4. Convergence analysis
        dashboard["convergence_analysis"] = self._create_convergence_analysis(
            training_results
        )

        logger.info("Training dashboard created successfully")
        return dashboard

    def create_temporal_analysis_dashboard(
        self, simulation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create temporal analysis dashboard"""
        logger.info("Creating temporal analysis dashboard")

        dashboard = {}

        # 1. Decision timeline
        dashboard["decision_timeline"] = self._create_decision_timeline(
            simulation_results
        )

        # 2. Patient state evolution
        dashboard["patient_evolution"] = self._create_patient_evolution_chart(
            simulation_results
        )

        # 3. Consensus score trends
        dashboard["consensus_trends"] = self._create_consensus_trends(
            simulation_results
        )

        # 4. System performance metrics
        dashboard["system_metrics"] = self._create_system_metrics_chart(
            simulation_results
        )

        logger.info("Temporal analysis dashboard created successfully")
        return dashboard

    def _create_patient_info_card(self, patient_state: PatientState) -> plt.Figure:
        """Create patient information card"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.axis("off")

        # Patient basic information
        info_text = f"""
        Patient ID: {patient_state.patient_id}
        Age: {patient_state.age} years
        Diagnosis: {patient_state.diagnosis}
        Stage: Stage {patient_state.stage}
        Quality of Life Score: {patient_state.quality_of_life_score:.2f}
        Psychological Status: {patient_state.psychological_status}
        
        Laboratory Results:
        {self._format_dict_as_text(patient_state.lab_results, indent=2)}
        
        Vital Signs:
        {self._format_dict_as_text(patient_state.vital_signs, indent=2)}
        
        Symptoms: {', '.join(patient_state.symptoms) if patient_state.symptoms else 'None'}
        
        Comorbidities: {', '.join(patient_state.comorbidities) if patient_state.comorbidities else 'None'}
        """

        # Add background color and border
        bbox_props = dict(
            boxstyle="round,pad=0.5", facecolor=self.theme["light"], alpha=0.8
        )
        ax.text(
            0.05,
            0.95,
            info_text.strip(),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=bbox_props,
            family="monospace",
        )

        ax.set_title(
            f"Patient Information Card - {patient_state.patient_id}",
            fontsize=16,
            fontweight="bold",
            color=self.theme["primary"],
        )

        plt.tight_layout()
        return fig

    def _create_consensus_heatmap(
        self, consensus_result: ConsensusResult
    ) -> plt.Figure:
        """Create consensus matrix heatmap"""
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        # Get consensus matrix data
        consensus_matrix = consensus_result.consensus_matrix

        # Create heatmap
        im = ax.imshow(
            consensus_matrix.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1
        )

        # Set axis labels
        ax.set_xticks(range(len(consensus_matrix.columns)))
        ax.set_xticklabels(consensus_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(consensus_matrix.index)))
        ax.set_yticklabels(consensus_matrix.index)

        # Add value annotations
        for i in range(len(consensus_matrix.index)):
            for j in range(len(consensus_matrix.columns)):
                value = consensus_matrix.iloc[i, j]
                color = "white" if abs(value) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Support Level (-1: Strongly Oppose, +1: Strongly Support)", rotation=270, labelpad=15)

        ax.set_title(
            "Medical Team Consensus Matrix",
            fontsize=16,
            fontweight="bold",
            color=self.theme["primary"],
        )
        ax.set_xlabel("Medical Team Roles", fontsize=12)
        ax.set_ylabel("Treatment Options", fontsize=12)

        plt.tight_layout()
        return fig

    def _create_treatment_radar_chart(
        self, consensus_result: ConsensusResult
    ) -> go.Figure:
        """Create treatment options radar chart"""
        # Prepare data
        treatments = list(consensus_result.aggregated_scores.keys())
        scores = list(consensus_result.aggregated_scores.values())

        # Create radar chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=scores,
                theta=[t.value for t in treatments],
                fill="toself",
                name="Consensus Score",
                line_color=self.theme["primary"],
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
            title={
                "text": "Treatment Options Consensus Radar Chart",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16, "color": self.theme["primary"]},
            },
            showlegend=True,
        )

        return fig

    def _create_role_comparison_chart(
        self, consensus_result: ConsensusResult
    ) -> plt.Figure:
        """Create role opinion comparison chart"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Prepare data
        roles = list(consensus_result.role_opinions.keys())
        treatments = list(TreatmentOption)

        # Create grouped bar chart
        x = np.arange(len(treatments))
        width = 0.15

        for i, role in enumerate(roles):
            if role in consensus_result.role_opinions:
                opinion = consensus_result.role_opinions[role]
                scores = [
                    opinion.treatment_preferences.get(treatment, 0)
                    for treatment in treatments
                ]

                color = COLOR_THEMES["roles"].get(role.value, self.theme["primary"])
                bars = ax.bar(
                    x + i * width,
                    scores,
                    width,
                    label=role.value,
                    color=color,
                    alpha=0.8,
                )

                # Add value labels
                for bar, score in zip(bars, scores):
                    if abs(score) > 0.1:  # Only show significant values
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.02 if height >= 0 else height - 0.05,
                            f"{score:.2f}",
                            ha="center",
                            va="bottom" if height >= 0 else "top",
                            fontsize=8,
                        )

        # Set chart properties
        ax.set_xlabel("Treatment Options", fontsize=12)
        ax.set_ylabel("Support Score", fontsize=12)
        ax.set_title(
            "Treatment Preference Comparison by Role",
            fontsize=16,
            fontweight="bold",
            color=self.theme["primary"],
        )
        ax.set_xticks(x + width * (len(roles) - 1) / 2)
        ax.set_xticklabels([t.value for t in treatments], rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_conflict_analysis_chart(
        self, consensus_result: ConsensusResult
    ) -> plt.Figure:
        """Create conflict and consistency analysis chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left chart: Conflict analysis
        if consensus_result.conflicts:
            conflicts_df = pd.DataFrame(consensus_result.conflicts)
            treatments = [
                conflict["treatment"].value for conflict in consensus_result.conflicts
            ]
            variances = [
                conflict["variance"] for conflict in consensus_result.conflicts
            ]

            bars1 = ax1.bar(
                treatments, variances, color=self.theme["secondary"], alpha=0.7
            )
            ax1.set_title("Treatment Option Disagreement Level", fontsize=14, fontweight="bold")
            ax1.set_xlabel("Treatment Options")
            ax1.set_ylabel("Disagreement Level (Variance)")
            ax1.tick_params(axis="x", rotation=45)

            # Add value labels
            for bar, variance in zip(bars1, variances):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{variance:.3f}",
                    ha="center",
                    va="bottom",
                )
        else:
            ax1.text(
                0.5,
                0.5,
                "No Significant Conflicts",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=16,
                color=self.theme["success"],
            )
            ax1.set_title("Treatment Option Disagreement Level", fontsize=14, fontweight="bold")

        # Right chart: Consistency analysis
        if consensus_result.agreements:
            agreements_df = pd.DataFrame(consensus_result.agreements)
            treatments = [
                agreement["treatment"].value
                for agreement in consensus_result.agreements
            ]
            strengths = [
                agreement["agreement_strength"]
                for agreement in consensus_result.agreements
            ]

            bars2 = ax2.bar(
                treatments, strengths, color=self.theme["success"], alpha=0.7
            )
            ax2.set_title("Treatment Option Agreement Strength", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Treatment Options")
            ax2.set_ylabel("Agreement Strength")
            ax2.tick_params(axis="x", rotation=45)

            # Add value labels
            for bar, strength in zip(bars2, strengths):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{strength:.3f}",
                    ha="center",
                    va="bottom",
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "No Strong Agreement",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=16,
                color=self.theme["warning"],
            )
            ax2.set_title("Treatment Option Agreement Strength", fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    def _create_dialogue_flow_chart(
        self, consensus_result: ConsensusResult
    ) -> go.Figure:
        """Create dialogue flow chart"""
        dialogue_summary = consensus_result.dialogue_summary

        # Create network graph representing dialogue flow
        fig = go.Figure()

        # Simulate dialogue nodes and connections
        if dialogue_summary and "key_topics" in dialogue_summary:
            topics = [topic[0].value for topic in dialogue_summary["key_topics"][:5]]

            # Create Sankey diagram showing topic flow
            source = []
            target = []
            value = []

            for i in range(len(topics) - 1):
                source.append(i)
                target.append(i + 1)
                value.append(1)

            fig.add_trace(
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=topics,
                        color=self.theme["primary"],
                    ),
                    link=dict(source=source, target=target, value=value),
                )
            )

        fig.update_layout(
            title={
                "text": "MDT Dialogue Flow Chart",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16, "color": self.theme["primary"]},
            }
        )

        return fig

    def _create_learning_curve(self, training_results: Dict[str, Any]) -> plt.Figure:
        """Create learning curve"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        learning_curve = training_results.get("learning_curve", [])
        if not learning_curve:
            return fig

        episodes = list(range(len(learning_curve)))

        # Original learning curve
        ax1.plot(
            episodes, learning_curve, alpha=0.6, color=self.theme["info"], linewidth=1
        )

        # Moving average
        window_size = max(1, len(learning_curve) // 20)
        if window_size > 1:
            moving_avg = pd.Series(learning_curve).rolling(window=window_size).mean()
            ax1.plot(
                episodes,
                moving_avg,
                color=self.theme["primary"],
                linewidth=2,
                label=f"Moving Average ({window_size})",
            )

        ax1.set_xlabel("Training Episodes")
        ax1.set_ylabel("Reward Value")
        ax1.set_title("Reinforcement Learning Training Curve", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Reward distribution histogram
        ax2.hist(
            learning_curve,
            bins=50,
            alpha=0.7,
            color=self.theme["secondary"],
            edgecolor="black",
        )
        ax2.set_xlabel("Reward Value")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Reward Distribution Histogram", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_reward_distribution(
        self, training_results: Dict[str, Any]
    ) -> plt.Figure:
        """Create reward distribution chart"""
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        learning_curve = training_results.get("learning_curve", [])
        if not learning_curve:
            ax.text(
                0.5,
                0.5,
                "No Training Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            return fig

        # Create combination of box plot and violin plot
        parts = ax.violinplot(
            [learning_curve], positions=[1], widths=0.8, showmeans=True
        )

        # Customize colors
        for pc in parts["bodies"]:
            pc.set_facecolor(self.theme["primary"])
            pc.set_alpha(0.6)

        # Add statistical information
        mean_reward = np.mean(learning_curve)
        std_reward = np.std(learning_curve)
        max_reward = np.max(learning_curve)
        min_reward = np.min(learning_curve)

        stats_text = f"""
        Mean Reward: {mean_reward:.3f}
        Std Dev: {std_reward:.3f}
        Max Reward: {max_reward:.3f}
        Min Reward: {min_reward:.3f}
        """

        ax.text(
            0.02,
            0.98,
            stats_text.strip(),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=self.theme["light"]),
        )

        ax.set_title("Training Reward Distribution Analysis", fontsize=16, fontweight="bold")
        ax.set_ylabel("Reward Value")
        ax.set_xticks([1])
        ax.set_xticklabels(["Reward Distribution"])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_performance_trends(
        self, training_results: Dict[str, Any]
    ) -> plt.Figure:
        """Create performance metrics trend chart"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Simulate performance metrics data
        episodes = len(training_results.get("learning_curve", []))
        if episodes == 0:
            return fig

        # Prepare simulated performance metrics
        metrics = {
            "Average Reward": (
                np.cumsum(np.random.normal(0.01, 0.1, episodes // 100))
                if episodes >= 100
                else [0]
            ),
            "Convergence": (
                1 - np.exp(-np.linspace(0, 3, episodes // 100))
                if episodes >= 100
                else [0]
            ),
            "Exploration Rate": (
                np.exp(-np.linspace(0, 2, episodes // 100)) if episodes >= 100 else [1]
            ),
            "Loss Function": (
                np.exp(-np.linspace(0, 4, episodes // 100))
                + np.random.normal(0, 0.05, episodes // 100)
                if episodes >= 100
                else [1]
            ),
        }

        for i, (metric_name, values) in enumerate(metrics.items()):
            if i >= 4:
                break

            x_axis = np.linspace(0, episodes, len(values))
            axes[i].plot(x_axis, values, color=self.theme["primary"], linewidth=2)
            axes[i].set_title(metric_name, fontweight="bold")
            axes[i].set_xlabel("Training Episodes")
            axes[i].grid(True, alpha=0.3)

            # Add trend line
            if len(values) > 1:
                z = np.polyfit(x_axis, values, 1)
                p = np.poly1d(z)
                axes[i].plot(
                    x_axis, p(x_axis), "--", color=self.theme["secondary"], alpha=0.8
                )

        plt.suptitle("Training Performance Metrics Trends", fontsize=16, fontweight="bold")
        plt.tight_layout()
        return fig

    def _create_convergence_analysis(
        self, training_results: Dict[str, Any]
    ) -> plt.Figure:
        """Create convergence analysis chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        learning_curve = training_results.get("learning_curve", [])
        if not learning_curve:
            return fig

        # Left chart: Convergence trend
        window_sizes = [10, 50, 100]
        colors = [self.theme["primary"], self.theme["secondary"], self.theme["info"]]

        for window, color in zip(window_sizes, colors):
            if len(learning_curve) > window:
                rolling_std = pd.Series(learning_curve).rolling(window=window).std()
                episodes = list(range(len(rolling_std)))
                ax1.plot(
                    episodes,
                    rolling_std,
                    label=f"Rolling Std ({window})",
                    color=color,
                    linewidth=2,
                )

        ax1.set_xlabel("Training Episodes")
        ax1.set_ylabel("Standard Deviation")
        ax1.set_title("Convergence Trend Analysis", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right chart: Stability analysis of last 1000 episodes
        if len(learning_curve) > 1000:
            recent_rewards = learning_curve[-1000:]

            # Calculate rolling variance
            rolling_var = pd.Series(recent_rewards).rolling(window=50).var()
            ax2.plot(rolling_var, color=self.theme["primary"], linewidth=2)
            ax2.set_title("Recent Stability Analysis (Last 1000 Episodes)", fontweight="bold")
            ax2.set_xlabel("Episodes (Relative Position)")
            ax2.set_ylabel("Variance")
            ax2.grid(True, alpha=0.3)

            # Add stability threshold line
            stable_threshold = np.percentile(rolling_var.dropna(), 25)
            ax2.axhline(
                y=stable_threshold,
                color=self.theme["success"],
                linestyle="--",
                label=f"Stability Threshold: {stable_threshold:.3f}",
            )
            ax2.legend()

        plt.tight_layout()
        return fig

    def _create_decision_timeline(
        self, simulation_results: Dict[str, Any]
    ) -> go.Figure:
        """Create decision timeline"""
        mdt_discussions = simulation_results.get("mdt_discussions", [])

        if not mdt_discussions:
            fig = go.Figure()
            fig.add_annotation(text="No MDT Discussion Data", x=0.5, y=0.5, showarrow=False)
            return fig

        # Prepare timeline data
        days = [d["day"] for d in mdt_discussions]
        treatments = [
            d["decision"]["recommended_treatment"].value for d in mdt_discussions
        ]
        consensus_scores = [d["decision"]["consensus_score"] for d in mdt_discussions]

        # Create timeline chart
        fig = go.Figure()

        # Add decision points
        fig.add_trace(
            go.Scatter(
                x=days,
                y=consensus_scores,
                mode="markers+lines",
                marker=dict(
                    size=[score * 20 + 10 for score in consensus_scores],
                    color=consensus_scores,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Consensus Score"),
                ),
                text=[
                    f"Day {day}: {treatment}<br>Score: {score:.3f}"
                    for day, treatment, score in zip(days, treatments, consensus_scores)
                ],
                hovertemplate="%{text}<extra></extra>",
                name="MDT Decisions",
            )
        )

        fig.update_layout(
            title="MDT Decision Timeline",
            xaxis_title="Days",
            yaxis_title="Consensus Score",
            hovermode="closest",
        )

        return fig

    def _create_patient_evolution_chart(
        self, simulation_results: Dict[str, Any]
    ) -> plt.Figure:
        """Create patient state evolution chart"""
        daily_events = simulation_results.get("daily_events", [])

        if not daily_events:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
            ax.text(
                0.5,
                0.5,
                "No Patient Evolution Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            return fig

        # Extract time series data
        days = [event["day"] for event in daily_events]
        quality_scores = [
            event["patient_state"].get("quality_of_life_score", 0.5)
            for event in daily_events
        ]
        creatinine_levels = [
            event["patient_state"]["lab_results"].get("creatinine", 1.0)
            for event in daily_events
        ]

        # Create dual y-axis chart
        fig, ax1 = plt.subplots(1, 1, figsize=self.figure_size)
        ax2 = ax1.twinx()

        # Quality of life score
        line1 = ax1.plot(
            days,
            quality_scores,
            color=self.theme["primary"],
            linewidth=2,
            marker="o",
            markersize=4,
            label="Quality of Life Score",
        )
        ax1.set_ylabel("Quality of Life Score", color=self.theme["primary"])
        ax1.tick_params(axis="y", labelcolor=self.theme["primary"])

        # Creatinine level
        line2 = ax2.plot(
            days,
            creatinine_levels,
            color=self.theme["secondary"],
            linewidth=2,
            marker="s",
            markersize=4,
            label="Creatinine Level",
        )
        ax2.set_ylabel("Creatinine Level (mg/dL)", color=self.theme["secondary"])
        ax2.tick_params(axis="y", labelcolor=self.theme["secondary"])

        # Set x-axis
        ax1.set_xlabel("Days")
        ax1.set_title("Patient State Evolution Trajectory", fontsize=16, fontweight="bold")

        # Add reference lines
        ax1.axhline(
            y=0.5,
            color=self.theme["warning"],
            linestyle="--",
            alpha=0.7,
            label="Quality of Life Alert Line",
        )
        ax2.axhline(
            y=1.5,
            color=self.theme["warning"],
            linestyle="--",
            alpha=0.7,
            label="Creatinine Alert Line",
        )

        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _create_consensus_trends(
        self, simulation_results: Dict[str, Any]
    ) -> plt.Figure:
        """Create consensus score trends"""
        mdt_discussions = simulation_results.get("mdt_discussions", [])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        if not mdt_discussions:
            ax1.text(
                0.5,
                0.5,
                "No MDT Discussion Data",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=16,
            )
            ax2.text(
                0.5,
                0.5,
                "No Discussion Rounds Data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=16,
            )
            return fig

        days = [d["day"] for d in mdt_discussions]
        consensus_scores = [d["decision"]["consensus_score"] for d in mdt_discussions]
        discussion_rounds = [
            d["consensus_result"].total_rounds for d in mdt_discussions
        ]

        # Top chart: Consensus score trends
        ax1.plot(
            days,
            consensus_scores,
            marker="o",
            linewidth=2,
            color=self.theme["primary"],
            markersize=6,
        )
        ax1.set_ylabel("Consensus Score")
        ax1.set_title("MDT Consensus Score Trends", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)

        # Add trend line
        if len(days) > 1:
            z = np.polyfit(days, consensus_scores, 1)
            p = np.poly1d(z)
            ax1.plot(
                days,
                p(days),
                "--",
                color=self.theme["secondary"],
                alpha=0.8,
                label=f"Trend Line (Slope: {z[0]:.4f})",
            )
            ax1.legend()

        # Bottom chart: Discussion rounds trends
        bars = ax2.bar(days, discussion_rounds, alpha=0.7, color=self.theme["info"])
        ax2.set_ylabel("Discussion Rounds")
        ax2.set_xlabel("Days")
        ax2.set_title("MDT Discussion Rounds Changes", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, rounds in zip(bars, discussion_rounds):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{rounds}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        return fig

    def _create_system_metrics_chart(
        self, simulation_results: Dict[str, Any]
    ) -> plt.Figure:
        """Create system performance metrics chart"""
        performance_metrics = simulation_results.get("performance_metrics", {})

        if not performance_metrics:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
            ax.text(
                0.5,
                0.5,
                "No Performance Metrics Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            return fig

        # Prepare metrics data
        metrics = {
            "Avg Consensus Score": performance_metrics.get("avg_consensus_score", 0),
            "Avg Discussion Rounds": performance_metrics.get("avg_discussion_rounds", 0)
            / 5,  # Normalized
            "Convergence Rate": performance_metrics.get("convergence_rate", 0),
            "Decision Consistency": performance_metrics.get("decision_consistency", 0),
            "Avg RL Reward": (performance_metrics.get("avg_rl_reward", 0) + 1)
            / 2,  # Normalized to [0,1]
        }

        # Create radar chart
        fig, ax = plt.subplots(
            1, 1, figsize=(10, 10), subplot_kw=dict(projection="polar")
        )

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values = list(metrics.values())

        # Close radar chart
        angles += angles[:1]
        values += values[:1]

        # Draw radar chart
        ax.plot(angles, values, "o-", linewidth=2, color=self.theme["primary"])
        ax.fill(angles, values, alpha=0.25, color=self.theme["primary"])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(list(metrics.keys()))
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.grid(True)

        # Add value annotations
        for angle, value, metric in zip(angles[:-1], values[:-1], metrics.keys()):
            ax.text(
                angle,
                value + 0.05,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontweight="bold",
            )

        ax.set_title(
            "System Comprehensive Performance Metrics",
            fontsize=16,
            fontweight="bold",
            color=self.theme["primary"],
            pad=20,
        )

        plt.tight_layout()
        return fig

    def _format_dict_as_text(self, data_dict: Dict[str, Any], indent: int = 0) -> str:
        """格式化字典为文本"""
        lines = []
        prefix = "  " * indent
        for key, value in data_dict.items():
            if isinstance(value, float):
                lines.append(f"{prefix}{key}: {value:.2f}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)

    def save_all_figures(self, dashboard: Dict[str, Any], output_dir: str) -> None:
        """保存所有图表"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        for name, fig in dashboard.items():
            if hasattr(fig, "savefig"):  # matplotlib figure
                fig.savefig(
                    f"{output_dir}/{name}.png", dpi=self.dpi, bbox_inches="tight"
                )
            elif hasattr(fig, "write_image"):  # plotly figure
                fig.write_image(f"{output_dir}/{name}.png")

        logger.info(f"All figures saved to {output_dir}")

    def create_summary_report_figure(
        self, patient_analysis: Dict[str, Any], training_results: Dict[str, Any] = None
    ) -> plt.Figure:
        """创建综合摘要报告图"""
        fig = plt.figure(figsize=(20, 14))

        # 创建复杂的子图布局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 患者信息概览
        ax1 = fig.add_subplot(gs[0, :2])
        self._add_patient_summary_to_axis(ax1, patient_analysis)

        # 治疗推荐
        ax2 = fig.add_subplot(gs[0, 2:])
        self._add_treatment_recommendation_to_axis(ax2, patient_analysis)

        # 共识矩阵（简化版）
        ax3 = fig.add_subplot(gs[1, :2])
        self._add_mini_consensus_matrix_to_axis(ax3, patient_analysis)

        # 系统性能指标
        ax4 = fig.add_subplot(gs[1, 2:])
        if training_results:
            self._add_performance_summary_to_axis(ax4, training_results)
        else:
            ax4.text(
                0.5,
                0.5,
                "无训练数据",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=14,
            )

        # 决策摘要
        ax5 = fig.add_subplot(gs[2, :])
        self._add_decision_summary_to_axis(ax5, patient_analysis)

        plt.suptitle(
            "MDT医疗智能体系统 - 综合分析报告",
            fontsize=20,
            fontweight="bold",
            color=self.theme["primary"],
        )

        return fig

    def _add_patient_summary_to_axis(
        self, ax, patient_analysis: Dict[str, Any]
    ) -> None:
        """在子图中添加患者摘要"""
        ax.axis("off")

        patient_info = patient_analysis.get("patient_info", {})
        summary_text = f"""
        患者基本信息
        
        ID: {patient_info.get('patient_id', 'N/A')}
        年龄: {patient_info.get('age', 'N/A')}岁
        诊断: {patient_info.get('diagnosis', 'N/A')}
        分期: {patient_info.get('stage', 'N/A')}期
        """

        ax.text(
            0.05,
            0.95,
            summary_text.strip(),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=self.theme["light"]),
        )

        ax.set_title("患者概览", fontsize=14, fontweight="bold")

    def _add_treatment_recommendation_to_axis(
        self, ax, patient_analysis: Dict[str, Any]
    ) -> None:
        """在子图中添加治疗推荐"""
        ax.axis("off")

        consensus_result = patient_analysis.get("consensus_result", {})

        recommendation_text = f"""
        治疗推荐
        
        推荐方案: {consensus_result.get('recommended_treatment', 'N/A')}
        共识得分: {consensus_result.get('consensus_score', 0):.3f}
        讨论轮数: {consensus_result.get('total_rounds', 0)}
        收敛状态: {'已收敛' if consensus_result.get('convergence_achieved', False) else '未收敛'}
        """

        ax.text(
            0.05,
            0.95,
            recommendation_text.strip(),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor=self.theme["success"], alpha=0.3
            ),
        )

        ax.set_title("推荐结果", fontsize=14, fontweight="bold")

    def _add_mini_consensus_matrix_to_axis(
        self, ax, patient_analysis: Dict[str, Any]
    ) -> None:
        """在子图中添加简化的共识矩阵"""
        # 这里添加一个简化的共识矩阵可视化
        ax.text(
            0.5,
            0.5,
            "共识矩阵\n(简化视图)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("团队共识", fontsize=14, fontweight="bold")

    def _add_performance_summary_to_axis(
        self, ax, training_results: Dict[str, Any]
    ) -> None:
        """在子图中添加性能摘要"""
        final_metrics = training_results.get("final_metrics", {})

        performance_text = f"""
        系统性能
        
        平均奖励: {final_metrics.get('recent_average_reward', 0):.3f}
        学习改进: {final_metrics.get('improvement', 0):+.3f}
        训练轮数: {final_metrics.get('total_episodes', 0)}
        """

        ax.text(
            0.05,
            0.95,
            performance_text.strip(),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor=self.theme["info"], alpha=0.3
            ),
        )

        ax.set_title("系统性能", fontsize=14, fontweight="bold")

    def _add_decision_summary_to_axis(
        self, ax, patient_analysis: Dict[str, Any]
    ) -> None:
        """在子图中添加决策摘要"""
        ax.axis("off")

        consensus_result = patient_analysis.get("consensus_result", {})

        summary_text = f"""
        决策分析摘要
        
        • 推荐治疗方案: {consensus_result.get('recommended_treatment', 'N/A')}
        • 团队共识得分: {consensus_result.get('consensus_score', 0):.3f}
        • 讨论收敛状态: {'✓ 已达成共识' if consensus_result.get('convergence_achieved', False) else '⚠ 需要进一步讨论'}
        • 发现冲突: {consensus_result.get('conflicts', 0)}个
        • 一致意见: {consensus_result.get('agreements', 0)}个
        
        系统建议: 基于多智能体协商，推荐采用上述治疗方案，并建议定期复议。
        """

        ax.text(
            0.05,
            0.95,
            summary_text.strip(),
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=self.theme["light"]),
        )

        ax.set_title("决策摘要与建议", fontsize=14, fontweight="bold")
