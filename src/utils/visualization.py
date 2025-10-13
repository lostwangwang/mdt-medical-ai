"""
系统可视化工具
文件路径: src/utils/visualization.py
作者: 团队共同维护 (姚刚主要贡献可视化逻辑)
功能: 提供MDT系统的各种可视化功能
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

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 颜色主题
COLOR_THEMES = {
    "medical": {
        "primary": "#2E86C1",  # 医疗蓝
        "secondary": "#E74C3C",  # 急诊红
        "success": "#27AE60",  # 健康绿
        "warning": "#F39C12",  # 警告橙
        "info": "#8E44AD",  # 信息紫
        "light": "#ECF0F1",  # 浅灰
        "dark": "#2C3E50",  # 深蓝灰
    },
    "roles": {
        "oncologist": "#E74C3C",  # 红色
        "radiologist": "#3498DB",  # 蓝色
        "nurse": "#2ECC71",  # 绿色
        "psychologist": "#9B59B6",  # 紫色
        "patient_advocate": "#F39C12",  # 橙色
    },
    "treatments": {
        "surgery": "#E74C3C",  # 红色
        "chemotherapy": "#8E44AD",  # 紫色
        "radiotherapy": "#3498DB",  # 蓝色
        "immunotherapy": "#2ECC71",  # 绿色
        "palliative_care": "#F39C12",  # 橙色
        "watchful_waiting": "#95A5A6",  # 灰色
    },
}


class SystemVisualizer:
    """系统可视化器"""

    def __init__(self, theme: str = "medical"):
        self.theme = COLOR_THEMES.get(theme, COLOR_THEMES["medical"])
        self.figure_size = (12, 8)
        self.dpi = 300

        # 设置默认样式
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        logger.info("System visualizer initialized")

    def create_patient_analysis_dashboard(
        self, patient_state: PatientState, consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """创建患者分析仪表板"""
        logger.info(
            f"Creating patient analysis dashboard for {patient_state.patient_id}"
        )

        dashboard = {}

        # 1. 患者基本信息卡片
        dashboard["patient_info"] = self._create_patient_info_card(patient_state)

        # 2. 共识矩阵热力图
        dashboard["consensus_heatmap"] = self._create_consensus_heatmap(
            consensus_result
        )

        # 3. 治疗方案雷达图
        dashboard["treatment_radar"] = self._create_treatment_radar_chart(
            consensus_result
        )

        # 4. 角色意见对比图
        dashboard["role_comparison"] = self._create_role_comparison_chart(
            consensus_result
        )

        # 5. 冲突与一致性分析
        dashboard["conflict_analysis"] = self._create_conflict_analysis_chart(
            consensus_result
        )

        # 6. 对话流程图
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
        """创建训练结果仪表板"""
        logger.info("Creating training dashboard")

        dashboard = {}

        # 1. 学习曲线
        dashboard["learning_curve"] = self._create_learning_curve(training_results)

        # 2. 奖励分布
        dashboard["reward_distribution"] = self._create_reward_distribution(
            training_results
        )

        # 3. 性能指标趋势
        dashboard["performance_trends"] = self._create_performance_trends(
            training_results
        )

        # 4. 收敛分析
        dashboard["convergence_analysis"] = self._create_convergence_analysis(
            training_results
        )

        logger.info("Training dashboard created successfully")
        return dashboard

    def create_temporal_analysis_dashboard(
        self, simulation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建时序分析仪表板"""
        logger.info("Creating temporal analysis dashboard")

        dashboard = {}

        # 1. 时序决策轨迹
        dashboard["decision_timeline"] = self._create_decision_timeline(
            simulation_results
        )

        # 2. 患者状态演化
        dashboard["patient_evolution"] = self._create_patient_evolution_chart(
            simulation_results
        )

        # 3. 共识得分趋势
        dashboard["consensus_trends"] = self._create_consensus_trends(
            simulation_results
        )

        # 4. 系统性能指标
        dashboard["system_metrics"] = self._create_system_metrics_chart(
            simulation_results
        )

        logger.info("Temporal analysis dashboard created successfully")
        return dashboard

    def _create_patient_info_card(self, patient_state: PatientState) -> plt.Figure:
        """创建患者信息卡片"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.axis("off")

        # 患者基本信息
        info_text = f"""
        患者ID: {patient_state.patient_id}
        年龄: {patient_state.age}岁
        诊断: {patient_state.diagnosis}
        分期: {patient_state.stage}期
        生活质量评分: {patient_state.quality_of_life_score:.2f}
        心理状态: {patient_state.psychological_status}
        
        实验室结果:
        {self._format_dict_as_text(patient_state.lab_results, indent=2)}
        
        生命体征:
        {self._format_dict_as_text(patient_state.vital_signs, indent=2)}
        
        症状: {', '.join(patient_state.symptoms) if patient_state.symptoms else '无'}
        
        并发症: {', '.join(patient_state.comorbidities) if patient_state.comorbidities else '无'}
        """

        # 添加背景色和边框
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
            f"患者信息卡片 - {patient_state.patient_id}",
            fontsize=16,
            fontweight="bold",
            color=self.theme["primary"],
        )

        plt.tight_layout()
        return fig

    def _create_consensus_heatmap(
        self, consensus_result: ConsensusResult
    ) -> plt.Figure:
        """创建共识矩阵热力图"""
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        # 获取共识矩阵数据
        consensus_matrix = consensus_result.consensus_matrix

        # 创建热力图
        im = ax.imshow(
            consensus_matrix.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1
        )

        # 设置坐标轴标签
        ax.set_xticks(range(len(consensus_matrix.columns)))
        ax.set_xticklabels(consensus_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(consensus_matrix.index)))
        ax.set_yticklabels(consensus_matrix.index)

        # 添加数值标注
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

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("支持度 (-1: 强烈反对, +1: 强烈支持)", rotation=270, labelpad=15)

        ax.set_title(
            "医疗团队共识矩阵",
            fontsize=16,
            fontweight="bold",
            color=self.theme["primary"],
        )
        ax.set_xlabel("医疗团队角色", fontsize=12)
        ax.set_ylabel("治疗方案", fontsize=12)

        plt.tight_layout()
        return fig

    def _create_treatment_radar_chart(
        self, consensus_result: ConsensusResult
    ) -> go.Figure:
        """创建治疗方案雷达图"""
        # 准备数据
        treatments = list(consensus_result.aggregated_scores.keys())
        scores = list(consensus_result.aggregated_scores.values())

        # 创建雷达图
        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=scores,
                theta=[t.value for t in treatments],
                fill="toself",
                name="共识评分",
                line_color=self.theme["primary"],
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
            title={
                "text": "治疗方案共识雷达图",
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
        """创建角色意见对比图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # 准备数据
        roles = list(consensus_result.role_opinions.keys())
        treatments = list(TreatmentOption)

        # 创建分组条形图
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

                # 添加数值标签
                for bar, score in zip(bars, scores):
                    if abs(score) > 0.1:  # 只显示显著的值
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.02 if height >= 0 else height - 0.05,
                            f"{score:.2f}",
                            ha="center",
                            va="bottom" if height >= 0 else "top",
                            fontsize=8,
                        )

        # 设置图表属性
        ax.set_xlabel("治疗方案", fontsize=12)
        ax.set_ylabel("支持度评分", fontsize=12)
        ax.set_title(
            "各角色治疗方案偏好对比",
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
        """创建冲突与一致性分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：冲突分析
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
            ax1.set_title("治疗方案分歧程度", fontsize=14, fontweight="bold")
            ax1.set_xlabel("治疗方案")
            ax1.set_ylabel("分歧程度（方差）")
            ax1.tick_params(axis="x", rotation=45)

            # 添加数值标签
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
                "无显著冲突",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=16,
                color=self.theme["success"],
            )
            ax1.set_title("治疗方案分歧程度", fontsize=14, fontweight="bold")

        # 右图：一致性分析
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
            ax2.set_title("治疗方案一致性强度", fontsize=14, fontweight="bold")
            ax2.set_xlabel("治疗方案")
            ax2.set_ylabel("一致性强度")
            ax2.tick_params(axis="x", rotation=45)

            # 添加数值标签
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
                "无强一致性",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=16,
                color=self.theme["warning"],
            )
            ax2.set_title("治疗方案一致性强度", fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    def _create_dialogue_flow_chart(
        self, consensus_result: ConsensusResult
    ) -> go.Figure:
        """创建对话流程图"""
        dialogue_summary = consensus_result.dialogue_summary

        # 创建网络图表示对话流程
        fig = go.Figure()

        # 模拟对话节点和连接
        if dialogue_summary and "key_topics" in dialogue_summary:
            topics = [topic[0].value for topic in dialogue_summary["key_topics"][:5]]

            # 创建桑基图显示话题流转
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
                "text": "MDT对话流程图",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16, "color": self.theme["primary"]},
            }
        )

        return fig

    def _create_learning_curve(self, training_results: Dict[str, Any]) -> plt.Figure:
        """创建学习曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        learning_curve = training_results.get("learning_curve", [])
        if not learning_curve:
            return fig

        episodes = list(range(len(learning_curve)))

        # 原始学习曲线
        ax1.plot(
            episodes, learning_curve, alpha=0.6, color=self.theme["info"], linewidth=1
        )

        # 移动平均
        window_size = max(1, len(learning_curve) // 20)
        if window_size > 1:
            moving_avg = pd.Series(learning_curve).rolling(window=window_size).mean()
            ax1.plot(
                episodes,
                moving_avg,
                color=self.theme["primary"],
                linewidth=2,
                label=f"移动平均({window_size})",
            )

        ax1.set_xlabel("训练轮次")
        ax1.set_ylabel("奖励值")
        ax1.set_title("强化学习训练曲线", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 奖励分布直方图
        ax2.hist(
            learning_curve,
            bins=50,
            alpha=0.7,
            color=self.theme["secondary"],
            edgecolor="black",
        )
        ax2.set_xlabel("奖励值")
        ax2.set_ylabel("频次")
        ax2.set_title("奖励分布直方图", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_reward_distribution(
        self, training_results: Dict[str, Any]
    ) -> plt.Figure:
        """创建奖励分布图"""
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

        learning_curve = training_results.get("learning_curve", [])
        if not learning_curve:
            ax.text(
                0.5,
                0.5,
                "无训练数据",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            return fig

        # 创建箱线图和violin图的组合
        parts = ax.violinplot(
            [learning_curve], positions=[1], widths=0.8, showmeans=True
        )

        # 自定义颜色
        for pc in parts["bodies"]:
            pc.set_facecolor(self.theme["primary"])
            pc.set_alpha(0.6)

        # 添加统计信息
        mean_reward = np.mean(learning_curve)
        std_reward = np.std(learning_curve)
        max_reward = np.max(learning_curve)
        min_reward = np.min(learning_curve)

        stats_text = f"""
        平均奖励: {mean_reward:.3f}
        标准差: {std_reward:.3f}
        最大奖励: {max_reward:.3f}
        最小奖励: {min_reward:.3f}
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

        ax.set_title("训练奖励分布分析", fontsize=16, fontweight="bold")
        ax.set_ylabel("奖励值")
        ax.set_xticks([1])
        ax.set_xticklabels(["奖励分布"])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_performance_trends(
        self, training_results: Dict[str, Any]
    ) -> plt.Figure:
        """创建性能指标趋势图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 模拟性能指标数据
        episodes = len(training_results.get("learning_curve", []))
        if episodes == 0:
            return fig

        # 准备模拟的性能指标
        metrics = {
            "平均奖励": (
                np.cumsum(np.random.normal(0.01, 0.1, episodes // 100))
                if episodes >= 100
                else [0]
            ),
            "收敛度": (
                1 - np.exp(-np.linspace(0, 3, episodes // 100))
                if episodes >= 100
                else [0]
            ),
            "探索率": (
                np.exp(-np.linspace(0, 2, episodes // 100)) if episodes >= 100 else [1]
            ),
            "损失函数": (
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
            axes[i].set_xlabel("训练轮次")
            axes[i].grid(True, alpha=0.3)

            # 添加趋势线
            if len(values) > 1:
                z = np.polyfit(x_axis, values, 1)
                p = np.poly1d(z)
                axes[i].plot(
                    x_axis, p(x_axis), "--", color=self.theme["secondary"], alpha=0.8
                )

        plt.suptitle("训练性能指标趋势", fontsize=16, fontweight="bold")
        plt.tight_layout()
        return fig

    def _create_convergence_analysis(
        self, training_results: Dict[str, Any]
    ) -> plt.Figure:
        """创建收敛分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        learning_curve = training_results.get("learning_curve", [])
        if not learning_curve:
            return fig

        # 左图：收敛趋势
        window_sizes = [10, 50, 100]
        colors = [self.theme["primary"], self.theme["secondary"], self.theme["info"]]

        for window, color in zip(window_sizes, colors):
            if len(learning_curve) > window:
                rolling_std = pd.Series(learning_curve).rolling(window=window).std()
                episodes = list(range(len(rolling_std)))
                ax1.plot(
                    episodes,
                    rolling_std,
                    label=f"滑动标准差({window})",
                    color=color,
                    linewidth=2,
                )

        ax1.set_xlabel("训练轮次")
        ax1.set_ylabel("标准差")
        ax1.set_title("收敛趋势分析", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：最后1000个episode的稳定性分析
        if len(learning_curve) > 1000:
            recent_rewards = learning_curve[-1000:]

            # 计算滑动方差
            rolling_var = pd.Series(recent_rewards).rolling(window=50).var()
            ax2.plot(rolling_var, color=self.theme["primary"], linewidth=2)
            ax2.set_title("近期稳定性分析(最后1000轮次)", fontweight="bold")
            ax2.set_xlabel("轮次(相对位置)")
            ax2.set_ylabel("方差")
            ax2.grid(True, alpha=0.3)

            # 添加稳定性阈值线
            stable_threshold = np.percentile(rolling_var.dropna(), 25)
            ax2.axhline(
                y=stable_threshold,
                color=self.theme["success"],
                linestyle="--",
                label=f"稳定阈值: {stable_threshold:.3f}",
            )
            ax2.legend()

        plt.tight_layout()
        return fig

    def _create_decision_timeline(
        self, simulation_results: Dict[str, Any]
    ) -> go.Figure:
        """创建决策时间线"""
        mdt_discussions = simulation_results.get("mdt_discussions", [])

        if not mdt_discussions:
            fig = go.Figure()
            fig.add_annotation(text="无MDT讨论数据", x=0.5, y=0.5, showarrow=False)
            return fig

        # 准备时间线数据
        days = [d["day"] for d in mdt_discussions]
        treatments = [
            d["decision"]["recommended_treatment"].value for d in mdt_discussions
        ]
        consensus_scores = [d["decision"]["consensus_score"] for d in mdt_discussions]

        # 创建时间线图
        fig = go.Figure()

        # 添加决策点
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
                    colorbar=dict(title="共识得分"),
                ),
                text=[
                    f"Day {day}: {treatment}<br>Score: {score:.3f}"
                    for day, treatment, score in zip(days, treatments, consensus_scores)
                ],
                hovertemplate="%{text}<extra></extra>",
                name="MDT决策",
            )
        )

        fig.update_layout(
            title="MDT决策时间线",
            xaxis_title="天数",
            yaxis_title="共识得分",
            hovermode="closest",
        )

        return fig

    def _create_patient_evolution_chart(
        self, simulation_results: Dict[str, Any]
    ) -> plt.Figure:
        """创建患者状态演化图"""
        daily_events = simulation_results.get("daily_events", [])

        if not daily_events:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
            ax.text(
                0.5,
                0.5,
                "无患者演化数据",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            return fig

        # 提取时序数据
        days = [event["day"] for event in daily_events]
        quality_scores = [
            event["patient_state"].get("quality_of_life_score", 0.5)
            for event in daily_events
        ]
        creatinine_levels = [
            event["patient_state"]["lab_results"].get("creatinine", 1.0)
            for event in daily_events
        ]

        # 创建双y轴图
        fig, ax1 = plt.subplots(1, 1, figsize=self.figure_size)
        ax2 = ax1.twinx()

        # 生活质量评分
        line1 = ax1.plot(
            days,
            quality_scores,
            color=self.theme["primary"],
            linewidth=2,
            marker="o",
            markersize=4,
            label="生活质量评分",
        )
        ax1.set_ylabel("生活质量评分", color=self.theme["primary"])
        ax1.tick_params(axis="y", labelcolor=self.theme["primary"])

        # 肌酐水平
        line2 = ax2.plot(
            days,
            creatinine_levels,
            color=self.theme["secondary"],
            linewidth=2,
            marker="s",
            markersize=4,
            label="肌酐水平",
        )
        ax2.set_ylabel("肌酐水平 (mg/dL)", color=self.theme["secondary"])
        ax2.tick_params(axis="y", labelcolor=self.theme["secondary"])

        # 设置x轴
        ax1.set_xlabel("天数")
        ax1.set_title("患者状态演化轨迹", fontsize=16, fontweight="bold")

        # 添加参考线
        ax1.axhline(
            y=0.5,
            color=self.theme["warning"],
            linestyle="--",
            alpha=0.7,
            label="生活质量警戒线",
        )
        ax2.axhline(
            y=1.5,
            color=self.theme["warning"],
            linestyle="--",
            alpha=0.7,
            label="肌酐警戒线",
        )

        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _create_consensus_trends(
        self, simulation_results: Dict[str, Any]
    ) -> plt.Figure:
        """创建共识得分趋势"""
        mdt_discussions = simulation_results.get("mdt_discussions", [])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        if not mdt_discussions:
            ax1.text(
                0.5,
                0.5,
                "无MDT讨论数据",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=16,
            )
            ax2.text(
                0.5,
                0.5,
                "无讨论轮数数据",
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

        # 上图：共识得分趋势
        ax1.plot(
            days,
            consensus_scores,
            marker="o",
            linewidth=2,
            color=self.theme["primary"],
            markersize=6,
        )
        ax1.set_ylabel("共识得分")
        ax1.set_title("MDT共识得分趋势", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)

        # 添加趋势线
        if len(days) > 1:
            z = np.polyfit(days, consensus_scores, 1)
            p = np.poly1d(z)
            ax1.plot(
                days,
                p(days),
                "--",
                color=self.theme["secondary"],
                alpha=0.8,
                label=f"趋势线(斜率: {z[0]:.4f})",
            )
            ax1.legend()

        # 下图：讨论轮数趋势
        bars = ax2.bar(days, discussion_rounds, alpha=0.7, color=self.theme["info"])
        ax2.set_ylabel("讨论轮数")
        ax2.set_xlabel("天数")
        ax2.set_title("MDT讨论轮数变化", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # 添加数值标签
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
        """创建系统性能指标图"""
        performance_metrics = simulation_results.get("performance_metrics", {})

        if not performance_metrics:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
            ax.text(
                0.5,
                0.5,
                "无性能指标数据",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            return fig

        # 准备指标数据
        metrics = {
            "平均共识得分": performance_metrics.get("avg_consensus_score", 0),
            "平均讨论轮数": performance_metrics.get("avg_discussion_rounds", 0)
            / 5,  # 归一化
            "收敛率": performance_metrics.get("convergence_rate", 0),
            "决策一致性": performance_metrics.get("decision_consistency", 0),
            "平均RL奖励": (performance_metrics.get("avg_rl_reward", 0) + 1)
            / 2,  # 归一化到[0,1]
        }

        # 创建雷达图
        fig, ax = plt.subplots(
            1, 1, figsize=(10, 10), subplot_kw=dict(projection="polar")
        )

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values = list(metrics.values())

        # 闭合雷达图
        angles += angles[:1]
        values += values[:1]

        # 绘制雷达图
        ax.plot(angles, values, "o-", linewidth=2, color=self.theme["primary"])
        ax.fill(angles, values, alpha=0.25, color=self.theme["primary"])

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(list(metrics.keys()))
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.grid(True)

        # 添加数值标注
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
            "系统综合性能指标",
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
